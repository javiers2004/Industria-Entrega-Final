"""
Script para construir el dataset final a partir de los datos raw.

Este script realiza:
1. Carga y estandarizacion de los archivos CSV
2. Agregacion de series temporales (gases, inyecciones)
3. Pivotado de materiales agregados
4. Fusion con el target (temperatura final)
5. Limpieza y guardado del dataset final

Uso:
    python -m src.features.build_features

Requisitos:
    - Los datos raw deben estar en data/raw/
    - Ejecutar primero: python -m src.data.download_kaggle
"""

import os
from pathlib import Path
from typing import Optional

import pandas as pd

from src.logging_config import get_logger

logger = get_logger(__name__)


def get_project_root() -> Path:
    """Obtiene la raiz del proyecto."""
    return Path(__file__).parent.parent.parent


def load_standardized(filepath: Path) -> pd.DataFrame:
    """
    Carga un CSV y estandariza los nombres de columnas.

    Args:
        filepath: Ruta al archivo CSV

    Returns:
        DataFrame con columnas en minusculas y sin espacios
    """
    df = pd.read_csv(filepath, low_memory=False)
    df.columns = df.columns.str.lower().str.strip()

    # Convertir comas decimales a puntos (formato europeo -> americano)
    for col in df.select_dtypes(include=['object']).columns:
        if df[col].astype(str).str.match(r'^-?\d+,\d+$').any():
            df[col] = df[col].astype(str).str.replace(',', '.', regex=False)

    return df


def aggregate_gas_data(df_gas: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega los datos de gas lance por colada.

    Args:
        df_gas: DataFrame con datos de eaf_gaslance_mat.csv

    Returns:
        DataFrame agregado por heatid con totales de O2 y gas
    """
    # Limpiar tipos
    cols_gas = ['o2_amount', 'gas_amount']
    for col in cols_gas:
        df_gas[col] = pd.to_numeric(df_gas[col], errors='coerce')

    # Agregar por colada
    grp_gas = df_gas.groupby('heatid').agg({
        'o2_amount': 'max',
        'gas_amount': 'max'
    }).rename(columns={
        'o2_amount': 'total_o2_lance',
        'gas_amount': 'total_gas_lance'
    })

    return grp_gas


def aggregate_injection_data(df_inj: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega los datos de inyecciones de carbon por colada.

    Args:
        df_inj: DataFrame con datos de inj_mat.csv

    Returns:
        DataFrame agregado por heatid con total de carbon inyectado
    """
    df_inj['inj_amount_carbon'] = pd.to_numeric(df_inj['inj_amount_carbon'], errors='coerce')

    grp_inj = df_inj.groupby('heatid').agg({
        'inj_amount_carbon': 'max'
    }).rename(columns={'inj_amount_carbon': 'total_injected_carbon'})

    return grp_inj


def pivot_materials(df_ladle: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """
    Pivota los materiales agregados, seleccionando los top_n mas frecuentes.

    Args:
        df_ladle: DataFrame con datos de ladle_tapping.csv
        top_n: Numero de materiales mas frecuentes a incluir

    Returns:
        DataFrame pivotado con columnas por codigo de material
    """
    df_ladle['charge_amount'] = pd.to_numeric(df_ladle['charge_amount'], errors='coerce')

    # Seleccionar top materiales por frecuencia
    top_materials = df_ladle['mat_code'].value_counts().head(top_n).index
    df_ladle_filtered = df_ladle[df_ladle['mat_code'].isin(top_materials)]

    # Pivotar
    pivot_ladle = df_ladle_filtered.pivot_table(
        index='heatid',
        columns='mat_code',
        values='charge_amount',
        aggfunc='sum',
        fill_value=0
    ).add_prefix('added_mat_')

    return pivot_ladle


def get_final_chemical_composition(df_chem_final: pd.DataFrame) -> pd.DataFrame:
    """
    Obtiene la composicion quimica final de cada colada.

    Args:
        df_chem_final: DataFrame con datos de eaf_final_chemical_measurements.csv

    Returns:
        DataFrame con heatid y targets quimicos (target_valc, target_valmn, etc.)
    """
    # Elementos quimicos a extraer como targets
    chemical_elements = ['valc', 'valmn', 'valsi', 'valp', 'vals']

    # Verificar que existan las columnas
    available_elements = [col for col in chemical_elements if col in df_chem_final.columns]

    if not available_elements:
        logger.warning("No se encontraron columnas de elementos quimicos en el archivo")
        return pd.DataFrame()

    # Seleccionar heatid y elementos quimicos
    cols_to_select = ['heatid'] + available_elements
    df_targets = df_chem_final[cols_to_select].copy()

    # Convertir a numerico
    for col in available_elements:
        df_targets[col] = pd.to_numeric(df_targets[col], errors='coerce')

    # Renombrar con prefijo target_
    rename_dict = {col: f'target_{col}' for col in available_elements}
    df_targets = df_targets.rename(columns=rename_dict)

    # Eliminar duplicados por heatid (tomar el ultimo registro)
    df_targets = df_targets.drop_duplicates(subset=['heatid'], keep='last')

    logger.info(f"Targets quimicos extraidos: {len(df_targets)} coladas, {len(available_elements)} elementos")

    return df_targets


def get_final_temperature(df_temp: pd.DataFrame) -> pd.DataFrame:
    """
    Obtiene la temperatura final (al vaciado) de cada colada.

    Args:
        df_temp: DataFrame con datos de eaf_temp.csv

    Returns:
        DataFrame con heatid y target_temperature
    """
    # Detectar columnas automaticamente
    cols_temp = [c for c in df_temp.columns if 'temp' in c and 'time' not in c]
    cols_time = [c for c in df_temp.columns if 'time' in c or 'date' in c]

    col_temp_name = cols_temp[0] if cols_temp else 'temp'
    col_time_name = cols_time[0] if cols_time else 'datetime'

    # Limpiar tipos
    df_temp[col_temp_name] = pd.to_numeric(df_temp[col_temp_name], errors='coerce')
    df_temp[col_time_name] = pd.to_datetime(df_temp[col_time_name], errors='coerce')

    # Obtener la ULTIMA medicion (temperatura al vaciado)
    df_target = df_temp.sort_values(col_time_name).groupby('heatid').tail(1)

    # Seleccionar solo ID y temperatura
    df_target = df_target[['heatid', col_temp_name]].rename(
        columns={col_temp_name: 'target_temperature'}
    )

    return df_target


def build_master_dataset(raw_data_dir: Path) -> pd.DataFrame:
    """
    Construye el dataset maestro combinando todas las fuentes de datos.

    Args:
        raw_data_dir: Ruta al directorio con los datos raw

    Returns:
        DataFrame maestro con inputs
    """
    logger.info("Cargando archivos...")

    # Cargar archivos necesarios
    df_gas = load_standardized(raw_data_dir / "eaf_gaslance_mat.csv")
    df_inj = load_standardized(raw_data_dir / "inj_mat.csv")
    df_ladle = load_standardized(raw_data_dir / "ladle_tapping.csv")
    df_chem_initial = load_standardized(raw_data_dir / "lf_initial_chemical_measurements.csv")

    logger.info("Agregando series temporales...")

    # Agregar datos
    grp_gas = aggregate_gas_data(df_gas)
    grp_inj = aggregate_injection_data(df_inj)

    logger.info("Pivotando materiales...")
    pivot_ladle = pivot_materials(df_ladle)

    logger.info("Fusionando dataset maestro...")

    # Dataset base: mediciones quimicas iniciales
    df_master = df_chem_initial.copy()

    # Merges (left joins)
    df_master = df_master.merge(grp_gas, on='heatid', how='left')
    df_master = df_master.merge(grp_inj, on='heatid', how='left')
    df_master = df_master.merge(pivot_ladle, on='heatid', how='left')

    # Rellenar nulos tecnicos
    cols_to_fix = ['total_o2_lance', 'total_gas_lance', 'total_injected_carbon']
    df_master[cols_to_fix] = df_master[cols_to_fix].fillna(0)

    logger.info(f"Dataset maestro (inputs): {df_master.shape}")

    return df_master


def add_chemical_targets(df_master: pd.DataFrame, raw_data_dir: Path) -> pd.DataFrame:
    """
    Agrega las variables target de composicion quimica al dataset maestro.

    Args:
        df_master: DataFrame con inputs
        raw_data_dir: Ruta al directorio con los datos raw

    Returns:
        DataFrame final con inputs y targets quimicos
    """
    logger.info("Agregando targets de composicion quimica...")

    df_chem_final = load_standardized(raw_data_dir / "eaf_final_chemical_measurements.csv")
    df_targets = get_final_chemical_composition(df_chem_final)

    if df_targets.empty:
        raise ValueError("No se pudieron extraer los targets quimicos")

    # Merge (inner join - solo coladas con datos completos)
    df_final = df_master.merge(df_targets, on='heatid', how='inner')

    # Limpieza final
    cols_drop = ['datetime', 'positionrow', 'filter_key_date', 'measure_time']
    df_final = df_final.drop(columns=[c for c in cols_drop if c in df_final.columns])

    # Eliminar filas donde todos los targets son nulos
    target_cols = [c for c in df_final.columns if c.startswith('target_')]
    df_final = df_final.dropna(subset=target_cols, how='all')

    # Rellenar nulos en inputs con 0
    df_final = df_final.fillna(0)

    logger.info(f"Dataset quimico final: {df_final.shape}")

    return df_final


def add_target(df_master: pd.DataFrame, raw_data_dir: Path) -> pd.DataFrame:
    """
    Agrega la variable target (temperatura final) al dataset maestro.

    Args:
        df_master: DataFrame con inputs
        raw_data_dir: Ruta al directorio con los datos raw

    Returns:
        DataFrame final con inputs y target
    """
    logger.info("Agregando variable target (temperatura)...")

    df_temp = load_standardized(raw_data_dir / "eaf_temp.csv")
    df_target = get_final_temperature(df_temp)

    # Merge (inner join - solo coladas con datos completos)
    df_final = df_master.merge(df_target, on='heatid', how='inner')

    # Limpieza final
    cols_drop = ['datetime', 'positionrow', 'filter_key_date', 'measure_time']
    df_final = df_final.drop(columns=[c for c in cols_drop if c in df_final.columns])

    # Eliminar nulos en el target
    df_final = df_final.dropna(subset=['target_temperature'])

    # Rellenar nulos en inputs con 0
    df_final = df_final.fillna(0)

    logger.info(f"Dataset final: {df_final.shape}")

    return df_final


def build_temperature_dataset(raw_data_dir: Path, processed_data_dir: Path, force: bool = False) -> Optional[Path]:
    """
    Construye el dataset para prediccion de temperatura.

    Args:
        raw_data_dir: Ruta al directorio con los datos raw
        processed_data_dir: Ruta al directorio de salida
        force: Si es True, reconstruye aunque exista el archivo

    Returns:
        Path al archivo de dataset o None si ya existe y force=False
    """
    output_file = processed_data_dir / "dataset_final_temp.csv"

    # Verificar si ya existe
    if output_file.exists() and not force:
        logger.info(f"Dataset de temperatura ya existe: {output_file}")
        return output_file

    logger.info("=" * 50)
    logger.info("CONSTRUYENDO DATASET DE TEMPERATURA")
    logger.info("=" * 50)

    df_master = build_master_dataset(raw_data_dir)
    df_final = add_target(df_master, raw_data_dir)

    # Guardar
    df_final.to_csv(output_file, index=False)
    logger.info(f"Dataset de temperatura guardado en: {output_file}")

    # Mostrar resumen
    logger.info(f"Filas: {len(df_final)}, Columnas: {len(df_final.columns)}")

    return output_file


def build_chemical_dataset(raw_data_dir: Path, processed_data_dir: Path, force: bool = False) -> Optional[Path]:
    """
    Construye el dataset para prediccion de composicion quimica.

    Args:
        raw_data_dir: Ruta al directorio con los datos raw
        processed_data_dir: Ruta al directorio de salida
        force: Si es True, reconstruye aunque exista el archivo

    Returns:
        Path al archivo de dataset o None si ya existe y force=False
    """
    output_file = processed_data_dir / "dataset_final_chemical.csv"

    # Verificar si ya existe
    if output_file.exists() and not force:
        logger.info(f"Dataset quimico ya existe: {output_file}")
        return output_file

    logger.info("=" * 50)
    logger.info("CONSTRUYENDO DATASET DE COMPOSICION QUIMICA")
    logger.info("=" * 50)

    df_master = build_master_dataset(raw_data_dir)
    df_final = add_chemical_targets(df_master, raw_data_dir)

    # Guardar
    df_final.to_csv(output_file, index=False)
    logger.info(f"Dataset quimico guardado en: {output_file}")

    # Mostrar resumen
    logger.info(f"Filas: {len(df_final)}, Columnas: {len(df_final.columns)}")
    target_cols = [c for c in df_final.columns if c.startswith('target_')]
    logger.info(f"Targets: {target_cols}")

    return output_file


def build_features(force: bool = False, temp_only: bool = False, chem_only: bool = False) -> tuple:
    """
    Pipeline completo para construir los datasets finales.

    Args:
        force: Si es True, reconstruye aunque existan los archivos
        temp_only: Si es True, solo genera dataset de temperatura
        chem_only: Si es True, solo genera dataset quimico

    Returns:
        Tuple con (path_temp, path_chemical) o solo uno si se especifica *_only
    """
    project_root = get_project_root()
    raw_data_dir = project_root / "data" / "raw"
    processed_data_dir = project_root / "data" / "processed"

    # Verificar que existan los datos raw
    if not raw_data_dir.exists():
        raise FileNotFoundError(
            f"No existe el directorio {raw_data_dir}. "
            "Ejecuta primero: python -m src.data.download_kaggle"
        )

    # Crear directorio de salida
    processed_data_dir.mkdir(parents=True, exist_ok=True)

    path_temp = None
    path_chem = None

    # Construir dataset de temperatura
    if not chem_only:
        path_temp = build_temperature_dataset(raw_data_dir, processed_data_dir, force)

    # Construir dataset quimico
    if not temp_only:
        path_chem = build_chemical_dataset(raw_data_dir, processed_data_dir, force)

    logger.info("=" * 50)
    logger.info("CONSTRUCCION COMPLETADA")
    logger.info("=" * 50)

    if path_temp:
        logger.info(f"  - Temperatura: {path_temp}")
    if path_chem:
        logger.info(f"  - Quimico: {path_chem}")

    return (path_temp, path_chem)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Construir datasets de features para el proyecto EAF")
    parser.add_argument("--force", "-f", action="store_true", help="Forzar reconstruccion de los datasets")
    parser.add_argument("--temp-only", action="store_true", help="Solo generar dataset de temperatura")
    parser.add_argument("--chem-only", action="store_true", help="Solo generar dataset de composicion quimica")

    args = parser.parse_args()

    if args.temp_only and args.chem_only:
        parser.error("No se puede usar --temp-only y --chem-only a la vez")

    build_features(force=args.force, temp_only=args.temp_only, chem_only=args.chem_only)
