"""
Script para descargar los datos del horno de arco electrico desde Kaggle.

Uso:
    python -m src.data.download_kaggle

Requisitos:
    - kagglehub instalado (pip install kagglehub)
    - Credenciales de Kaggle configuradas
"""

import os
import shutil
from pathlib import Path
from typing import List

import kagglehub

from src.logging_config import get_logger

logger = get_logger(__name__)


# Archivos esperados del dataset
ARCHIVOS_ESPERADOS = [
    "eaf_transformer.csv",
    "basket_charged.csv",
    "eaf_temp.csv",
    "eaf_final_chemical_measurements.csv",
    "eaf_added_materials.csv",
    "inj_mat.csv",
    "eaf_gaslance_mat.csv",
    "lf_initial_chemical_measurements.csv",
    "ladle_tapping.csv",
    "lf_added_materials.csv",
    "ferro.csv"
]

# Dataset de Kaggle
KAGGLE_DATASET = "yuriykatser/industrial-data-from-the-arc-furnace"


def get_project_root() -> Path:
    """Obtiene la raiz del proyecto."""
    # Este archivo esta en src/data/, subimos dos niveles
    return Path(__file__).parent.parent.parent


def download_data(force: bool = False) -> Path:
    """
    Descarga los datos de Kaggle y los copia al directorio data/raw/.

    Args:
        force: Si es True, sobreescribe los archivos existentes.

    Returns:
        Path al directorio data/raw/ con los datos descargados.
    """
    project_root = get_project_root()
    raw_data_dir = project_root / "data" / "raw"

    # Crear directorio si no existe
    raw_data_dir.mkdir(parents=True, exist_ok=True)

    # Verificar si ya existen los datos
    archivos_existentes = [f for f in ARCHIVOS_ESPERADOS if (raw_data_dir / f).exists()]

    if len(archivos_existentes) == len(ARCHIVOS_ESPERADOS) and not force:
        logger.info(f"Los datos ya existen en {raw_data_dir}")
        logger.info("Usa force=True para volver a descargar.")
        return raw_data_dir

    logger.info(f"Descargando dataset desde Kaggle: {KAGGLE_DATASET}")
    logger.info("Esto puede tardar unos minutos...")

    # Descargar usando kagglehub
    kaggle_path = kagglehub.dataset_download(KAGGLE_DATASET)
    kaggle_path = Path(kaggle_path)

    logger.info(f"Dataset descargado en: {kaggle_path}")
    logger.info(f"Copiando archivos a: {raw_data_dir}")

    # Copiar cada archivo al directorio del proyecto
    archivos_copiados = 0
    archivos_no_encontrados = []

    for archivo in ARCHIVOS_ESPERADOS:
        origen = kaggle_path / archivo
        destino = raw_data_dir / archivo

        if origen.exists():
            shutil.copy2(origen, destino)
            size_mb = destino.stat().st_size / (1024 * 1024)
            logger.info(f"  - {archivo} ({size_mb:.2f} MB)")
            archivos_copiados += 1
        else:
            archivos_no_encontrados.append(archivo)
            logger.warning(f"  - {archivo} (NO ENCONTRADO)")

    logger.info("-" * 50)
    logger.info(f"Archivos copiados: {archivos_copiados}/{len(ARCHIVOS_ESPERADOS)}")

    if archivos_no_encontrados:
        logger.warning(f"Archivos no encontrados: {archivos_no_encontrados}")

    logger.info(f"Datos guardados en: {raw_data_dir}")

    return raw_data_dir


def verificar_datos() -> bool:
    """
    Verifica que todos los archivos necesarios existan.

    Returns:
        True si todos los archivos existen, False en caso contrario.
    """
    project_root = get_project_root()
    raw_data_dir = project_root / "data" / "raw"

    if not raw_data_dir.exists():
        logger.warning(f"El directorio {raw_data_dir} no existe.")
        logger.info("Ejecuta download_data() para descargar los datos.")
        return False

    archivos_faltantes = []
    for archivo in ARCHIVOS_ESPERADOS:
        if not (raw_data_dir / archivo).exists():
            archivos_faltantes.append(archivo)

    if archivos_faltantes:
        logger.warning(f"Faltan {len(archivos_faltantes)} archivos:")
        for archivo in archivos_faltantes:
            logger.warning(f"  - {archivo}")
        return False

    logger.info(f"Todos los archivos ({len(ARCHIVOS_ESPERADOS)}) estan disponibles en {raw_data_dir}")
    return True


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Descargar datos de Kaggle para el proyecto EAF")
    parser.add_argument("--force", "-f", action="store_true", help="Forzar descarga aunque existan los archivos")
    parser.add_argument("--verify", "-v", action="store_true", help="Solo verificar que existan los archivos")

    args = parser.parse_args()

    if args.verify:
        verificar_datos()
    else:
        download_data(force=args.force)
