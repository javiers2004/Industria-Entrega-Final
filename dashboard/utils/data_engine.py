"""
Funciones para carga y limpieza de datos del dashboard.
Sin dependencias de src/ - completamente autonomo.
"""
from pathlib import Path
import os
import pandas as pd

from dashboard.config import CHEMICAL_COLUMNS


def get_project_root() -> Path:
    """
    Obtiene la raiz del proyecto desde la ubicacion del dashboard.

    El dashboard esta en: PROJECT_ROOT/dashboard/
    Por lo tanto, la raiz es el padre del directorio dashboard.

    Returns:
        Path a la raiz del proyecto
    """
    # __file__ es dashboard/utils/data_engine.py
    # Subimos 2 niveles: utils -> dashboard -> PROJECT_ROOT
    return Path(__file__).resolve().parent.parent.parent


def get_data_path(file_name: str = "dataset_final_temp.csv") -> Path:
    """
    Busca el archivo de datos usando multiples estrategias.

    Parameters:
        file_name: Nombre del archivo CSV (default: dataset_final_temp.csv)

    Returns:
        Path al archivo de datos
    """
    # Estrategia 1: Relativo al proyecto
    PROJECT_ROOT = get_project_root()
    DATA_PATH = PROJECT_ROOT / "data" / "processed" / file_name

    if DATA_PATH.exists():
        return DATA_PATH

    # Estrategia 2: Desde working directory
    ALT_DATA_PATH = Path(os.getcwd()) / "data" / "processed" / file_name
    if ALT_DATA_PATH.exists():
        return ALT_DATA_PATH

    # Estrategia 3: Buscar en padres del working directory
    for parent in Path(os.getcwd()).parents:
        candidate = parent / "data" / "processed" / file_name
        if candidate.exists():
            return candidate

    return DATA_PATH


def load_and_clean_data(file_name: str = "dataset_final_temp.csv") -> pd.DataFrame:
    """
    Carga y limpia el dataset de acero.
    Convierte columnas quimicas de formato europeo (coma) a float.

    Parameters:
        file_name: Nombre del archivo CSV (default: dataset_final_temp.csv)

    Returns:
        DataFrame limpio

    Raises:
        FileNotFoundError: Si no se encuentra el archivo de datos
    """
    data_path = get_data_path(file_name)

    if not data_path.exists():
        raise FileNotFoundError(f"No se encuentra el dataset en: {data_path}")

    df = pd.read_csv(data_path)

    # Convertir columnas quimicas de formato europeo (coma) a float
    for col in CHEMICAL_COLUMNS:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(',', '.', regex=False)
            df[col] = pd.to_numeric(df[col], errors='coerce')

    return df
