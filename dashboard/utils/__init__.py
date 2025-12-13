"""
Utilidades del dashboard EAF.
Re-exporta desde src/ y añade wrappers de Streamlit.
"""
# Importa solo las funciones de carga desde cached_loader
from .cached_loader import load_and_clean_data, get_data_path, load_single_chemical_result
# Eliminadas las importaciones de src.scripts.evaluation, train_temperature, y train_chemical

__all__ = [
    'get_data_path',
    'load_and_clean_data',
    'load_single_chemical_result' # Añadida la nueva funcion de carga
]