"""
Utilidades del dashboard EAF.
Re-exporta desde src/ y añade wrappers de Streamlit.
"""
from .cached_loader import load_and_clean_data, get_data_path
from src.scripts.evaluation import get_feature_importance
from src.scripts.train_temperature import train_temperature_model, XGBOOST_AVAILABLE
from src.scripts.train_chemical import train_chemical_model

__all__ = [
    'get_data_path',
    'load_and_clean_data',
    'train_temperature_model',
    'train_chemical_model',
    'XGBOOST_AVAILABLE',
    'get_feature_importance'
]
