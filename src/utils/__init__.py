"""
Utilidades compartidas del proyecto EAF.
"""
from .data_loader import get_project_root, get_data_path, load_and_clean_data
from .evaluation import get_feature_importance, calculate_metrics

__all__ = [
    'get_project_root',
    'get_data_path',
    'load_and_clean_data',
    'get_feature_importance',
    'calculate_metrics'
]
