"""
Configuracion y constantes globales del proyecto EAF.
Compartidas entre src/ y dashboard/.
"""
import os
from typing import Dict, List, Tuple

# Features de Input disponibles en el dataset
INPUT_FEATURES = [
    'total_o2_lance', 'total_gas_lance', 'total_injected_carbon',
    'valc', 'valsi', 'valmn', 'valp', 'vals', 'valcu', 'valcr', 'valmo', 'valni',
    'added_mat_140107', 'added_mat_202007', 'added_mat_202008',
    'added_mat_202039', 'added_mat_202063', 'added_mat_203068',
    'added_mat_203085', 'added_mat_205069', 'added_mat_360258', 'added_mat_705043'
]

# Targets quimicos para prediccion (valores FINALES a predecir)
CHEMICAL_TARGETS = [
    'target_valc', 'target_valmn', 'target_valsi', 'target_valp', 'target_vals',
    'target_valcu', 'target_valcr', 'target_valmo', 'target_valni' # <-- Lista Completa
]

# Targets de temperatura (columnas que deben excluirse de features para temperatura)
TEMPERATURE_TARGETS = ['target_temperature']

# Columnas a excluir siempre como features (IDs, targets, etc.)
EXCLUDE_FROM_FEATURES = [
    'heatid', 'target_temperature',
    'target_valc', 'target_valmn', 'target_valsi', 'target_valp', 'target_vals',
    'target_valcu', 'target_valcr', 'target_valmo', 'target_valni' # <-- Lista Completa
]

# Columnas quimicas que usan coma como separador decimal en el CSV
CHEMICAL_COLUMNS = ['valc', 'valsi', 'valmn', 'valp', 'vals', 'valcu', 'valcr', 'valmo', 'valni']

# Modelos disponibles para entrenamiento
AVAILABLE_MODELS = ['xgboost', 'random_forest', 'linear']

# Mapeo de nombres de modelos (CLI -> display)
MODEL_DISPLAY_NAMES = {
    'xgboost': 'XGBoost Regressor',
    'random_forest': 'Random Forest Regressor',
    'linear': 'Linear Regression'
}

# Rangos de especificacion quimica para valores FINALES (min, max)
CHEMICAL_SPECS = {
    'target_valc': (0.05, 0.50),
    'target_valmn': (0.30, 1.50),
    'target_valsi': (0.10, 0.60),
    'target_valp': (0.001, 0.025),
    'target_vals': (0.001, 0.025),
    'target_valcu': (0.001, 0.030),
    'target_valcr': (0.001, 0.030),
    'target_valmo': (0.001, 0.010),
    'target_valni': (0.001, 0.030)
}

# Rangos de temperatura para indicador de calidad
TEMPERATURE_RANGES = {
    'optimal_min': 1580,
    'optimal_max': 1650,
}

# URL del servicio BentoML (configurable via variable de entorno)
BENTOML_URL: str = os.getenv('BENTOML_URL', 'http://localhost:3000')

# Mapeo de nombres de modelos para UI (inverso de MODEL_DISPLAY_NAMES)
UI_MODEL_NAMES: Dict[str, str] = {
    'Linear Regression': 'linear',
    'Random Forest Regressor': 'random_forest',
    'XGBoost Regressor': 'xgboost'
}

# Hiperparametros por defecto
DEFAULT_HYPERPARAMS = {
    'n_estimators': 100,
    'max_depth': 6,
    'learning_rate': 0.1,
    'test_size': 0.2,
    'random_state': 42
}

# Datasets disponibles para EDA
EDA_DATASETS: Dict[str, str] = {
    "Temperatura": "dataset_final_temp.csv",
    "Quimica": "dataset_final_chemical.csv"
}