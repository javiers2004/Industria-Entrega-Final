"""
Indicadores de calidad industrial.
"""
from src.config import CHEMICAL_SPECS, TEMPERATURE_RANGES


def temperature_quality_indicator(temp: float) -> tuple:
    """
    Genera indicador de calidad para temperatura.

    Parameters:
    -----------
    temp : float - Temperatura en grados Celsius

    Returns:
    --------
    tuple: (status, label, description)
        - status: 'success', 'warning', o 'error'
        - label: Etiqueta corta
        - description: Descripcion del estado
    """
    optimal_min = TEMPERATURE_RANGES['optimal_min']
    optimal_max = TEMPERATURE_RANGES['optimal_max']

    if optimal_min <= temp <= optimal_max:
        return "success", "OPTIMA", "Temperatura dentro del rango ideal"
    elif temp > optimal_max:
        return "warning", "ALTA", "Riesgo de sobrecalentamiento / desperdicio de energia"
    else:
        return "error", "BAJA", "Riesgo de retraso / reprocesamiento"


def chemical_spec_indicator(value: float, element: str) -> tuple:
    """
    Genera indicador de especificacion quimica.

    Parameters:
    -----------
    value : float - Valor del elemento quimico
    element : str - Nombre del elemento (valc, valmn, etc.)

    Returns:
    --------
    tuple: (status, label, description)
        - status: 'success', 'error', o 'info'
        - label: Etiqueta corta
        - description: Descripcion del estado
    """
    if element in CHEMICAL_SPECS:
        min_val, max_val = CHEMICAL_SPECS[element]
        if min_val <= value <= max_val:
            return "success", "DENTRO DE ESPECIFICACION", f"Valor entre {min_val} y {max_val}"
        else:
            return "error", "FUERA DE ESPECIFICACION", f"Valor fuera del rango {min_val} - {max_val}"
    return "info", "SIN ESPECIFICACION", "No hay rango definido para este elemento"
