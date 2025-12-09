"""
Tabs del dashboard EAF.
"""
from .tab_eda import render_eda_tab
from .tab_temperature import render_temperature_tab
from .tab_chemical import render_chemical_tab

__all__ = [
    'render_eda_tab',
    'render_temperature_tab',
    'render_chemical_tab'
]
