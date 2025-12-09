"""
Dashboard EAF - Prediccion de Temperatura y Composicion Quimica
Punto de entrada principal de la aplicacion Streamlit.

Ejecutar con: streamlit run dashboard/app.py
"""
import sys
from pathlib import Path

# Añadir raiz del proyecto al path para poder importar src/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Añadir directorio dashboard al path para imports internos
DASHBOARD_ROOT = Path(__file__).resolve().parent
if str(DASHBOARD_ROOT) not in sys.path:
    sys.path.insert(0, str(DASHBOARD_ROOT))

import streamlit as st

from utils.cached_loader import load_and_clean_data
from tabs.tab_eda import render_eda_tab
from tabs.tab_temperature import render_temperature_tab
from tabs.tab_chemical import render_chemical_tab


# Configuracion de pagina (debe ser lo primero)
st.set_page_config(
    page_title="Prediccion EAF - Grupo 13",
    page_icon="🏭",
    layout="wide"
)


def init_session_state():
    """Inicializa variables de session_state."""
    defaults = {
        'temp_model': None,
        'temp_features': None,
        'chem_model': None,
        'chem_features': None,
        'chem_target': None
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def main():
    """Funcion principal del dashboard."""
    # Titulo
    st.title("Simulador de Horno de Arco Electrico (EAF)")
    st.markdown("### Optimizacion de Temperatura y Composicion Quimica - Grupo 13")

    # Inicializar estado
    init_session_state()

    # Cargar datos
    df = load_and_clean_data()
    if df is None:
        st.stop()

    # Tabs principales
    tab1, tab2, tab3 = st.tabs([
        "EDA y Resumen del Dataset",
        "Tarea 1: Prediccion de Temperatura",
        "Tarea 2: Prediccion de Composicion Quimica"
    ])

    with tab1:
        render_eda_tab(df)

    with tab2:
        render_temperature_tab(df)

    with tab3:
        render_chemical_tab(df)

    # Footer
    st.divider()
    st.markdown(
        "<div style='text-align: center; color: gray;'>"
        "Desarrollado por Grupo 13 - Curso de Industria 4.0 | 2024-2025"
        "</div>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
