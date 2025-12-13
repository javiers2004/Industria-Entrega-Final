"""
Tab 3: Analisis y Predicciones de Composicion Quimica.
"""
import streamlit as st
import pandas as pd
import numpy as np

# Importaciones asumidas del proyecto
from src.config import CHEMICAL_TARGETS, CHEMICAL_SPECS
from components.visualizations import plot_prediction_vs_real, plot_feature_importance
# ACTUALIZACIÓN: Importar la función real de carga del .pkl
from dashboard.utils.cached_loader import load_single_chemical_result


# --- FUNCIONES DE CARGA DUMMY ELIMINADAS ---
# La funcion load_chemical_results (la que generaba datos aleatorios) ha sido ELIMINADA.


# --- FUNCIONES DE RENDERIZADO DEL TAB ---

def _render_summary_metrics(results: dict):
    """Renderiza metricas de resumen para todos los targets quimicos."""
    st.subheader("1. Resumen de Desempeño por Componente")

    if not results:
        st.info("No hay resultados de modelo disponibles.")
        return

    # Usar una columna para cada target, limitado a 5 columnas
    cols = st.columns(min(len(results), 5))

    all_mae = []
    all_r2 = []

    for idx, (target, data) in enumerate(results.items()):
        metrics = data['metrics']
        mae = metrics.get('MAE', np.nan)
        r2 = metrics.get('R2', np.nan)

        all_mae.append(mae)
        all_r2.append(r2)

        with cols[idx]:
            # Mostrar el rango de especificacion si existe
            spec_range = CHEMICAL_SPECS.get(target, (None, None))
            spec_text = f" ({spec_range[0]} - {spec_range[1]})" if all(v is not None for v in spec_range) else ""

            st.markdown(f"#### {target.upper()}{spec_text}")
            st.metric("MAE", f"{mae:.4f}", delta_color="inverse")
            st.metric("R²", f"{r2:.4f}")

    # Mostrar resumen promedio al final
    st.markdown("---")
    avg_mae = np.mean([m for m in all_mae if not np.isnan(m)]) if all_mae else 0
    avg_r2 = np.mean([r for r in all_r2 if not np.isnan(r)]) if all_r2 else 0

    col1, col2 = st.columns(2)
    with col1:
        st.metric("MAE Promedio Global", f"{avg_mae:.4f}", delta_color="inverse")
    with col2:
        st.metric("R² Promedio Global", f"{avg_r2:.4f}")


def _render_prediction_analysis(target_data: dict, target_name: str):
    """Renderiza el grafico de Prediccion vs Real para el target seleccionado."""
    st.subheader(f"2. Análisis de Predicción (Predicho vs Real): {target_name.upper()}")

    y_test = target_data.get('y_test')
    y_pred = target_data.get('y_pred')
    metrics = target_data.get('metrics')

    if y_test is None or y_pred is None:
        st.warning(f"Datos de predicción no disponibles para {target_name}.")
        return

    # Mostrar métricas específicas del target
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        st.metric("MAE", f"{metrics['MAE']:.4f}", delta_color="inverse")
    with col2:
        st.metric("R²", f"{metrics['R2']:.4f}")

    # Mostrar especificaciones
    spec_range = CHEMICAL_SPECS.get(target_name, (None, None))
    if all(v is not None for v in spec_range):
        with col3:
            st.info(f"Rango de Especificación: **{spec_range[0]}** a **{spec_range[1]}**")

    # Generar y mostrar el gráfico
    fig_pred = plot_prediction_vs_real(
        y_test,
        y_pred,
        title=f"Prediccion vs Real para {target_name.upper()}"
    )
    # Corrección: Usar width='stretch' en lugar de use_container_width=True
    st.plotly_chart(fig_pred, width='stretch')


def _render_feature_importance(target_data: dict, target_name: str):
    """Renderiza el grafico de importancia de caracteristicas para el target seleccionado."""
    st.subheader(f"3. Top 15 Variables más Importantes: {target_name.upper()}")

    importance_df = target_data.get('importance_df')

    if importance_df is None or importance_df.empty:
        st.warning(f"Datos de importancia de variables no disponibles para {target_name}.")
        return

    # Generar y mostrar el gráfico
    fig_imp = plot_feature_importance(
        importance_df,
        title=f"Importancia de Variables para {target_name.upper()}"
    )
    # Corrección: Usar width='stretch' en lugar de use_container_width=True
    st.plotly_chart(fig_imp, width='stretch')


def render_chemical_tab():
    """
    Funcion principal para renderizar el tab de Composicion Quimica.
    """
    st.title("🧪 Predicción de Composición Química")
    st.markdown("Esta sección presenta el análisis de los resultados de los modelos entrenados para predecir la composición química (elementos como **C, Mn, Si, P, S**).")

    # 1. Cargar Resultados (Lógica de carga de datos REALES)
    st.info("Cargando resultados de modelos químicos pre-calculados...")

    all_results = {}
    # Iterar sobre los targets definidos en src/config.py
    for target in CHEMICAL_TARGETS:
        # Usar la funcion cacheada para cargar los resultados de cada target
        result = load_single_chemical_result(target)
        if result:
            all_results[target] = result

    # COMPROBACIÓN CRÍTICA: Si no hay resultados, mostrar error y salir.
    if not all_results:
        st.error("🚨 ERROR: No se pudieron cargar los resultados de ningún modelo químico.")
        st.warning("Verifique que haya ejecutado los scripts de entrenamiento (ej: `python -m src.scripts.train_chemical --target valc`) y que los archivos de resultados (`results_*.pkl`) existan en `models/chemical_results/`.")
        return

    results = all_results # Usamos 'results' para mantener la compatibilidad con el resto de la función

    # 2. Resumen de Métricas Globales
    _render_summary_metrics(results)
    st.divider()

    # 3. Selector de Componente Químico
    available_targets = sorted(list(results.keys()))

    if not available_targets:
        st.warning("No se encontraron targets químicos en los resultados cargados.")
        return

    selected_target = st.selectbox(
        "Selecciona el Componente Químico para el Análisis Detallado:",
        options=available_targets,
        key='chemical_target_selector'
    )
    st.divider()

    # 4. Análisis Detallado del Target Seleccionado
    if selected_target:
        target_data = results.get(selected_target)
        if target_data:
            _render_prediction_analysis(target_data, selected_target)
            st.divider()
            _render_feature_importance(target_data, selected_target)
        else:
            # Esto solo debería ocurrir si un target se carga en la lista, pero luego falla
            st.error(f"Error al obtener los datos para el target: {selected_target}")