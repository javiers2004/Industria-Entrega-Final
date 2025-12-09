"""
Tab 3: Analisis y Predicciones de Composicion Quimica.
"""
import streamlit as st
import pandas as pd
import numpy as np

# Importaciones asumidas del proyecto
from src.config import CHEMICAL_TARGETS, CHEMICAL_SPECS
from components.visualizations import plot_prediction_vs_real, plot_feature_importance
# NOTA: Se asume que se importa la funcion load_chemical_results (implementada abajo para simular)
# from dashboard.utils.cached_loader import load_chemical_results


# --- FUNCION DE CARGA DUMMY (DEBE SER REEMPLAZADA) ---
# Esta funcion simula la carga usando los targets y rangos reales, pero con datos aleatorios.
# DEBE ser reemplazada por la lógica real de carga de resultados pre-calculados.
def load_chemical_results():
    """Simula la carga de resultados del modelo de composicion quimica (Usando targets reales)."""
    st.info("🚨 ADVERTENCIA: Usando datos de modelo simulados para targets químicos.")

    results = {}

    # Usar los targets reales definidos en src.config
    targets_to_simulate = CHEMICAL_TARGETS

    for target in targets_to_simulate:
        # Generar datos simulados para y_test y y_pred (simulando variabilidad real)
        np.random.seed(hash(target) % 1000) # Semilla diferente por target
        n_samples = 100

        # Usar el rango de especificacion para simular valores reales cercanos
        spec_min, spec_max = CHEMICAL_SPECS.get(target, (0, 1))

        # Simular y_test dentro de un rango cercano a las especificaciones
        y_test = np.random.uniform(spec_min, spec_max, n_samples)
        # y_pred simula una prediccion con ruido bajo
        y_pred = y_test + np.random.normal(0, (spec_max - spec_min) * 0.05, n_samples)

        # Simular feature importance (estructura requerida por plot_feature_importance)
        features = [
            'total_o2_lance', 'total_gas_lance', 'valc_init', 'added_mat_140107',
            'added_mat_202007', 'added_mat_360258'
        ]
        importance = np.random.uniform(0.01, 0.5, len(features))
        importance_df = pd.DataFrame({'Feature': features, 'Importance': importance})
        importance_df = importance_df.sort_values('Importance', ascending=True)

        # Simular métricas (calculadas a partir de los datos simulados)
        mae = np.mean(np.abs(y_test - y_pred))
        r2 = 1 - np.sum((y_test - y_pred)**2) / np.sum((y_test - np.mean(y_test))**2)

        results[target] = {
            'y_test': pd.Series(y_test),
            'y_pred': pd.Series(y_pred),
            'importance_df': importance_df,
            'metrics': {'MAE': mae, 'R2': r2}
        }

    return results

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
    st.plotly_chart(fig_pred, use_container_width=True)


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
    st.plotly_chart(fig_imp, use_container_width=True)


def render_chemical_tab():
    """
    Funcion principal para renderizar el tab de Composicion Quimica.
    """
    st.title("🧪 Predicción de Composición Química")
    st.markdown("Esta sección presenta el análisis de los resultados de los modelos entrenados para predecir la composición química (elementos como **C, Mn, Si, P, S**).")

    # 1. Cargar Resultados (Usando la funcion dummy/simulada que debe ser reemplazada)
    results = load_chemical_results()

    if not results:
        st.error("No se pudieron cargar los resultados del modelo químico. Verifique la implementación de la función de carga.")
        return

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
            st.error(f"Error al obtener los datos para el target: {selected_target}")