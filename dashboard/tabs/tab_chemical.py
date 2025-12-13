"""
Tab 3: Analisis y Predicciones de Composicion Quimica.
"""
import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, Any

# Importaciones asumidas del proyecto
from src.config import CHEMICAL_TARGETS, CHEMICAL_SPECS
from components.visualizations import plot_prediction_vs_real, plot_feature_importance, plot_distribution # ASUMO plot_distribution
from dashboard.utils.cached_loader import load_single_chemical_result


# --- FUNCIONES DE CÁLCULO ADICIONAL ---

def calculate_spec_metrics(y_test: pd.Series, y_pred: pd.Series, target_name: str) -> Dict[str, float]:
    """Calcula el porcentaje de valores que caen dentro de la especificacion."""
    spec_min, spec_max = CHEMICAL_SPECS.get(target_name, (None, None))

    if spec_min is None or spec_max is None:
        return {'real_in_spec': np.nan, 'pred_in_spec': np.nan}

    # Contar reales
    in_spec_test = ((y_test >= spec_min) & (y_test <= spec_max)).sum()
    real_in_spec = (in_spec_test / len(y_test)) * 100

    # Contar predicciones
    in_spec_pred = ((y_pred >= spec_min) & (y_pred <= spec_max)).sum()
    pred_in_spec = (in_spec_pred / len(y_pred)) * 100

    return {
        'real_in_spec': real_in_spec,
        'pred_in_spec': pred_in_spec,
        'spec_min': spec_min,
        'spec_max': spec_max
    }


# --- FUNCIONES DE RENDERIZADO DEL TAB ---

def _render_summary_metrics(results: Dict[str, Dict[str, Any]]):
    """Renderiza metricas de resumen para todos los targets quimicos."""
    st.subheader("1. Resumen de Desempeño por Componente")

    if not results:
        st.info("No hay resultados de modelo disponibles.")
        return

    # Usar dos filas de columnas para métricas (R2/MAE) y métricas de especificación
    col_R2 = st.columns(min(len(results), 5))
    col_Spec = st.columns(min(len(results), 5))


    all_mae = []
    all_r2 = []
    all_pred_in_spec = []

    st.markdown("##### Métrica Principal y R²")
    for idx, (target, data) in enumerate(results.items()):
        metrics = data['metrics']
        mae = metrics.get('MAE', np.nan)
        r2 = metrics.get('R2', np.nan)

        all_mae.append(mae)
        all_r2.append(r2)

        with col_R2[idx]:
            spec_range = CHEMICAL_SPECS.get(target, (None, None))
            spec_text = f" ({spec_range[0]} - {spec_range[1]})" if all(v is not None for v in spec_range) else ""

            st.markdown(f"**{target.upper()}**{spec_text}")
            st.metric("MAE", f"{mae:.4f}", delta_color="inverse")
            st.metric("R²", f"{r2:.4f}")

    st.markdown("---")
    st.markdown("##### Porcentaje de Predicciones Dentro de Especificación")

    for idx, (target, data) in enumerate(results.items()):
        y_test = data.get('y_test', pd.Series())
        y_pred = data.get('y_pred', pd.Series())

        spec_metrics = calculate_spec_metrics(y_test, y_pred, target)
        pred_in_spec = spec_metrics.get('pred_in_spec', np.nan)
        real_in_spec = spec_metrics.get('real_in_spec', np.nan)

        all_pred_in_spec.append(pred_in_spec)

        # Calcular delta (mejora o empeoramiento de la predicción respecto a la realidad)
        if not np.isnan(pred_in_spec) and not np.isnan(real_in_spec):
             delta = pred_in_spec - real_in_spec
             delta_str = f"{delta:.2f}%"
        else:
             delta_str = None

        with col_Spec[idx]:
            st.markdown(f"**{target.upper()}**")
            st.metric("% Pred. en Especificación", f"{pred_in_spec:.2f}%", delta=delta_str)

    # Mostrar resumen promedio al final
    st.markdown("---")
    avg_mae = np.mean([m for m in all_mae if not np.isnan(m)]) if all_mae else 0
    avg_r2 = np.mean([r for r in all_r2 if not np.isnan(r)]) if all_r2 else 0
    avg_pred_in_spec = np.mean([p for p in all_pred_in_spec if not np.isnan(p)]) if all_pred_in_spec else 0

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("MAE Promedio Global", f"{avg_mae:.4f}", delta_color="inverse")
    with col2:
        st.metric("R² Promedio Global", f"{avg_r2:.4f}")
    with col3:
        st.metric("% Pred. en Especificación Promedio", f"{avg_pred_in_spec:.2f}%")


def _render_prediction_analysis(target_data: Dict[str, Any], target_name: str):
    """Renderiza el grafico de Prediccion vs Real y la Distribucion."""
    st.subheader(f"2. Análisis Detallado de Predicción: {target_name.upper()}")

    y_test = target_data.get('y_test')
    y_pred = target_data.get('y_pred')
    metrics = target_data.get('metrics')

    if y_test is None or y_pred is None or metrics is None:
        st.warning(f"Datos de predicción no disponibles para {target_name}.")
        return

    # Calcular métricas de especificación
    spec_metrics = calculate_spec_metrics(y_test, y_pred, target_name)

    col1, col2, col3, col4 = st.columns([1, 1, 1, 3])
    with col1:
        st.metric("MAE", f"{metrics['MAE']:.4f}", delta_color="inverse")
    with col2:
        st.metric("R²", f"{metrics['R2']:.4f}")
    with col3:
        st.metric("% Pred. en Espec.", f"{spec_metrics['pred_in_spec']:.2f}%")

    spec_range = CHEMICAL_SPECS.get(target_name, (None, None))
    if all(v is not None for v in spec_range):
        with col4:
            st.info(f"Rango de Especificación: **{spec_range[0]}** a **{spec_range[1]}** (Real en Espec.: {spec_metrics['real_in_spec']:.2f}%)")

    # Mostrar gráficos en dos columnas
    col_chart1, col_chart2 = st.columns(2)

    with col_chart1:
        st.markdown("##### Predicción vs Real (Diagrama de Dispersión)")
        fig_pred = plot_prediction_vs_real(
            y_test,
            y_pred,
            title=f"Prediccion vs Real para {target_name.upper()}"
        )
        st.plotly_chart(fig_pred, width='stretch')

    # NUEVA FUNCIONALIDAD: Histograma/KDE de Distribución
    with col_chart2:
        st.markdown("##### Distribución de Valores (Real vs Predicho)")
        # Se necesita plot_distribution en components/visualizations.py
        try:
            fig_dist = plot_distribution(
                y_test,
                y_pred,
                spec_range=(spec_metrics['spec_min'], spec_metrics['spec_max']) if 'spec_min' in spec_metrics else None,
                title=f"Distribución de {target_name.upper()}"
            )
            st.plotly_chart(fig_dist, width='stretch')
        except NameError:
             st.warning("La función `plot_distribution` no está definida o importada. (Asumida en components/visualizations.py)")


def _render_feature_importance(target_data: Dict[str, Any], target_name: str):
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
        st.warning(f"Verifique que haya ejecutado los scripts de entrenamiento y que los archivos de resultados (`results_*.pkl`) existan en `models/chemical_results/`. Targets esperados: {CHEMICAL_TARGETS}")
        return

    results = all_results

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