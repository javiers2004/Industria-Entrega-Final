"""
Tab 3: Analisis y Predicciones de Composicion Quimica.
Incluye secciones de Entrenamiento y Evaluacion.
"""
import streamlit as st
import pandas as pd
import numpy as np
import os
import json
import joblib
from typing import Dict, Any, List
from sklearn.model_selection import train_test_split

from dashboard.config import (
    CHEMICAL_TARGETS, CHEMICAL_SPECS, MODEL_DISPLAY_NAMES,
    UI_MODEL_NAMES, DEFAULT_HYPERPARAMS
)
from dashboard.components.visualizations import (
    plot_prediction_vs_real, plot_feature_importance, plot_distribution
)
from dashboard.utils.cached_loader import load_single_chemical_result
from dashboard.utils.data_engine import get_project_root
from dashboard.utils.chemical_engine import (
    train_chemical_model,
    load_chemical_data,
    get_chemical_features,
    get_trained_chemical_models,
    load_chemical_model,
    load_chemical_model_metadata
)
from dashboard.utils.model_engine import get_feature_importance, calculate_metrics


# =============================================================================
# Funciones Auxiliares
# =============================================================================

def calculate_spec_metrics(y_test: pd.Series, y_pred: pd.Series, target_name: str) -> Dict[str, float]:
    """Calcula el porcentaje de valores que caen dentro de la especificacion."""
    spec_min, spec_max = CHEMICAL_SPECS.get(target_name, (None, None))

    if spec_min is None or spec_max is None:
        return {'real_in_spec': np.nan, 'pred_in_spec': np.nan}

    # Convertir a numpy arrays para operaciones
    y_test_arr = np.asarray(y_test)
    y_pred_arr = np.asarray(y_pred)

    # Contar reales
    in_spec_test = ((y_test_arr >= spec_min) & (y_test_arr <= spec_max)).sum()
    real_in_spec = (in_spec_test / len(y_test_arr)) * 100

    # Contar predicciones
    in_spec_pred = ((y_pred_arr >= spec_min) & (y_pred_arr <= spec_max)).sum()
    pred_in_spec = (in_spec_pred / len(y_pred_arr)) * 100

    return {
        'real_in_spec': real_in_spec,
        'pred_in_spec': pred_in_spec,
        'spec_min': spec_min,
        'spec_max': spec_max
    }


# =============================================================================
# Funcion Principal del Tab
# =============================================================================

def render_chemical_tab():
    """
    Funcion principal para renderizar el tab de Composicion Quimica.
    Organizado en sub-tabs de Entrenamiento y Evaluacion.
    """
    st.header("Tarea 2: Prediccion de Composicion Quimica")
    st.markdown("Entrena y evalua modelos para predecir la composicion quimica (C, Mn, Si, P, S).")

    # Crear sub-tabs
    tab_train, tab_eval = st.tabs([
        "Entrenamiento",
        "Evaluacion"
    ])

    with tab_train:
        _render_training_section()

    with tab_eval:
        _render_evaluation_section()


# =============================================================================
# Sub-tab: Entrenamiento
# =============================================================================

def _render_training_section():
    """Seccion de entrenamiento de modelos quimicos."""
    st.subheader("Entrenamiento de Modelos Quimicos")

    # Cargar datos para obtener features disponibles
    try:
        df = load_chemical_data()
    except FileNotFoundError as e:
        st.error(f"No se pudo cargar el dataset quimico: {e}")
        return

    st.markdown("#### Configuracion del Modelo")

    # 1. Selector de Target Quimico (diferencia clave con temperatura)
    col_target, col_model = st.columns(2)

    with col_target:
        selected_target = st.selectbox(
            "Target Quimico a Predecir:",
            options=CHEMICAL_TARGETS,
            format_func=lambda x: f"{x.upper()} - {_get_target_description(x)}",
            key='chem_target_select'
        )

        # Mostrar especificacion del target
        if selected_target in CHEMICAL_SPECS:
            spec_min, spec_max = CHEMICAL_SPECS[selected_target]
            st.info(f"Especificacion: {spec_min} - {spec_max}")

    with col_model:
        # 2. Selector de tipo de modelo
        ui_model_options = list(UI_MODEL_NAMES.keys())
        chem_model_ui = st.selectbox(
            "Tipo de Modelo:",
            options=ui_model_options,
            key='chem_model_select'
        )
        model_type = UI_MODEL_NAMES[chem_model_ui]

    # 3. Selector de features
    available_features = get_chemical_features(df, selected_target)

    chem_selected_features = st.multiselect(
        "Features de Entrada:",
        options=available_features,
        default=available_features[:10] if len(available_features) >= 10 else available_features,
        key='chem_features_select',
        help="Selecciona las variables predictoras. El target seleccionado se excluye automaticamente."
    )

    # 4. Hiperparametros condicionales
    st.markdown("##### Hiperparametros")
    n_estimators = 100
    max_depth = 6
    learning_rate = 0.1

    col_hp1, col_hp2, col_hp3 = st.columns(3)

    with col_hp1:
        if model_type in ['random_forest', 'xgboost']:
            n_estimators = st.slider(
                "Numero de Estimadores:",
                min_value=50, max_value=500, value=100, step=50,
                key='chem_n_estimators'
            )

    with col_hp2:
        if model_type in ['random_forest', 'xgboost']:
            max_depth = st.slider(
                "Profundidad Maxima:",
                min_value=2, max_value=20, value=6, step=1,
                key='chem_max_depth'
            )

    with col_hp3:
        if model_type == 'xgboost':
            learning_rate = st.slider(
                "Learning Rate:",
                min_value=0.01, max_value=0.3, value=0.1, step=0.01,
                key='chem_lr'
            )

    # 5. Boton de entrenamiento
    st.markdown("---")
    train_chem_btn = st.button(
        f"Entrenar Modelo para {selected_target.upper()}",
        type="primary",
        key='train_chem'
    )

    if train_chem_btn and len(chem_selected_features) > 0:
        with st.spinner(f"Entrenando modelo para {selected_target.upper()}..."):
            try:
                model, metrics, features, X_test, y_test, y_pred, model_path = train_chemical_model(
                    target=selected_target,
                    model_type=model_type,
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    learning_rate=learning_rate,
                    save_model=True,
                    feature_list=chem_selected_features
                )

                st.success(f"Modelo entrenado y guardado en: `{model_path}`")

            except Exception as e:
                st.error(f"Error durante el entrenamiento: {e}")

    elif train_chem_btn:
        st.warning("Selecciona al menos una feature para entrenar el modelo.")


def _get_target_description(target: str) -> str:
    """Retorna una descripcion breve del target quimico."""
    descriptions = {
        'target_valc': 'Carbono Final',
        'target_valmn': 'Manganeso Final',
        'target_valsi': 'Silicio Final',
        'target_valp': 'Fosforo Final',
        'target_vals': 'Azufre Final'
    }
    return descriptions.get(target, target)


# =============================================================================
# Sub-tab: Evaluacion
# =============================================================================

def _render_evaluation_section():
    """Seccion de evaluacion de modelos quimicos entrenados."""
    st.subheader("Evaluacion de Modelos Quimicos")

    # Obtener modelos entrenados
    trained_models = get_trained_chemical_models()

    if not trained_models:
        st.info("No hay modelos quimicos entrenados. Entrena un modelo en la seccion de Entrenamiento.")

        # Fallback: intentar cargar resultados pre-calculados (pkl)
        st.markdown("---")
        st.markdown("#### Resultados Pre-calculados")
        _render_precalculated_results()
        return

    # Selector de modelo
    selected_model_file = st.selectbox(
        "Selecciona un modelo entrenado:",
        options=trained_models,
        format_func=_format_model_name,
        key="chem_eval_model_select"
    )

    # Boton para evaluar
    eval_button = st.button("Evaluar Modelo", type="primary", key="chem_eval_model_btn")

    # Session state para mantener estado de evaluacion
    if 'chem_eval_results_visible' not in st.session_state:
        st.session_state['chem_eval_results_visible'] = False
        st.session_state['chem_eval_model_name'] = None

    if eval_button and selected_model_file:
        st.session_state['chem_eval_results_visible'] = True
        st.session_state['chem_eval_model_name'] = selected_model_file

    if st.session_state.get('chem_eval_model_name') != selected_model_file:
        st.session_state['chem_eval_results_visible'] = False

    if st.session_state['chem_eval_results_visible'] and selected_model_file:
        _render_model_evaluation(selected_model_file)


def _format_model_name(model_name: str) -> str:
    """Formatea el nombre del modelo para mostrar en el selector."""
    # chem_valc_xgboost_20241215_143022 -> VALC - XGBoost (2024-12-15 14:30)
    parts = model_name.split('_')
    if len(parts) >= 4:
        target = parts[1].upper()
        model_type = parts[2]
        timestamp = f"{parts[3][:4]}-{parts[3][4:6]}-{parts[3][6:8]}"
        if len(parts) >= 5:
            time_part = f"{parts[4][:2]}:{parts[4][2:4]}"
            return f"{target} - {MODEL_DISPLAY_NAMES.get(model_type, model_type)} ({timestamp} {time_part})"
        return f"{target} - {MODEL_DISPLAY_NAMES.get(model_type, model_type)} ({timestamp})"
    return model_name


def _render_model_evaluation(model_name: str):
    """Renderiza la evaluacion completa de un modelo quimico."""
    metadata = load_chemical_model_metadata(model_name)

    st.markdown("---")
    _render_model_metadata_display(metadata)
    st.markdown("---")

    with st.spinner("Cargando evaluacion..."):
        try:
            model = load_chemical_model(model_name)

            if metadata and 'features' in metadata:
                model_features = metadata['features']
                model_type = metadata.get('model_type', 'xgboost')
                target = metadata.get('target', 'valc')
            else:
                st.error("No se encontraron metadatos del modelo.")
                return

            # Cargar datos
            df = load_chemical_data()

            # Verificar features
            missing_features = [f for f in model_features if f not in df.columns]
            if missing_features:
                st.error(f"Features faltantes en el dataset: {missing_features}")
                return

            X = df[model_features].copy()
            y = df[target].copy()

            # =========================================================================
            # FIX APLICADO: REPLICACIÓN EXACTA DEL PRE-PROCESAMIENTO DEL ENTRENAMIENTO
            # Esto garantiza que X e Y sean idénticos al momento del split original.
            # (Basado en dashboard/utils/chemical_engine.py)
            # =========================================================================

            # 1. Eliminar filas donde el target es nulo (siempre obligatorio)
            mask_target = y.notnull()
            X = X[mask_target]
            y = y[mask_target]

            # 2. Capping del Target (y) para eliminar Outliers extremos (5%/95% cuantil)
            if len(y) > 100 and y.dtype in [np.float64, np.float32]:
                try:
                    # Usamos 5% y 95%
                    lower_bound = y.quantile(0.05)
                    upper_bound = y.quantile(0.95)

                    outlier_mask = (y >= lower_bound) & (y <= upper_bound)

                    # Aplicar el filtro a X y a y
                    X = X.loc[outlier_mask]
                    y = y.loc[outlier_mask]

                except Exception:
                    # Si falla el capping, se ignora y se procede.
                    pass

            # 3. Imputación de NaNs en features: reemplazar nulos por 0
            X = X.fillna(0)

            # 4. Limpieza final de valores infinitos
            X = X.replace([np.inf, -np.inf], 0)
            y = y.replace([np.inf, -np.inf], np.nan).dropna()

            # 5. Asegurar alineación final
            X = X.loc[y.index]

            if len(X) == 0:
                 st.error("El dataset de evaluación quedó vacío después de la limpieza. No se puede evaluar.")
                 return

            # =========================================================================
            # FIN DEL FIX
            # =========================================================================

            # Recrear split
            test_size = DEFAULT_HYPERPARAMS['test_size']
            random_state = DEFAULT_HYPERPARAMS['random_state']

            if metadata and 'hyperparameters' in metadata:
                test_size = metadata['hyperparameters'].get('test_size', test_size)
                random_state = metadata['hyperparameters'].get('random_state', random_state)

            # Ahora el split se aplica a un conjunto de datos idéntico al usado en el entrenamiento
            _, X_test, _, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )

            # Predecir
            y_pred = model.predict(X_test)

            # Metricas de especificacion
            spec_metrics = calculate_spec_metrics(y_test, y_pred, target)

            # Mostrar metricas
            st.markdown("##### Metricas de Evaluacion")
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                # Al recalcular aquí, el valor debe coincidir con el guardado en metadata
                metrics = calculate_metrics(y_test, y_pred)
                st.metric("RMSE", f"{metrics['RMSE']:.6f}")
            with m2:
                st.metric("R2", f"{metrics['R2']:.4f}")
            with m3:
                st.metric("MAE", f"{metrics['MAE']:.6f}")
            with m4:
                if not np.isnan(spec_metrics.get('pred_in_spec', np.nan)):
                    st.metric("% en Especificacion", f"{spec_metrics['pred_in_spec']:.2f}%")

            # Graficos
            st.markdown("##### Visualizaciones")

            col1, col2 = st.columns(2)

            with col1:
                # Importancia de variables
                importance_df = get_feature_importance(model, model_features, model_type)
                if importance_df is not None:
                    fig_imp = plot_feature_importance(
                        importance_df,
                        f"Importancia de Variables - {target.upper()}"
                    )
                    st.plotly_chart(fig_imp, width='stretch')

            with col2:
                # Prediccion vs Real
                fig_pred = plot_prediction_vs_real(
                    y_test, y_pred,
                    f"Prediccion vs Real - {target.upper()}"
                )
                st.plotly_chart(fig_pred, width='stretch')

        except Exception as e:
            st.error(f"Error durante la evaluacion: {e}")


def _render_model_metadata_display(metadata: dict):
    """Muestra los metadatos del modelo."""
    if not metadata:
        st.info("No hay metadatos disponibles.")
        return

    st.markdown("##### Informacion del Modelo")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Target:**")
        target = metadata.get('target', 'N/A')
        st.write(f"{target.upper()} - {_get_target_description(target)}")

        st.markdown("**Tipo de modelo:**")
        model_type = metadata.get('model_type', 'Desconocido')
        st.write(MODEL_DISPLAY_NAMES.get(model_type, model_type))

        st.markdown("**Fecha:**")
        st.write(metadata.get('timestamp', 'N/A'))

    with col2:
        st.markdown("**Metricas guardadas:**")
        metrics = metadata.get('metrics', {})
        if metrics:
            st.write(f"- RMSE: {metrics.get('RMSE', 'N/A'):.6f}")
            st.write(f"- R2: {metrics.get('R2', 'N/A'):.4f}")
            st.write(f"- MAE: {metrics.get('MAE', 'N/A'):.6f}")

        st.markdown("**Hiperparametros:**")
        hyperparams = metadata.get('hyperparameters', {})
        if hyperparams:
            for key, value in hyperparams.items():
                st.write(f"- {key}: {value}")

    with st.expander("Ver features utilizados"):
        features = metadata.get('features', [])
        if features:
            st.write(", ".join(features))


def _render_precalculated_results():
    """
    Renderiza resultados pre-calculados desde archivos pkl.
    Fallback cuando no hay modelos entrenados desde el dashboard.
    """
    st.info("Mostrando resultados de modelos pre-calculados (si existen)...")

    all_results = {}
    for target in CHEMICAL_TARGETS:
        result = load_single_chemical_result(target)
        if result:
            all_results[target] = result

    if not all_results:
        st.warning(
            f"No se encontraron resultados pre-calculados. "
            f"Entrena modelos desde la seccion de Entrenamiento o ejecuta los scripts de entrenamiento."
        )
        return

    # Mostrar resumen
    _render_summary_metrics(all_results)
    st.divider()

    # Selector de target
    available_targets = sorted(list(all_results.keys()))
    selected_target = st.selectbox(
        "Selecciona el Componente Quimico:",
        options=available_targets,
        key='chem_precalc_target_selector'
    )

    if selected_target:
        target_data = all_results.get(selected_target)
        if target_data:
            _render_prediction_analysis(target_data, selected_target)
            st.divider()
            _render_feature_importance_section(target_data, selected_target)


def _render_summary_metrics(results: Dict[str, Dict[str, Any]]):
    """Renderiza metricas de resumen para todos los targets quimicos."""
    st.markdown("##### Resumen de Desempeno por Componente")

    if not results:
        return

    num_results = min(len(results), 5)
    cols = st.columns(num_results)

    for idx, (target, data) in enumerate(results.items()):
        metrics = data.get('metrics', {})
        mae = metrics.get('MAE', np.nan)
        r2 = metrics.get('R2', np.nan)

        with cols[idx]:
            spec_range = CHEMICAL_SPECS.get(target, (None, None))
            spec_text = f" ({spec_range[0]}-{spec_range[1]})" if all(v is not None for v in spec_range) else ""
            st.markdown(f"**{target.upper()}**{spec_text}")
            st.metric("MAE", f"{mae:.4f}" if not np.isnan(mae) else "N/A")
            st.metric("R2", f"{r2:.4f}" if not np.isnan(r2) else "N/A")


def _render_prediction_analysis(target_data: Dict[str, Any], target_name: str):
    """Renderiza el grafico de Prediccion vs Real."""
    st.markdown(f"##### Analisis de Prediccion: {target_name.upper()}")

    y_test = target_data.get('y_test')
    y_pred = target_data.get('y_pred')
    metrics = target_data.get('metrics')

    if y_test is None or y_pred is None:
        st.warning("Datos no disponibles.")
        return

    spec_metrics = calculate_spec_metrics(y_test, y_pred, target_name)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("MAE", f"{metrics.get('MAE', 0):.4f}")
    with col2:
        st.metric("R2", f"{metrics.get('R2', 0):.4f}")
    with col3:
        if not np.isnan(spec_metrics.get('pred_in_spec', np.nan)):
            st.metric("% en Especificacion", f"{spec_metrics['pred_in_spec']:.2f}%")

    col_chart1, col_chart2 = st.columns(2)

    with col_chart1:
        fig_pred = plot_prediction_vs_real(
            y_test, y_pred,
            title=f"Prediccion vs Real - {target_name.upper()}"
        )
        st.plotly_chart(fig_pred, width='stretch')

    with col_chart2:
        try:
            fig_dist = plot_distribution(
                y_test, y_pred,
                spec_range=(spec_metrics.get('spec_min'), spec_metrics.get('spec_max')),
                title=f"Distribucion - {target_name.upper()}"
            )
            st.plotly_chart(fig_dist, width='stretch')
        except Exception:
            pass


def _render_feature_importance_section(target_data: Dict[str, Any], target_name: str):
    """Renderiza la importancia de caracteristicas."""
    st.markdown(f"##### Importancia de Variables: {target_name.upper()}")

    importance_df = target_data.get('importance_df')

    if importance_df is None or importance_df.empty:
        st.warning("Datos de importancia no disponibles.")
        return

    fig_imp = plot_feature_importance(
        importance_df,
        title=f"Top 15 Variables - {target_name.upper()}"
    )
    st.plotly_chart(fig_imp, width='stretch')