"""
Tab 2: Prediccion de Temperatura.
"""
import streamlit as st
import pandas as pd
import requests

from src.config import INPUT_FEATURES, AVAILABLE_MODELS, MODEL_DISPLAY_NAMES, BENTOML_URL
from src.scripts.data_loader import get_data_path
from src.scripts.evaluation import get_feature_importance
from src.scripts.train_temperature import train_temperature_model, XGBOOST_AVAILABLE
from components.visualizations import plot_feature_importance, plot_prediction_vs_real
from components.indicators import temperature_quality_indicator


# Mapeo inverso para UI
UI_MODEL_NAMES = {
    'Linear Regression': 'linear',
    'Random Forest Regressor': 'random_forest',
    'XGBoost Regressor': 'xgboost'
}


def render_temperature_tab(df: pd.DataFrame):
    """
    Renderiza el tab de prediccion de temperatura.

    Parameters:
    -----------
    df : DataFrame con los datos del dataset
    """
    st.header("Tarea 1: Prediccion de Temperatura Final")

    section1, section2 = st.tabs([
        "Entrenamiento y Evaluacion",
        "Simulacion de Inferencia (BentoML)"
    ])

    with section1:
        _render_training_section(df)

    with section2:
        _render_inference_section()


def _render_training_section(df: pd.DataFrame):
    """Seccion de entrenamiento y evaluacion de modelo."""
    st.subheader("Entrenamiento y Evaluacion Dinamica")

    col_config, col_results = st.columns([1, 2])

    with col_config:
        st.markdown("#### Configuracion del Modelo")

        # Seleccion de modelo (nombres amigables)
        ui_model_options = list(UI_MODEL_NAMES.keys())
        temp_model_ui = st.selectbox(
            "Modelo:",
            options=ui_model_options,
            key='temp_model_select'
        )
        model_type = UI_MODEL_NAMES[temp_model_ui]

        # Seleccion de features
        temp_available_features = [f for f in INPUT_FEATURES if f in df.columns and f != 'target_temperature']
        temp_selected_features = st.multiselect(
            "Features de Entrada:",
            options=temp_available_features,
            default=temp_available_features[:8],
            key='temp_features_select'
        )

        # Hiperparametros condicionales
        st.markdown("#### Hiperparametros")
        n_estimators = 100
        max_depth = 6
        learning_rate = 0.1

        if model_type in ['random_forest', 'xgboost']:
            n_estimators = st.slider(
                "Numero de Estimadores:",
                min_value=50, max_value=500, value=100, step=50,
                key='temp_n_estimators'
            )
            max_depth = st.slider(
                "Profundidad Maxima:",
                min_value=2, max_value=20, value=6, step=1,
                key='temp_max_depth'
            )

        if model_type == 'xgboost':
            learning_rate = st.slider(
                "Learning Rate:",
                min_value=0.01, max_value=0.3, value=0.1, step=0.01,
                key='temp_lr'
            )

        # Boton de entrenamiento
        train_temp_btn = st.button("Entrenar Modelo de Temperatura", type="primary", key='train_temp')

    with col_results:
        if train_temp_btn and len(temp_selected_features) > 0:
            with st.spinner("Entrenando modelo..."):
                try:
                    # Entrenar usando la funcion de src/
                    model, metrics, feature_names, X_test, y_test, y_pred = train_temperature_model(
                        model_type=model_type,
                        n_estimators=n_estimators,
                        max_depth=max_depth,
                        learning_rate=learning_rate
                    )

                    # Guardar en session_state
                    st.session_state.temp_model = model
                    st.session_state.temp_features = feature_names
                    st.session_state.temp_model_type = model_type

                    # Mostrar metricas
                    st.markdown("#### Metricas del Modelo")
                    m1, m2 = st.columns(2)
                    with m1:
                        st.metric("RMSE", f"{metrics['RMSE']:.2f}")
                    with m2:
                        st.metric("R2", f"{metrics['R2']:.4f}")

                    # Graficos
                    st.markdown("#### Visualizaciones")

                    # Importancia de variables
                    importance_df = get_feature_importance(model, feature_names, model_type)
                    if importance_df is not None:
                        fig_imp = plot_feature_importance(importance_df, "Importancia de Variables - Temperatura")
                        st.plotly_chart(fig_imp, use_container_width=True)

                    # Prediccion vs Real
                    fig_pred = plot_prediction_vs_real(y_test, y_pred, "Prediccion vs Real - Temperatura")
                    st.plotly_chart(fig_pred, use_container_width=True)

                except Exception as e:
                    st.error(f"Error durante el entrenamiento: {e}")

        elif train_temp_btn:
            st.warning("Selecciona al menos una feature para entrenar el modelo.")


def _render_inference_section():
    """Seccion de inferencia con BentoML."""
    st.subheader("Simulacion de Inferencia con BentoML")

    col_inputs, col_output = st.columns([1, 1])

    with col_inputs:
        st.markdown("#### Parametros de Entrada")

        # Inputs criticos para temperatura
        inp_o2_lance = st.slider(
            "Oxigeno Lance (total_o2_lance):",
            min_value=0.0, max_value=5000.0, value=1000.0, step=100.0,
            key='inf_o2_lance'
        )

        inp_gas_lance = st.slider(
            "Gas Lance (total_gas_lance):",
            min_value=0.0, max_value=5000.0, value=500.0, step=100.0,
            key='inf_gas_lance'
        )

        inp_carbon = st.slider(
            "Carbono Inyectado (total_injected_carbon):",
            min_value=0.0, max_value=2000.0, value=200.0, step=50.0,
            key='inf_carbon'
        )

        inp_mat_140107 = st.number_input(
            "Material 140107 (kg):",
            min_value=0.0, max_value=5000.0, value=0.0, step=100.0,
            key='inf_mat_140107'
        )

        inp_mat_360258 = st.number_input(
            "Material 360258 (kg):",
            min_value=0.0, max_value=5000.0, value=414.0, step=100.0,
            key='inf_mat_360258'
        )

        inp_valcr = st.slider(
            "Cromo Inicial (valcr %):",
            min_value=0.0, max_value=5.0, value=0.15, step=0.01,
            key='inf_valcr'
        )

        predict_temp_btn = st.button("Predecir Temperatura con BentoML", type="primary", key='predict_temp')

    with col_output:
        st.markdown("#### Resultado de Prediccion")

        if predict_temp_btn:
            _call_bentoml_prediction(
                inp_o2_lance, inp_gas_lance, inp_carbon,
                inp_mat_140107, inp_mat_360258, inp_valcr
            )


def _call_bentoml_prediction(inp_o2_lance, inp_gas_lance, inp_carbon,
                              inp_mat_140107, inp_mat_360258, inp_valcr):
    """Realiza la llamada a BentoML y muestra resultados."""
    try:
        df_structure = pd.read_csv(get_data_path(), nrows=1)
        feature_columns = [c for c in df_structure.columns if c not in ['heatid', 'target_temperature']]

        # Crear DataFrame de input con todos los features en 0.0
        input_df = pd.DataFrame(0.0, index=[0], columns=feature_columns)

        # Rellenar con valores del formulario
        input_df.at[0, 'total_o2_lance'] = float(inp_o2_lance)
        input_df.at[0, 'total_gas_lance'] = float(inp_gas_lance)
        input_df.at[0, 'total_injected_carbon'] = float(inp_carbon)
        input_df.at[0, 'added_mat_140107'] = float(inp_mat_140107)
        input_df.at[0, 'added_mat_360258'] = float(inp_mat_360258)
        input_df.at[0, 'valcr'] = float(inp_valcr)

        # Convertir a payload para BentoML
        payload = input_df.to_dict(orient='records')

        # Llamada a la API
        response = requests.post(
            f"{BENTOML_URL}/predict",
            json={"inputs": payload},
            headers={"content-type": "application/json"},
            timeout=10
        )

        if response.status_code == 200:
            temp_pred = response.json()

            # Mostrar temperatura predicha
            st.metric(
                label="Temperatura Estimada",
                value=f"{temp_pred:.2f} C"
            )

            # Semaforo de calidad
            status, label, description = temperature_quality_indicator(temp_pred)

            if status == "success":
                st.success(f"VERDE {label}: {description}")
            elif status == "warning":
                st.warning(f"AMARILLO {label}: {description}")
            else:
                st.error(f"ROJO {label}: {description}")

            # Rangos de referencia
            st.markdown("---")
            st.markdown("**Rangos de Temperatura:**")
            st.markdown("- VERDE **Optima:** 1580C - 1650C")
            st.markdown("- AMARILLO **Alta:** > 1650C (sobrecalentamiento)")
            st.markdown("- ROJO **Baja:** < 1580C (riesgo de reprocesamiento)")
        else:
            st.error(f"Error del servidor: {response.text}")

    except requests.exceptions.ConnectionError:
        st.error("No se pudo conectar con BentoML. Asegurate de que el servidor esta corriendo en localhost:3000")
    except Exception as e:
        st.error(f"Error: {e}")
