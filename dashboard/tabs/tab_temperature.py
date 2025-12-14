"""
Tab 2: Prediccion de Temperatura.
"""
import streamlit as st
import pandas as pd
import requests

from src.config import (
    INPUT_FEATURES, AVAILABLE_MODELS, MODEL_DISPLAY_NAMES,
    BENTOML_URL, UI_MODEL_NAMES
)
from src.scripts.data_loader import get_data_path
from src.scripts.evaluation import get_feature_importance
from src.scripts.train_temperature import train_temperature_model, XGBOOST_AVAILABLE
from components.visualizations import plot_feature_importance, plot_prediction_vs_real
from components.indicators import temperature_quality_indicator


"""
Tab 2: Prediccion de Temperatura.
"""
import streamlit as st
import pandas as pd
import requests
import os
import json
import joblib
from pathlib import Path

from src.config import (
    INPUT_FEATURES, AVAILABLE_MODELS, MODEL_DISPLAY_NAMES,
    BENTOML_URL, UI_MODEL_NAMES
)
from src.scripts.data_loader import get_data_path, get_project_root
from src.scripts.evaluation import get_feature_importance
from src.scripts.train_temperature import train_temperature_model, XGBOOST_AVAILABLE
from components.visualizations import plot_feature_importance, plot_prediction_vs_real
from components.indicators import temperature_quality_indicator


def get_trained_models():
    """Escanea el directorio de modelos y devuelve una lista de modelos entrenados."""
    models_dir = get_project_root() / "trained_models"
    models_dir.mkdir(exist_ok=True)
    
    model_files = [f for f in os.listdir(models_dir) if f.endswith(".joblib") and f.startswith("temp_")]
    
    # Ordenar por fecha de modificacion
    model_files.sort(key=lambda f: os.path.getmtime(models_dir / f), reverse=True)
    
    return model_files


def render_temperature_tab(df: pd.DataFrame):
    """
    Renderiza el tab de prediccion de temperatura.

    Parameters:
    -----------
    df : DataFrame con los datos del dataset
    """
    st.header("Tarea 1: Prediccion de Temperatura Final")

    _render_training_section(df)
    
    st.divider()
    
    _render_inference_section()


def _render_training_section(df: pd.DataFrame):
    """Seccion de entrenamiento y evaluacion de modelo."""
    st.subheader("Entrenamiento y Evaluacion de Modelos")

    col_train, col_eval = st.columns(2)

    with col_train:
        st.markdown("#### 1. Entrenar un Nuevo Modelo")

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
        st.markdown("##### Hiperparametros")
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

        if train_temp_btn and len(temp_selected_features) > 0:
            with st.spinner("Entrenando y guardando modelo..."):
                try:
                    # Entrenar usando la funcion de src/
                    _, _, _, _, _, _, model_path = train_temperature_model(
                        model_type=model_type,
                        n_estimators=n_estimators,
                        max_depth=max_depth,
                        learning_rate=learning_rate,
                        save_model=True
                    )
                    st.success(f"Modelo entrenado y guardado en: `{model_path}`")
                    # No es necesario recargar, el nuevo modelo aparecerá en la lista
                except Exception as e:
                    st.error(f"Error durante el entrenamiento: {e}")

        elif train_temp_btn:
            st.warning("Selecciona al menos una feature para entrenar el modelo.")

    with col_eval:
        st.markdown("#### 2. Evaluar Modelos Entrenados")
        
        trained_models = get_trained_models()

        if not trained_models:
            st.info("No hay modelos entrenados. Entrena un modelo a la izquierda.")
            return

        selected_model_file = st.selectbox(
            "Selecciona un modelo para evaluar:",
            options=trained_models,
            key="eval_model_select"
        )
        
        eval_button = st.button("Evaluar Modelo", key="eval_model_btn")

        if eval_button and selected_model_file:
            with st.spinner("Evaluando modelo..."):
                try:
                    models_dir = get_project_root() / "trained_models"
                    model_path = models_dir / selected_model_file
                    
                    # Cargar modelo
                    model = joblib.load(model_path)
                    
                    # Cargar datos para evaluacion
                    from src.scripts.data_loader import load_and_clean_data
                    from sklearn.model_selection import train_test_split
                    from src.config import DEFAULT_HYPERPARAMS
                    from src.scripts.evaluation import calculate_metrics

                    df_eval = load_and_clean_data()
                    
                    # Asegurar que las features del modelo esten en el df
                    # En una implementacion mas robusta, se guardarian las features con el modelo
                    try:
                        model_features = model.feature_names_in_
                    except AttributeError:
                        # Modelos como LinearRegression no tienen 'feature_names_in_' antes de fit
                        # Asumimos las features de la config. Esto es una simplificacion.
                        model_features = [f for f in INPUT_FEATURES if f in df_eval.columns]


                    X = df_eval[model_features]
                    y = df_eval['target_temperature']

                    # Recrear el mismo split de datos
                    _, X_test, _, y_test = train_test_split(
                        X, y, 
                        test_size=DEFAULT_HYPERPARAMS['test_size'], 
                        random_state=DEFAULT_HYPERPARAMS['random_state']
                    )

                    # Predecir y calcular metricas
                    y_pred = model.predict(X_test)
                    metrics = calculate_metrics(y_test, y_pred)

                    # Extraer el tipo de modelo del nombre del archivo
                    model_type_from_file = selected_model_file.split('_')[1]

                    # Mostrar metricas
                    st.markdown("##### Metricas de Evaluacion")
                    m1, m2 = st.columns(2)
                    with m1:
                        st.metric("RMSE", f"{metrics['RMSE']:.2f}")
                    with m2:
                        st.metric("R2", f"{metrics['R2']:.4f}")

                    # Graficos
                    st.markdown("##### Visualizaciones")
                    
                    # Importancia de variables
                    importance_df = get_feature_importance(model, model_features, model_type_from_file)
                    if importance_df is not None:
                        fig_imp = plot_feature_importance(importance_df, f"Importancia de Variables ({selected_model_file})")
                        st.plotly_chart(fig_imp, use_container_width=True)

                    # Prediccion vs Real
                    fig_pred = plot_prediction_vs_real(y_test, y_pred, f"Prediccion vs Real ({selected_model_file})")
                    st.plotly_chart(fig_pred, use_container_width=True)

                except Exception as e:
                    st.error(f"Error durante la evaluacion: {e}")



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
            # Validar y parsear respuesta JSON
            try:
                result = response.json()
                # Manejar diferentes formatos de respuesta
                if isinstance(result, (int, float)):
                    temp_pred = float(result)
                elif isinstance(result, dict) and 'prediction' in result:
                    temp_pred = float(result['prediction'])
                elif isinstance(result, list) and len(result) > 0:
                    temp_pred = float(result[0])
                else:
                    st.error(f"Formato de respuesta inesperado: {type(result)}")
                    return
            except (ValueError, TypeError, KeyError) as e:
                st.error(f"Error procesando respuesta del servidor: {e}")
                return

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
            st.error(f"Error del servidor (HTTP {response.status_code}): {response.text}")

    except requests.exceptions.Timeout:
        st.error("Timeout: El servidor BentoML tardo demasiado en responder. Intenta de nuevo.")
    except requests.exceptions.ConnectionError:
        st.error("No se pudo conectar con BentoML. Asegurate de que el servidor esta corriendo en localhost:3000")
    except requests.exceptions.RequestException as e:
        st.error(f"Error de red: {e}")
    except Exception as e:
        st.error(f"Error inesperado: {e}")
