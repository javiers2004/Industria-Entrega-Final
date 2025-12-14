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
    AVAILABLE_MODELS, MODEL_DISPLAY_NAMES,
    BENTOML_URL, UI_MODEL_NAMES, EXCLUDE_FROM_FEATURES
)
from src.scripts.data_loader import get_data_path, get_project_root
from src.scripts.evaluation import get_feature_importance, calculate_metrics
from src.scripts.train_temperature import train_temperature_model, XGBOOST_AVAILABLE
from components.visualizations import plot_feature_importance, plot_prediction_vs_real
from components.indicators import temperature_quality_indicator


def get_available_features(df: pd.DataFrame) -> list:
    """
    Obtiene la lista de features disponibles desde los headers del dataset,
    excluyendo las columnas definidas en EXCLUDE_FROM_FEATURES.
    """
    all_columns = df.columns.tolist()
    available = [col for col in all_columns if col not in EXCLUDE_FROM_FEATURES]
    return available


def get_trained_models():
    """Escanea el directorio de modelos y devuelve una lista de modelos entrenados."""
    models_dir = get_project_root() / "models"
    models_dir.mkdir(exist_ok=True)

    # Buscar subdirectorios que contengan model.joblib
    model_dirs = [
        d for d in os.listdir(models_dir)
        if os.path.isdir(models_dir / d) and d.startswith("temp_") and (models_dir / d / "model.joblib").exists()
    ]

    # Ordenar por fecha de modificacion del directorio
    model_dirs.sort(key=lambda d: os.path.getmtime(models_dir / d), reverse=True)

    return model_dirs


def load_model_metadata(model_name: str) -> dict:
    """Carga los metadatos de un modelo si existen."""
    models_dir = get_project_root() / "models"
    metadata_path = models_dir / model_name / "metadata.json"

    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            return json.load(f)
    return None


def load_model(model_name: str):
    """Carga un modelo desde su directorio."""
    models_dir = get_project_root() / "models"
    model_path = models_dir / model_name / "model.joblib"
    return joblib.load(model_path)


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

    _render_evaluation_section(df)

    st.divider()

    _render_inference_section()


def _render_training_section(df: pd.DataFrame):
    """Seccion de entrenamiento de modelo."""
    st.subheader("Entrenamiento de Modelos")

    st.markdown("#### Entrenar un Nuevo Modelo")

    # Seleccion de modelo (nombres amigables)
    ui_model_options = list(UI_MODEL_NAMES.keys())
    temp_model_ui = st.selectbox(
        "Modelo:",
        options=ui_model_options,
        key='temp_model_select'
    )
    model_type = UI_MODEL_NAMES[temp_model_ui]

    # Cargar features disponibles desde los headers del dataset
    available_features = get_available_features(df)

    temp_selected_features = st.multiselect(
        "Features de Entrada:",
        options=available_features,
        default=available_features[:8] if len(available_features) >= 8 else available_features,
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
    train_temp_btn = st.button("Entrenar Modelo", type="primary", key='train_temp')

    if train_temp_btn and len(temp_selected_features) > 0:
        with st.spinner("Entrenando y guardando modelo..."):
            try:
                # Entrenar usando la funcion de src/ con los features seleccionados
                _, _, _, _, _, _, model_path = train_temperature_model(
                    model_type=model_type,
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    learning_rate=learning_rate,
                    save_model=True,
                    feature_list=temp_selected_features
                )
                st.success(f"Modelo entrenado y guardado en: `{model_path}`")
            except Exception as e:
                st.error(f"Error durante el entrenamiento: {e}")

    elif train_temp_btn:
        st.warning("Selecciona al menos una feature para entrenar el modelo.")


def _render_evaluation_section(df: pd.DataFrame):
    """Seccion de evaluacion de modelos entrenados."""
    st.subheader("Evaluacion de Modelos")

    trained_models = get_trained_models()

    if not trained_models:
        st.info("No hay modelos entrenados. Entrena un modelo en la seccion anterior.")
        return

    selected_model_file = st.selectbox(
        "Selecciona un modelo para evaluar:",
        options=trained_models,
        key="eval_model_select"
    )

    # Mostrar metadatos si existen
    metadata = load_model_metadata(selected_model_file)
    if metadata:
        with st.expander("Ver metadatos del modelo"):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Tipo de modelo:**")
                st.write(MODEL_DISPLAY_NAMES.get(metadata.get('model_type', ''), metadata.get('model_type', '')))
                st.markdown("**Features utilizados:**")
                st.write(", ".join(metadata.get('features', [])))
            with col2:
                st.markdown("**Hiperparametros:**")
                hyperparams = metadata.get('hyperparameters', {})
                for key, value in hyperparams.items():
                    st.write(f"- {key}: {value}")

    eval_button = st.button("Evaluar Modelo", key="eval_model_btn")

    if eval_button and selected_model_file:
        with st.spinner("Evaluando modelo..."):
            try:
                # Cargar modelo
                model = load_model(selected_model_file)

                # Cargar metadatos para obtener los features usados
                metadata = load_model_metadata(selected_model_file)

                if metadata and 'features' in metadata:
                    model_features = metadata['features']
                    model_type_from_metadata = metadata.get('model_type', 'xgboost')
                else:
                    # Fallback: intentar obtener features del modelo
                    try:
                        model_features = list(model.feature_names_in_)
                    except AttributeError:
                        st.error("No se encontraron metadatos del modelo. No se puede determinar las features usadas.")
                        return
                    # Extraer el tipo de modelo del nombre del archivo
                    model_type_from_metadata = selected_model_file.split('_')[1]

                # Cargar datos para evaluacion
                from src.scripts.data_loader import load_and_clean_data
                from sklearn.model_selection import train_test_split
                from src.config import DEFAULT_HYPERPARAMS

                df_eval = load_and_clean_data()

                # Verificar que las features existen en el dataset
                missing_features = [f for f in model_features if f not in df_eval.columns]
                if missing_features:
                    st.error(f"Las siguientes features no existen en el dataset: {missing_features}")
                    return

                X = df_eval[model_features]
                y = df_eval['target_temperature']

                # Eliminar filas con valores nulos
                mask = ~(X.isnull().any(axis=1) | y.isnull())
                X = X[mask]
                y = y[mask]

                # Recrear el mismo split de datos
                test_size = DEFAULT_HYPERPARAMS['test_size']
                random_state = DEFAULT_HYPERPARAMS['random_state']

                if metadata and 'hyperparameters' in metadata:
                    test_size = metadata['hyperparameters'].get('test_size', test_size)
                    random_state = metadata['hyperparameters'].get('random_state', random_state)

                _, X_test, _, y_test = train_test_split(
                    X, y,
                    test_size=test_size,
                    random_state=random_state
                )

                # Predecir y calcular metricas
                y_pred = model.predict(X_test)
                metrics = calculate_metrics(y_test, y_pred)

                # Mostrar metricas
                st.markdown("##### Metricas de Evaluacion")
                m1, m2, m3 = st.columns(3)
                with m1:
                    st.metric("RMSE", f"{metrics['RMSE']:.2f}")
                with m2:
                    st.metric("R2", f"{metrics['R2']:.4f}")
                with m3:
                    st.metric("MAE", f"{metrics['MAE']:.2f}")

                # Graficos
                st.markdown("##### Visualizaciones")

                # Importancia de variables
                importance_df = get_feature_importance(model, model_features, model_type_from_metadata)
                if importance_df is not None:
                    fig_imp = plot_feature_importance(importance_df, f"Importancia de Variables")
                    st.plotly_chart(fig_imp, use_container_width=True)

                # Prediccion vs Real
                fig_pred = plot_prediction_vs_real(y_test, y_pred, f"Prediccion vs Real")
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
