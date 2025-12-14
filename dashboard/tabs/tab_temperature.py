"""
Tab 2: Prediccion de Temperatura.
"""
import streamlit as st
import pandas as pd
import os
import json
import joblib

from src.config import (
    MODEL_DISPLAY_NAMES, UI_MODEL_NAMES, EXCLUDE_FROM_FEATURES
)
from src.scripts.data_loader import get_project_root
from src.scripts.evaluation import get_feature_importance
from src.scripts.train_temperature import train_temperature_model
from components.visualizations import plot_feature_importance, plot_prediction_vs_real


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


def _render_model_metadata_display(metadata: dict):
    """
    Muestra los metadatos completos de un modelo incluyendo metricas guardadas.

    Parameters:
    -----------
    metadata : dict - Metadatos del modelo desde metadata.json
    """
    if not metadata:
        st.info("No hay metadatos disponibles para este modelo.")
        return

    # Mostrar metricas prominentemente
    st.markdown("##### Metricas Guardadas del Entrenamiento")
    metrics = metadata.get('metrics', {})
    if metrics:
        m1, m2, m3 = st.columns(3)
        with m1:
            rmse_val = metrics.get('RMSE')
            st.metric("RMSE", f"{rmse_val:.2f}" if isinstance(rmse_val, (int, float)) else "N/A")
        with m2:
            r2_val = metrics.get('R2')
            st.metric("R2", f"{r2_val:.4f}" if isinstance(r2_val, (int, float)) else "N/A")
        with m3:
            mae_val = metrics.get('MAE')
            st.metric("MAE", f"{mae_val:.2f}" if isinstance(mae_val, (int, float)) else "N/A")
    else:
        st.warning("Este modelo no tiene metricas guardadas.")

    # Mostrar otros metadatos
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Tipo de modelo:**")
        model_type = metadata.get('model_type', 'Desconocido')
        st.write(MODEL_DISPLAY_NAMES.get(model_type, model_type))

        st.markdown("**Fecha de entrenamiento:**")
        timestamp = metadata.get('timestamp', 'N/A')
        st.write(timestamp)

    with col2:
        st.markdown("**Hiperparametros:**")
        hyperparams = metadata.get('hyperparameters', {})
        if hyperparams:
            for key, value in hyperparams.items():
                st.write(f"- {key}: {value}")
        else:
            st.write("No disponibles")

    # Features en seccion expandible
    with st.expander("Ver features utilizados"):
        features = metadata.get('features', [])
        if features:
            st.write(", ".join(features))
        else:
            st.write("No disponibles")


# =============================================================================
# Funcion Principal del Tab
# =============================================================================

def render_temperature_tab(df: pd.DataFrame):
    """
    Renderiza el tab de prediccion de temperatura con sub-tabs.

    Parameters:
    -----------
    df : DataFrame con los datos del dataset
    """
    st.header("Tarea 1: Prediccion de Temperatura Final")

    # Crear sub-tabs para organizar las secciones (sin Inferencia)
    tab_train, tab_eval = st.tabs([
        "Entrenamiento",
        "Evaluacion"
    ])

    with tab_train:
        _render_training_section(df)

    with tab_eval:
        _render_evaluation_section(df)


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
        st.info("No hay modelos entrenados. Entrena un modelo en la seccion de Entrenamiento.")
        return

    selected_model_file = st.selectbox(
        "Selecciona un modelo:",
        options=trained_models,
        key="eval_model_select"
    )

    # Boton para evaluar - nada se muestra hasta que se pulse
    eval_button = st.button("Evaluar Modelo", type="primary", key="eval_model_btn")

    # Usar session_state para mantener el estado de evaluacion
    if 'eval_results_visible' not in st.session_state:
        st.session_state['eval_results_visible'] = False
        st.session_state['eval_model_name'] = None

    # Si se pulsa el boton, activar la visualizacion
    if eval_button and selected_model_file:
        st.session_state['eval_results_visible'] = True
        st.session_state['eval_model_name'] = selected_model_file

    # Si el modelo seleccionado cambia, resetear la visualizacion
    if st.session_state.get('eval_model_name') != selected_model_file:
        st.session_state['eval_results_visible'] = False

    # Solo mostrar resultados si se ha pulsado el boton
    if st.session_state['eval_results_visible'] and selected_model_file:
        metadata = load_model_metadata(selected_model_file)

        st.markdown("---")
        _render_model_metadata_display(metadata)
        st.markdown("---")

        # Evaluacion detallada (graficos)
        with st.spinner("Cargando graficos de evaluacion..."):
            try:
                # Cargar modelo
                model = load_model(selected_model_file)

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

                # Predecir
                y_pred = model.predict(X_test)

                # Graficos
                st.markdown("##### Visualizaciones")

                # Importancia de variables
                importance_df = get_feature_importance(model, model_features, model_type_from_metadata)
                if importance_df is not None:
                    fig_imp = plot_feature_importance(importance_df, "Importancia de Variables")
                    st.plotly_chart(fig_imp, use_container_width=True)

                # Prediccion vs Real
                fig_pred = plot_prediction_vs_real(y_test, y_pred, "Prediccion vs Real")
                st.plotly_chart(fig_pred, use_container_width=True)

            except Exception as e:
                st.error(f"Error durante la evaluacion: {e}")
