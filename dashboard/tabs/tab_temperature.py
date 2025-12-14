"""
Tab 2: Prediccion de Temperatura.
"""
import streamlit as st
import pandas as pd
import requests
import os
import json
import joblib
import subprocess
import socket
import signal
import time
import platform
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


# =============================================================================
# Funciones Helper para Control del Servicio BentoML
# =============================================================================

def is_port_in_use(port: int = 3000) -> bool:
    """
    Verifica si un puerto esta en uso (servicio BentoML corriendo).

    Parameters:
    -----------
    port : int - Numero de puerto a verificar (default 3000)

    Returns:
    --------
    bool - True si el puerto esta en uso, False en caso contrario
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1)
        result = s.connect_ex(('localhost', port))
        return result == 0


def _get_pid_on_port(port: int) -> list:
    """
    Obtiene los PIDs de procesos escuchando en un puerto (multiplataforma).

    Parameters:
    -----------
    port : int - Numero de puerto

    Returns:
    --------
    list - Lista de PIDs encontrados
    """
    pids = []
    try:
        if platform.system() == "Windows":
            result = subprocess.run(
                ["netstat", "-ano"],
                capture_output=True, text=True
            )
            for line in result.stdout.split('\n'):
                if f":{port}" in line and "LISTENING" in line:
                    parts = line.split()
                    if parts:
                        try:
                            pids.append(int(parts[-1]))
                        except ValueError:
                            pass
        else:  # macOS/Linux
            result = subprocess.run(
                ["lsof", "-ti", f":{port}"],
                capture_output=True, text=True
            )
            if result.stdout.strip():
                pids = [int(p) for p in result.stdout.strip().split('\n') if p.strip()]
    except Exception:
        pass
    return list(set(pids))


def _render_service_status() -> bool:
    """
    Muestra el indicador de estado del servicio BentoML.

    Returns:
    --------
    bool - True si el servicio esta online, False en caso contrario
    """
    is_online = is_port_in_use(3000)

    if is_online:
        st.success("🟢 ESTADO: **ONLINE** - Servicio BentoML activo en puerto 3000")
    else:
        st.error("🔴 ESTADO: **OFFLINE** - Servicio BentoML no disponible")

    return is_online


def _start_bentoml_service() -> bool:
    """
    Inicia el servicio BentoML en segundo plano.

    Returns:
    --------
    bool - True si se inicio correctamente, False en caso contrario
    """
    if is_port_in_use(3000):
        st.warning("El servicio ya esta corriendo en el puerto 3000")
        return False

    try:
        # Configurar argumentos segun el sistema operativo
        popen_kwargs = {
            "stdout": subprocess.DEVNULL,
            "stderr": subprocess.DEVNULL,
            "cwd": str(get_project_root()),
        }

        if platform.system() == "Windows":
            popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            popen_kwargs["start_new_session"] = True

        # Iniciar servicio BentoML
        process = subprocess.Popen(
            ["bentoml", "serve", "deploy.service:EAFModel", "--port", "3000"],
            **popen_kwargs
        )

        # Guardar PID en session state
        st.session_state['bentoml_pid'] = process.pid

        # Esperar brevemente para que inicie
        time.sleep(2)

        # Verificar que inicio
        if is_port_in_use(3000):
            st.success("Servicio BentoML iniciado correctamente")
            return True
        else:
            st.info("El servicio esta iniciando... Espera unos segundos y refresca la pagina.")
            return True

    except FileNotFoundError:
        st.error("BentoML no esta instalado o no se encuentra en el PATH del sistema")
        return False
    except Exception as e:
        st.error(f"Error al iniciar el servicio: {e}")
        return False


def _stop_bentoml_service() -> bool:
    """
    Detiene el servicio BentoML.

    Returns:
    --------
    bool - True si se detuvo correctamente, False en caso contrario
    """
    # Intentar usando el PID guardado primero
    if 'bentoml_pid' in st.session_state and st.session_state['bentoml_pid']:
        try:
            pid = st.session_state['bentoml_pid']
            if platform.system() == "Windows":
                subprocess.run(["taskkill", "/F", "/PID", str(pid)], capture_output=True)
            else:
                os.kill(pid, signal.SIGTERM)
            st.session_state['bentoml_pid'] = None
            time.sleep(1)
            if not is_port_in_use(3000):
                return True
        except ProcessLookupError:
            st.session_state['bentoml_pid'] = None
        except Exception:
            pass

    # Fallback: matar procesos en el puerto 3000
    pids = _get_pid_on_port(3000)
    for pid in pids:
        try:
            if platform.system() == "Windows":
                subprocess.run(["taskkill", "/F", "/PID", str(pid)], capture_output=True)
            else:
                os.kill(pid, signal.SIGTERM)
        except Exception:
            pass

    time.sleep(1)
    st.session_state['bentoml_pid'] = None
    return not is_port_in_use(3000)


def _register_model_to_bentoml(model_name: str) -> bool:
    """
    Registra un modelo local en BentoML para usarlo en inferencia.

    Parameters:
    -----------
    model_name : str - Nombre del directorio del modelo en /models/

    Returns:
    --------
    bool - True si se registro correctamente, False en caso contrario
    """
    try:
        import bentoml

        # Cargar modelo y metadatos
        model = load_model(model_name)
        metadata = load_model_metadata(model_name)
        model_type = metadata.get('model_type', 'xgboost') if metadata else 'xgboost'

        # Registrar en BentoML segun el tipo de modelo
        if model_type == 'xgboost':
            bentoml.xgboost.save_model(
                "eaf_temperature_model",
                model,
                signatures={"predict": {"batchable": True}},
                metadata={"source_model": model_name, "model_type": model_type}
            )
        else:  # sklearn models (linear, random_forest)
            bentoml.sklearn.save_model(
                "eaf_temperature_model",
                model,
                signatures={"predict": {"batchable": True}},
                metadata={"source_model": model_name, "model_type": model_type}
            )

        st.success(f"Modelo '{model_name}' registrado en BentoML correctamente")
        return True

    except ImportError:
        st.error("BentoML no esta instalado. Ejecuta: pip install bentoml")
        return False
    except Exception as e:
        st.error(f"Error al registrar modelo en BentoML: {e}")
        return False


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


def _init_temperature_session_state():
    """Inicializa el session state para el tab de temperatura."""
    if 'bentoml_pid' not in st.session_state:
        st.session_state['bentoml_pid'] = None
    if 'current_bentoml_model' not in st.session_state:
        st.session_state['current_bentoml_model'] = None


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
    # Inicializar session state
    _init_temperature_session_state()

    st.header("Tarea 1: Prediccion de Temperatura Final")

    # Crear sub-tabs para organizar las secciones
    tab_train, tab_eval, tab_infer = st.tabs([
        "Entrenamiento",
        "Evaluacion",
        "Inferencia"
    ])

    with tab_train:
        _render_training_section(df)

    with tab_eval:
        _render_evaluation_section(df)

    with tab_infer:
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
        st.info("No hay modelos entrenados. Entrena un modelo en la seccion de Entrenamiento.")
        return

    selected_model_file = st.selectbox(
        "Selecciona un modelo:",
        options=trained_models,
        key="eval_model_select"
    )

    # Mostrar metadatos automaticamente al seleccionar modelo
    if selected_model_file:
        metadata = load_model_metadata(selected_model_file)

        st.markdown("---")
        _render_model_metadata_display(metadata)
        st.markdown("---")

        # Boton para evaluacion detallada (graficos)
        st.markdown("##### Evaluacion Detallada")
        st.markdown("Ejecuta la evaluacion para ver graficos de importancia de variables y prediccion vs real.")

        eval_button = st.button("Ejecutar Evaluacion Detallada", key="eval_model_btn")

        if eval_button:
            with st.spinner("Evaluando modelo..."):
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


def _render_inference_section():
    """Seccion de inferencia con BentoML y control del servicio."""
    st.subheader("Inferencia con BentoML")

    # =========================================================================
    # Seleccion de Modelo
    # =========================================================================
    st.markdown("#### Seleccion de Modelo")

    trained_models = get_trained_models()

    if not trained_models:
        st.warning("No hay modelos entrenados. Entrena un modelo en la seccion de Entrenamiento primero.")
        return

    selected_model = st.selectbox(
        "Modelo para inferencia:",
        options=trained_models,
        key="inference_model_select"
    )

    # Mostrar metadatos del modelo seleccionado
    if selected_model:
        metadata = load_model_metadata(selected_model)
        if metadata:
            with st.expander("Ver detalles del modelo", expanded=False):
                _render_model_metadata_display(metadata)

    st.markdown("---")

    # =========================================================================
    # Control del Servicio BentoML
    # =========================================================================
    st.markdown("#### Control del Servicio BentoML")

    # Mostrar estado actual del servicio
    is_online = _render_service_status()

    # Mostrar que modelo esta activo en BentoML
    if st.session_state.get('current_bentoml_model'):
        st.info(f"Modelo activo en BentoML: `{st.session_state['current_bentoml_model']}`")

    # Botones de control
    col_start, col_stop, col_refresh = st.columns(3)

    with col_start:
        start_btn = st.button(
            "INICIAR Servicio",
            type="primary",
            key="btn_start_bentoml",
            disabled=is_online
        )
        if start_btn and selected_model:
            with st.spinner(f"Registrando modelo '{selected_model}' en BentoML..."):
                if _register_model_to_bentoml(selected_model):
                    st.session_state['current_bentoml_model'] = selected_model
                    with st.spinner("Iniciando servicio BentoML..."):
                        _start_bentoml_service()
            st.rerun()

    with col_stop:
        stop_btn = st.button(
            "DETENER Servicio",
            type="secondary",
            key="btn_stop_bentoml",
            disabled=not is_online
        )
        if stop_btn:
            with st.spinner("Deteniendo servicio..."):
                if _stop_bentoml_service():
                    st.session_state['current_bentoml_model'] = None
                    st.success("Servicio detenido correctamente")
            st.rerun()

    with col_refresh:
        if st.button("Refrescar Estado", key="btn_refresh_status"):
            st.rerun()

    st.markdown("---")

    # =========================================================================
    # Parametros de Entrada para Prediccion
    # =========================================================================
    st.markdown("#### Parametros de Entrada")

    if not is_online:
        st.warning("Inicia el servicio BentoML antes de realizar predicciones.")

    col_inputs, col_output = st.columns([1, 1])

    with col_inputs:
        # Inputs criticos para temperatura (deshabilitados si offline)
        inp_o2_lance = st.slider(
            "Oxigeno Lance (total_o2_lance):",
            min_value=0.0, max_value=5000.0, value=1000.0, step=100.0,
            key='inf_o2_lance',
            disabled=not is_online
        )

        inp_gas_lance = st.slider(
            "Gas Lance (total_gas_lance):",
            min_value=0.0, max_value=5000.0, value=500.0, step=100.0,
            key='inf_gas_lance',
            disabled=not is_online
        )

        inp_carbon = st.slider(
            "Carbono Inyectado (total_injected_carbon):",
            min_value=0.0, max_value=2000.0, value=200.0, step=50.0,
            key='inf_carbon',
            disabled=not is_online
        )

        inp_mat_140107 = st.number_input(
            "Material 140107 (kg):",
            min_value=0.0, max_value=5000.0, value=0.0, step=100.0,
            key='inf_mat_140107',
            disabled=not is_online
        )

        inp_mat_360258 = st.number_input(
            "Material 360258 (kg):",
            min_value=0.0, max_value=5000.0, value=414.0, step=100.0,
            key='inf_mat_360258',
            disabled=not is_online
        )

        inp_valcr = st.slider(
            "Cromo Inicial (valcr %):",
            min_value=0.0, max_value=5.0, value=0.15, step=0.01,
            key='inf_valcr',
            disabled=not is_online
        )

        predict_temp_btn = st.button(
            "Predecir Temperatura",
            type="primary",
            key='predict_temp',
            disabled=not is_online
        )

    with col_output:
        st.markdown("#### Resultado de Prediccion")

        if predict_temp_btn and is_online:
            _call_bentoml_prediction(
                inp_o2_lance, inp_gas_lance, inp_carbon,
                inp_mat_140107, inp_mat_360258, inp_valcr
            )
        elif predict_temp_btn and not is_online:
            st.error("El servicio BentoML no esta activo. Inicialo primero.")


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
