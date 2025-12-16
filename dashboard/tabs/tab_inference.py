"""
Tab de Inferencia en Tiempo Real.
Pestaña centralizada para realizar inferencias con BentoML.
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

from dashboard.config import BENTOML_URL, MODEL_DISPLAY_NAMES
from dashboard.utils.data_engine import get_project_root
from dashboard.components.indicators import temperature_quality_indicator


# =============================================================================
# Funciones de Escaneo de Modelos
# =============================================================================

def get_all_trained_models():
    """
    Escanea el directorio de modelos y devuelve una lista de todos los modelos
    entrenados (tanto de temperatura como quimicos).

    Returns:
    --------
    list - Lista de nombres de directorios de modelos
    """
    models_dir = get_project_root() / "models"
    models_dir.mkdir(exist_ok=True)

    # Buscar subdirectorios que contengan model.joblib
    model_dirs = [
        d for d in os.listdir(models_dir)
        if os.path.isdir(models_dir / d) and (models_dir / d / "model.joblib").exists()
    ]

    # Ordenar por fecha de modificacion del directorio (mas recientes primero)
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
# Funciones de Control del Servicio BentoML
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
            st.info("El servicio esta iniciando... Espera unos segundos y refresca.")
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
    import psutil  # Necesario para matar el árbol de procesos (padre + hijos)

    # Función interna para matar el proceso y toda su descendencia (workers)
    def kill_tree_recursively(pid):
        try:
            parent = psutil.Process(pid)
            children = parent.children(recursive=True)  # Obtener todos los hijos
            
            # 1. Matar a los hijos primero
            for child in children:
                try:
                    child.kill()
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            psutil.wait_procs(children, timeout=3) # Esperar a que mueran
            
            # 2. Matar al padre
            parent.kill()
            parent.wait(3)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            # Fallback: Si psutil falla, usar el método nativo del sistema
            try:
                if platform.system() == "Windows":
                    # El flag /T mata también a los procesos hijos (Tree)
                    subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid)], capture_output=True)
                else:
                    os.kill(pid, signal.SIGTERM)
            except Exception:
                pass

    # --- LÓGICA ORIGINAL MEJORADA ---

    # 1. Intentar usando el PID guardado primero
    if 'bentoml_pid' in st.session_state and st.session_state['bentoml_pid']:
        try:
            pid = st.session_state['bentoml_pid']
            st.write(f"🛑 Deteniendo PID guardado: {pid} (y subprocesos)...")
            
            kill_tree_recursively(pid)  # <--- Usamos la limpieza profunda aquí
            
            st.session_state['bentoml_pid'] = None
            time.sleep(1)
            if not is_port_in_use(3000):
                return True
        except Exception:
            st.session_state['bentoml_pid'] = None

    # 2. Fallback: matar procesos en el puerto 3000
    # Primero intentamos barrer con psutil (más preciso)
    try:
        for proc in psutil.process_iter(['pid']):
            try:
                for conn in proc.connections(kind='inet'):
                    if conn.laddr.port == 3000:
                        kill_tree_recursively(proc.info['pid'])
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                pass
    except Exception:
        pass

    # Segundo barrido con tu función original _get_pid_on_port por seguridad
    pids = _get_pid_on_port(3000)
    for pid in pids:
        try:
            kill_tree_recursively(pid)
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


def cleanup_bentoml_service():
    """
    Limpia el servicio BentoML al cerrar la aplicacion.
    Llamada desde app.py mediante atexit.
    """
    try:
        pids = _get_pid_on_port(3000)
        for pid in pids:
            try:
                if platform.system() == "Windows":
                    subprocess.run(["taskkill", "/F", "/PID", str(pid)],
                                   capture_output=True, timeout=5)
                else:
                    os.kill(pid, signal.SIGTERM)
            except Exception:
                pass
    except Exception:
        pass


# =============================================================================
# Funciones de Inicializacion de Session State
# =============================================================================

def _init_inference_session_state():
    """Inicializa el session state para la inferencia."""
    if 'bentoml_pid' not in st.session_state:
        st.session_state['bentoml_pid'] = None
    if 'current_bentoml_model' not in st.session_state:
        st.session_state['current_bentoml_model'] = None
    if 'inference_input_values' not in st.session_state:
        st.session_state['inference_input_values'] = {}


# =============================================================================
# Funcion Principal del Tab
# =============================================================================

def render_inference_tab():
    """
    Renderiza el tab de inferencia en tiempo real con BentoML.
    """
    # Inicializar session state
    _init_inference_session_state()

    st.header("Inferencia en Tiempo Real")
    st.markdown("Realiza predicciones utilizando modelos entrenados via BentoML.")

    # =========================================================================
    # Seccion 1: Seleccion de Modelo
    # =========================================================================
    st.subheader("Seleccion de Modelo")

    trained_models = get_all_trained_models()

    if not trained_models:
        st.warning("No hay modelos entrenados. Entrena un modelo en la seccion correspondiente primero.")
        return

    selected_model = st.selectbox(
        "Selecciona un modelo para inferencia:",
        options=trained_models,
        key="global_inference_model_select",
        help="Lista de todos los modelos disponibles (temperatura y quimicos)"
    )

    # Mostrar informacion del modelo seleccionado
    if selected_model:
        metadata = load_model_metadata(selected_model)
        if metadata:
            col_info1, col_info2, col_info3 = st.columns(3)
            with col_info1:
                model_type = metadata.get('model_type', 'Desconocido')
                st.info(f"**Tipo:** {MODEL_DISPLAY_NAMES.get(model_type, model_type)}")
            with col_info2:
                target = metadata.get('target', 'N/A')
                st.info(f"**Target:** {target}")
            with col_info3:
                metrics = metadata.get('metrics', {})
                r2 = metrics.get('R2', 'N/A')
                r2_str = f"{r2:.4f}" if isinstance(r2, (int, float)) else r2
                st.info(f"**R2:** {r2_str}")

    st.markdown("---")

    # =========================================================================
    # Seccion 2: Estado y Control del Servicio
    # =========================================================================
    st.subheader("Estado del Servicio BentoML")

    is_online = is_port_in_use(3000)
    current_model = st.session_state.get('current_bentoml_model')

    # Determinar el estado y el texto del boton
    if not is_online:
        # No hay servicio corriendo
        st.error("OFFLINE - Servicio BentoML no disponible")
        button_text = "Lanzar Inferencia (Iniciar BentoML)"
        button_type = "primary"
        show_button = True
    elif is_online and current_model != selected_model:
        # Servicio corriendo pero con modelo diferente
        st.warning(f"ONLINE - Modelo activo: `{current_model}` (diferente al seleccionado)")
        button_text = "Recargar Instancia con Nuevo Modelo"
        button_type = "primary"
        show_button = True
    else:
        # Servicio corriendo con el modelo correcto
        st.success(f"ONLINE - Modelo activo: `{current_model}`")
        button_text = None
        button_type = None
        show_button = False

    col_action, col_stop = st.columns([2, 1])

    with col_action:
        if show_button:
            action_btn = st.button(button_text, type=button_type, key="btn_action_bentoml")
            if action_btn and selected_model:
                # Si ya hay servicio, detenerlo primero
                if is_online:
                    with st.spinner("Deteniendo servicio anterior..."):
                        _stop_bentoml_service()
                        time.sleep(1)

                # Registrar y arrancar
                with st.spinner(f"Registrando modelo '{selected_model}' en BentoML..."):
                    if _register_model_to_bentoml(selected_model):
                        st.session_state['current_bentoml_model'] = selected_model
                        with st.spinner("Iniciando servicio BentoML..."):
                            _start_bentoml_service()
                st.rerun()

    with col_stop:
        if is_online:
            stop_btn = st.button("Detener Servicio", type="secondary", key="btn_stop_bentoml")
            if stop_btn:
                with st.spinner("Deteniendo servicio..."):
                    if _stop_bentoml_service():
                        st.session_state['current_bentoml_model'] = None
                        st.success("Servicio detenido correctamente")
                st.rerun()

    st.markdown("---")

    # =========================================================================
    # Seccion 3: Inputs Dinamicos y Prediccion
    # =========================================================================
    if not is_online:
        st.info("Inicia el servicio BentoML para realizar predicciones.")
        return

    st.subheader("Parametros de Entrada")

    # Obtener features del modelo activo (usar el modelo actual o el seleccionado como fallback)
    active_model_name = st.session_state.get('current_bentoml_model') or selected_model
    if not active_model_name:
        st.error("No hay modelo seleccionado.")
        return

    active_metadata = load_model_metadata(active_model_name)

    if not active_metadata or 'features' not in active_metadata:
        st.error("No se encontraron metadatos del modelo activo. No se pueden generar inputs.")
        return

    model_features = active_metadata['features']
    target = active_metadata.get('target', 'target_temperature')

    st.markdown(f"**Features requeridos:** {len(model_features)}")

    # Generar inputs dinamicamente basados en los features del modelo
    col_inputs, col_result = st.columns([1, 1])

    with col_inputs:
        st.markdown("##### Valores de Entrada")

        input_values = {}

        # Crear inputs en columnas para mejor organizacion
        num_features = len(model_features)
        features_per_col = (num_features + 1) // 2

        col_a, col_b = st.columns(2)

        for idx, feature in enumerate(model_features):
            # Alternar entre columnas
            target_col = col_a if idx < features_per_col else col_b

            with target_col:
                # Usar number_input para cada feature
                default_value = st.session_state.get(f'inf_input_{feature}', 0.0)
                input_values[feature] = st.number_input(
                    f"{feature}:",
                    value=default_value,
                    format="%.4f",
                    key=f'inf_input_{feature}'
                )

        # Boton de prediccion
        st.markdown("---")
        predict_btn = st.button(
            "Predecir",
            type="primary",
            key='btn_predict_inference'
        )

    with col_result:
        st.markdown("##### Resultado de Prediccion")

        if predict_btn:
            _call_bentoml_prediction(input_values, model_features, target)


def _call_bentoml_prediction(input_values: dict, features: list, target: str):
    """
    Realiza la llamada a BentoML con los valores de input dinamicos.

    Parameters:
    -----------
    input_values : dict - Diccionario con los valores de entrada {feature: valor}
    features : list - Lista de features esperados por el modelo
    target : str - Nombre del target para mostrar contexto
    """
    try:
        # Crear DataFrame con los valores de input
        input_df = pd.DataFrame([input_values])

        # Asegurar que las columnas estan en el orden correcto
        input_df = input_df[features]

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
                    prediction = float(result)
                elif isinstance(result, dict) and 'prediction' in result:
                    prediction = float(result['prediction'])
                elif isinstance(result, list) and len(result) > 0:
                    prediction = float(result[0])
                else:
                    st.error(f"Formato de respuesta inesperado: {type(result)}")
                    return
            except (ValueError, TypeError, KeyError) as e:
                st.error(f"Error procesando respuesta del servidor: {e}")
                return

            # Mostrar prediccion
            st.metric(
                label=f"Prediccion ({target})",
                value=f"{prediction:.4f}"
            )

            # Si es temperatura, mostrar indicador de calidad
            if 'temperature' in target.lower():
                st.markdown("---")
                status, label, description = temperature_quality_indicator(prediction)

                if status == "success":
                    st.success(f"VERDE {label}: {description}")
                elif status == "warning":
                    st.warning(f"AMARILLO {label}: {description}")
                else:
                    st.error(f"ROJO {label}: {description}")

                # Rangos de referencia
                st.markdown("**Rangos de Temperatura:**")
                st.markdown("- VERDE **Optima:** 1580C - 1650C")
                st.markdown("- AMARILLO **Alta:** > 1650C")
                st.markdown("- ROJO **Baja:** < 1580C")

        else:
            st.error(f"Error del servidor (HTTP {response.status_code}): {response.text}")

    except requests.exceptions.Timeout:
        st.error("Timeout: El servidor BentoML tardo demasiado en responder.")
    except requests.exceptions.ConnectionError:
        st.error("No se pudo conectar con BentoML. Verifica que el servicio este activo.")
    except requests.exceptions.RequestException as e:
        st.error(f"Error de red: {e}")
    except Exception as e:
        st.error(f"Error inesperado: {e}")
