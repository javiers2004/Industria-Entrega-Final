"""
Tab 3: Prediccion de Composicion Quimica.
"""
import streamlit as st
import pandas as pd

from src.config import (
    CHEMICAL_TARGETS, CHEMICAL_SPECS, INPUT_FEATURES, UI_MODEL_NAMES
)
from src.scripts.evaluation import get_feature_importance
from src.scripts.train_chemical import train_chemical_model
from components.visualizations import plot_feature_importance, plot_prediction_vs_real
from components.indicators import chemical_spec_indicator


def render_chemical_tab(df: pd.DataFrame):
    """
    Renderiza el tab de prediccion de composicion quimica.

    Parameters:
    -----------
    df : DataFrame con los datos del dataset
    """
    st.header("Tarea 2: Prediccion de Composicion Quimica")

    chem_section1, chem_section2 = st.tabs([
        "Entrenamiento y Evaluacion",
        "Simulacion de Inferencia Quimica"
    ])

    with chem_section1:
        _render_training_section(df)

    with chem_section2:
        _render_inference_section()


def _render_training_section(df: pd.DataFrame):
    """Seccion de entrenamiento y evaluacion de modelo quimico."""
    st.subheader("Entrenamiento y Evaluacion Dinamica")

    col_config, col_results = st.columns([1, 2])

    with col_config:
        st.markdown("#### Configuracion del Modelo")

        # Selector de target quimico
        chem_target = st.radio(
            "Elemento Quimico a Predecir:",
            options=CHEMICAL_TARGETS,
            format_func=lambda x: x.upper(),
            key='chem_target_select'
        )

        # Seleccion de modelo
        ui_model_options = list(UI_MODEL_NAMES.keys())
        chem_model_ui = st.selectbox(
            "Modelo:",
            options=ui_model_options,
            key='chem_model_select'
        )
        model_type = UI_MODEL_NAMES[chem_model_ui]

        # Seleccion de features (excluyendo el target)
        chem_available_features = [f for f in INPUT_FEATURES if f in df.columns and f != chem_target]
        chem_selected_features = st.multiselect(
            "Features de Entrada:",
            options=chem_available_features,
            default=[f for f in chem_available_features if f not in CHEMICAL_TARGETS][:8],
            key='chem_features_select'
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
                key='chem_n_estimators'
            )
            max_depth = st.slider(
                "Profundidad Maxima:",
                min_value=2, max_value=20, value=6, step=1,
                key='chem_max_depth'
            )

        if model_type == 'xgboost':
            learning_rate = st.slider(
                "Learning Rate:",
                min_value=0.01, max_value=0.3, value=0.1, step=0.01,
                key='chem_lr'
            )

        # Boton de entrenamiento
        train_chem_btn = st.button("Entrenar Modelo Quimico", type="primary", key='train_chem')

    with col_results:
        if train_chem_btn and len(chem_selected_features) > 0:
            with st.spinner("Entrenando modelo..."):
                try:
                    # Entrenar usando la funcion de src/
                    model, metrics, feature_names, X_test, y_test, y_pred = train_chemical_model(
                        target=chem_target,
                        model_type=model_type,
                        n_estimators=n_estimators,
                        max_depth=max_depth,
                        learning_rate=learning_rate
                    )

                    # Guardar en session_state
                    st.session_state.chem_model = model
                    st.session_state.chem_features = feature_names
                    st.session_state.chem_target = chem_target
                    st.session_state.chem_model_type = model_type

                    # Mostrar metricas
                    st.markdown(f"#### Metricas del Modelo - {chem_target.upper()}")
                    m1, m2 = st.columns(2)
                    with m1:
                        st.metric("RMSE", f"{metrics['RMSE']:.6f}")
                    with m2:
                        st.metric("R2", f"{metrics['R2']:.4f}")

                    # Graficos
                    st.markdown("#### Visualizaciones")

                    # Importancia de variables
                    importance_df = get_feature_importance(model, feature_names, model_type)
                    if importance_df is not None:
                        fig_imp = plot_feature_importance(importance_df, f"Importancia de Variables - {chem_target.upper()}")
                        st.plotly_chart(fig_imp, use_container_width=True)

                    # Prediccion vs Real
                    fig_pred = plot_prediction_vs_real(y_test, y_pred, f"Prediccion vs Real - {chem_target.upper()}")
                    st.plotly_chart(fig_pred, use_container_width=True)

                except Exception as e:
                    st.error(f"Error durante el entrenamiento: {e}")

        elif train_chem_btn:
            st.warning("Selecciona al menos una feature para entrenar el modelo.")


def _render_inference_section():
    """Seccion de inferencia quimica local."""
    st.subheader("Simulacion de Inferencia Quimica")

    col_inputs, col_output = st.columns([1, 1])

    with col_inputs:
        st.markdown("#### Parametros de Entrada Quimicos")

        # Inputs quimicos iniciales
        st.markdown("**Composicion Quimica Inicial:**")
        inp_valc = st.slider(
            "Carbono (valc %):",
            min_value=0.0, max_value=1.0, value=0.23, step=0.01,
            key='chem_inf_valc'
        )

        inp_valmn = st.slider(
            "Manganeso (valmn %):",
            min_value=0.0, max_value=2.0, value=1.27, step=0.01,
            key='chem_inf_valmn'
        )

        inp_valsi = st.slider(
            "Silicio (valsi %):",
            min_value=0.0, max_value=1.0, value=0.24, step=0.01,
            key='chem_inf_valsi'
        )

        inp_valp = st.slider(
            "Fosforo (valp %):",
            min_value=0.0, max_value=0.1, value=0.008, step=0.001,
            format="%.3f",
            key='chem_inf_valp'
        )

        inp_vals = st.slider(
            "Azufre (vals %):",
            min_value=0.0, max_value=0.1, value=0.015, step=0.001,
            format="%.3f",
            key='chem_inf_vals'
        )

        st.markdown("**Parametros de Proceso:**")
        inp_chem_carbon = st.number_input(
            "Carbono Inyectado (kg):",
            min_value=0.0, max_value=2000.0, value=0.0, step=50.0,
            key='chem_inf_carbon'
        )

        inp_chem_o2 = st.number_input(
            "Oxigeno Lance (m3):",
            min_value=0.0, max_value=5000.0, value=0.0, step=100.0,
            key='chem_inf_o2'
        )

        inp_chem_mat140107 = st.number_input(
            "Material 140107 (kg):",
            min_value=0.0, max_value=5000.0, value=0.0, step=100.0,
            key='chem_inf_mat140107'
        )

        inp_chem_mat360258 = st.number_input(
            "Material 360258 (kg):",
            min_value=0.0, max_value=5000.0, value=414.0, step=100.0,
            key='chem_inf_mat360258'
        )

        predict_chem_btn = st.button("Predecir Composicion Final", type="primary", key='predict_chem')

    with col_output:
        st.markdown("#### Resultado de Prediccion")

        if predict_chem_btn:
            _perform_chemical_prediction(
                inp_valc, inp_valmn, inp_valsi, inp_valp, inp_vals,
                inp_chem_carbon, inp_chem_o2, inp_chem_mat140107, inp_chem_mat360258
            )


def _perform_chemical_prediction(inp_valc, inp_valmn, inp_valsi, inp_valp, inp_vals,
                                  inp_chem_carbon, inp_chem_o2, inp_chem_mat140107, inp_chem_mat360258):
    """Realiza la prediccion quimica local."""
    if st.session_state.get('chem_model') is None:
        st.warning("Primero entrena un modelo en la seccion de 'Entrenamiento y Evaluacion'")
        return

    model = st.session_state.chem_model
    features = st.session_state.chem_features
    target = st.session_state.chem_target

    # Crear diccionario de input
    input_data = {
        'valc': inp_valc,
        'valmn': inp_valmn,
        'valsi': inp_valsi,
        'valp': inp_valp,
        'vals': inp_vals,
        'total_injected_carbon': inp_chem_carbon,
        'total_o2_lance': inp_chem_o2,
        'added_mat_140107': inp_chem_mat140107,
        'added_mat_360258': inp_chem_mat360258
    }

    # Crear DataFrame con solo las features que el modelo necesita (optimizado)
    feature_values = {col: input_data.get(col, 0.0) for col in features}
    input_df = pd.DataFrame([feature_values])

    # Realizar prediccion local
    try:
        prediction = model.predict(input_df)[0]

        # Mostrar prediccion
        st.metric(
            label=f"Composicion Final Estimada ({target.upper()})",
            value=f"{prediction:.4f} %"
        )

        # Indicador de especificacion
        status, label, description = chemical_spec_indicator(prediction, target)

        if status == "success":
            st.success(f"VERDE {label}: {description}")
        elif status == "error":
            st.error(f"ROJO {label}: {description}")
        else:
            st.info(f"INFO {label}: {description}")

        # Rangos de especificacion
        st.markdown("---")
        st.markdown("**Rangos de Especificacion Tipicos:**")
        for elem, (min_v, max_v) in CHEMICAL_SPECS.items():
            indicator = " <-- seleccionado" if elem == target else ""
            st.markdown(f"- **{elem.upper()}:** {min_v} - {max_v} %{indicator}")

    except Exception as e:
        st.error(f"Error en la prediccion: {e}")
