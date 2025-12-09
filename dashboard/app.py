import streamlit as st
import pandas as pd
import requests
import json
from pathlib import Path

# Obtener la ruta al directorio del proyecto
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "dataset_final_acero.csv"

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(page_title="Predicción EAF - Grupo 13", layout="wide")

st.title("🏭 Simulador de Horno de Arco Eléctrico")
st.markdown("### Optimización de Temperatura Final de Colada")

# --- COLUMNA LATERAL: CONTROLES (INPUTS) ---
st.sidebar.header("Parámetros de Entrada")

def user_input_features():
    # 1. Energía (Fundamental)
    power = st.sidebar.slider('Energía Activa Promedio (MW)', 5.0, 100.0, 45.0)
    time_on = st.sidebar.slider('Tiempo de Encendido (min)', 10, 120, 45)
    
    # 2. Materiales CLAVE (Los del Top 10 que te salieron)
    # Nota: Ajusta los nombres si descubres qué es el '140107'
    mat_140107 = st.sidebar.number_input('Adición Mat 140107 (kg)', 0, 5000, 0)
    mat_705043 = st.sidebar.number_input('Adición Mat 705043 (kg)', 0, 2000, 0)
    mat_360258 = st.sidebar.number_input('Adición Mat 360258 (kg)', 0, 2000, 0)
    
    # 3. Química Inicial (Valores típicos por defecto)
    val_cr = st.sidebar.slider('Cromo Inicial (%)', 0.0, 5.0, 0.15)
    val_p = st.sidebar.slider('Fósforo Inicial (%)', 0.0, 0.1, 0.01)
    
    # Guardamos esto en un diccionario simple
    data = {
        'power_active_mean': power,
        'total_power_on_time': time_on,
        'added_mat_140107': mat_140107,
        'added_mat_705043': mat_705043,
        'added_mat_360258': mat_360258,
        'valcr': val_cr,
        'valp': val_p
        # ... aquí podrías añadir más inputs si quieres
    }
    return data

input_dict = user_input_features()

# --- PANEL PRINCIPAL ---
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Resumen de la Colada")
    # Mostramos los datos que va a enviar el usuario
    st.json(input_dict)

# --- LÓGICA DE PREDICCIÓN ---
# Necesitamos cargar la estructura del dataset original para rellenar los huecos
# (XGBoost necesita recibir TODAS las columnas, aunque sean 0)
try:
    # Leemos solo la cabecera para estructura
    df_structure = pd.read_csv(DATA_PATH, nrows=1)
    feature_columns = [c for c in df_structure.columns if c not in ['heatid', 'target_temperature']]
except:
    st.error(f"⚠️ No encuentro el dataset en: {DATA_PATH}")
    feature_columns = []

if st.button('🔥 CALCULAR TEMPERATURA FINAL'):
    if feature_columns:
        # CORRECCIÓN AQUÍ: Inicializamos con 0.0 (float) en vez de 0 (int)
        # Esto evita el error de "incompatible dtype"
        input_df = pd.DataFrame(0.0, index=[0], columns=feature_columns)
        
        # Rellenar con lo que eligió el usuario
        for key, value in input_dict.items():
            if key in input_df.columns:
                # Ahora es seguro asignar floats
                input_df.at[0, key] = float(value)
        
        # Convertir a diccionario para BentoML
        payload = input_df.to_dict(orient='records')

        # Llamada a la API
        try:
            response = requests.post(
                "http://localhost:3000/predict",
                json={"inputs": payload}, 
                headers={"content-type": "application/json"}
            )
            
            if response.status_code == 200:
                temp_pred = response.json()
                
                with col2:
                    st.success("✅ Cálculo Exitoso")
                    # Formato bonito con 2 decimales
                    st.metric(label="Temperatura Estimada", value=f"{temp_pred:.2f} °C")
                    
                    # Semáforo de calidad
                    if temp_pred > 1680:
                        st.error("⚠️ PELIGRO: Sobrecalentamiento extremo")
                    elif temp_pred > 1650:
                        st.warning("🔥 Temperatura alta (consumo extra de energía)")
                    elif temp_pred < 1580:
                        st.info("❄️ Temperatura baja (riesgo en colada continua)")
                    else:
                        st.success("🎯 Temperatura Óptima")
                        
            else:
                st.error(f"Error del servidor: {response.text}")
                
        except Exception as e:
            st.error(f"No se pudo conectar con BentoML. Asegúrate de que la terminal 'bentoml serve' está corriendo.\nError: {e}")