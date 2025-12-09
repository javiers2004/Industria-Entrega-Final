"""
Tab 1: EDA y Resumen del Dataset.
"""
import streamlit as st
import pandas as pd
import numpy as np

from src.config import INPUT_FEATURES, CHEMICAL_TARGETS
from components.visualizations import plot_correlation_bar, plot_histogram


def render_eda_tab(df: pd.DataFrame):
    """
    Renderiza el tab de Analisis Exploratorio de Datos.

    Parameters:
    -----------
    df : DataFrame con los datos del dataset
    """
    st.header("Analisis Exploratorio de Datos (EDA)")

    _render_summary_metrics(df)
    st.divider()
    _render_correlation_analysis(df)
    st.divider()
    _render_distribution_analysis(df)


def _render_summary_metrics(df: pd.DataFrame):
    """Renderiza metricas de resumen del dataset."""
    st.subheader("Resumen General del Dataset")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Filas", df.shape[0])
    with col2:
        st.metric("Columnas", df.shape[1])
    with col3:
        st.metric("Valores Nulos", df.isnull().sum().sum())

    st.markdown("#### Estadisticas Descriptivas")
    st.dataframe(df.describe().T, use_container_width=True)


def _render_correlation_analysis(df: pd.DataFrame):
    """Renderiza analisis de correlaciones."""
    st.subheader("Analisis de Correlaciones")

    correlation_targets = ['target_temperature'] + CHEMICAL_TARGETS
    selected_corr_target = st.selectbox(
        "Selecciona la Variable Objetivo de Correlacion:",
        options=correlation_targets,
        key='corr_target'
    )

    # Calcular correlaciones
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if selected_corr_target in numeric_cols:
        corr_with_target = df[numeric_cols].corr()[selected_corr_target].drop(selected_corr_target)
        corr_df = pd.DataFrame({
            'Variable': corr_with_target.index,
            'Correlacion': corr_with_target.values
        }).sort_values('Correlacion', key=abs, ascending=False)

        fig_corr = plot_correlation_bar(corr_df, selected_corr_target)
        st.plotly_chart(fig_corr, use_container_width=True)


def _render_distribution_analysis(df: pd.DataFrame):
    """Renderiza analisis de distribuciones."""
    st.subheader("Analisis de Distribuciones")

    available_features = [f for f in INPUT_FEATURES if f in df.columns]
    selected_hist_var = st.selectbox(
        "Selecciona una Variable para el Histograma:",
        options=available_features,
        key='hist_var'
    )

    if selected_hist_var:
        fig_hist = plot_histogram(df, selected_hist_var)
        st.plotly_chart(fig_hist, use_container_width=True)
