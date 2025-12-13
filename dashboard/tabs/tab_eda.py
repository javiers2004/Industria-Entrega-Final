"""
Tab 1: EDA y Resumen del Dataset.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from dashboard.utils.cached_loader import load_data_for_eda
from dashboard.components.visualizations import (
    plot_correlation_bar,
    plot_histogram,
    plot_boxplot,
    plot_correlation_heatmap
)


# Mapeo de datasets disponibles
DATASET_OPTIONS = {
    "Temperatura": "dataset_final_temp.csv",
    "Quimica": "dataset_final_chemical.csv"
}


def render_eda_tab():
    """
    Renderiza el tab de Analisis Exploratorio de Datos.
    Incluye selector de dataset y multiples subsecciones de analisis.
    """
    st.header("Analisis Exploratorio de Datos (EDA)")

    # Selector de dataset
    st.subheader("Seleccion de Dataset")
    selected_dataset = st.selectbox(
        "Selecciona el dataset a analizar:",
        options=list(DATASET_OPTIONS.keys()),
        key='eda_dataset_selector'
    )

    # Cargar el dataset seleccionado
    file_name = DATASET_OPTIONS[selected_dataset]
    df = load_data_for_eda(file_name)

    # Validar que el DataFrame se cargo correctamente
    if df is None or df.empty:
        st.error(f"No se pudo cargar el dataset: {file_name}")
        return

    st.success(f"Dataset '{selected_dataset}' cargado correctamente.")

    # Renderizar subsecciones de EDA
    st.divider()
    _render_summary_metrics(df)

    st.divider()
    _render_correlation_analysis(df, selected_dataset)

    st.divider()
    _render_distribution_analysis(df)

    st.divider()
    _render_univariate_advanced(df)

    st.divider()
    _render_multivariate_analysis(df)


def _render_summary_metrics(df: pd.DataFrame):
    """1. Renderiza metricas de resumen del dataset."""
    st.subheader("1. Resumen General")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Filas", f"{df.shape[0]:,}")
    with col2:
        st.metric("Columnas", df.shape[1])
    with col3:
        st.metric("Valores Nulos", df.isnull().sum().sum())
    with col4:
        memory_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
        st.metric("Memoria (MB)", f"{memory_mb:.2f}")

    st.markdown("#### Estadisticas Descriptivas")
    st.dataframe(df.describe().T, use_container_width=True)


def _render_correlation_analysis(df: pd.DataFrame, selected_dataset: str):
    """2. Renderiza analisis de correlaciones con variable objetivo."""
    st.subheader("2. Correlaciones con Variable Objetivo")

    # Determinar targets disponibles segun el dataset
    if selected_dataset == "Temperatura":
        available_targets = ['target_temperature']
    else:
        # Para Quimica, buscar columnas que empiecen con 'target_'
        available_targets = [col for col in df.columns if col.startswith('target_')]

    if not available_targets:
        st.warning("No se encontraron variables objetivo en el dataset seleccionado.")
        return

    selected_corr_target = st.selectbox(
        "Selecciona la Variable Objetivo de Correlacion:",
        options=available_targets,
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
    else:
        st.warning(f"La variable '{selected_corr_target}' no es numerica o no existe en el dataset.")


def _render_distribution_analysis(df: pd.DataFrame):
    """3. Renderiza analisis de distribuciones con histograma."""
    st.subheader("3. Distribuciones - Histograma")

    # Columnas de fecha (excluir del selector de features)
    date_cols = ['fecha_inicio', 'fecha_fin']
    available_dates = [col for col in date_cols if col in df.columns]

    # Obtener todas las columnas numericas (excluir fechas)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [col for col in numeric_cols if col not in date_cols]

    if not numeric_cols:
        st.warning("No hay columnas numericas en el dataset.")
        return

    # Dos selectores lado a lado
    col1, col2 = st.columns(2)
    with col1:
        selected_hist_var = st.selectbox(
            "Selecciona una Variable:",
            options=numeric_cols,
            key='hist_var'
        )
    with col2:
        # Preseleccionar fecha_inicio si existe
        default_date_idx = 0 if 'fecha_inicio' in available_dates else None
        selected_date = st.selectbox(
            "Selecciona Fecha:",
            options=available_dates,
            index=default_date_idx,
            key='hist_date'
        ) if available_dates else None

    # Mostrar grafico segun seleccion
    if selected_date:
        # Grafico temporal: media diaria de la variable seleccionada
        df_temp = df.copy()
        df_temp[selected_date] = pd.to_datetime(df_temp[selected_date], errors='coerce')
        df_temp = df_temp.dropna(subset=[selected_date])

        # Extraer solo la fecha (sin hora) y agrupar calculando la media
        df_temp['fecha'] = df_temp[selected_date].dt.date
        df_agg = df_temp.groupby('fecha')[selected_hist_var].mean().reset_index()

        fig = px.line(
            df_agg,
            x='fecha',
            y=selected_hist_var,
            title=f'{selected_hist_var} - Media diaria ({selected_date})',
            markers=True
        )
        fig.update_layout(height=400)
    else:
        # Histograma de variable numerica
        fig = plot_histogram(df, selected_hist_var)

    st.plotly_chart(fig, use_container_width=True)


def _render_univariate_advanced(df: pd.DataFrame):
    """4. Renderiza analisis univariado avanzado con Box Plot."""
    st.subheader("4. Univariado Avanzado - Box Plot")

    # Obtener todas las columnas numericas
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if not numeric_cols:
        st.warning("No hay columnas numericas en el dataset.")
        return

    selected_box_var = st.selectbox(
        "Selecciona una Variable para el Box Plot:",
        options=numeric_cols,
        key='box_var'
    )

    if selected_box_var:
        fig_box = plot_boxplot(df, selected_box_var)
        st.plotly_chart(fig_box, use_container_width=True)

        # Mostrar estadisticas adicionales
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Min", f"{df[selected_box_var].min():.4f}")
        with col2:
            st.metric("Q1 (25%)", f"{df[selected_box_var].quantile(0.25):.4f}")
        with col3:
            st.metric("Mediana", f"{df[selected_box_var].median():.4f}")
        with col4:
            st.metric("Q3 (75%)", f"{df[selected_box_var].quantile(0.75):.4f}")
        with col5:
            st.metric("Max", f"{df[selected_box_var].max():.4f}")


def _render_bivariate_key(df: pd.DataFrame, selected_dataset: str):
    """5. Renderiza analisis bivariado clave con Scatter Plot."""
    st.subheader("5. Bivariado Clave - Scatter Plot")

    x_col = 'total_o2_lance'
    y_col = 'target_temperature'

    # Verificar que las columnas existen
    if x_col not in df.columns or y_col not in df.columns:
        st.info(
            f"Las columnas '{x_col}' y/o '{y_col}' no estan disponibles en el dataset "
            f"'{selected_dataset}'. Este grafico esta disenado para el dataset de Temperatura."
        )
        return

    fig_scatter = px.scatter(
        df,
        x=x_col,
        y=y_col,
        title=f'Relacion entre {x_col} y {y_col}',
        color_discrete_sequence=['steelblue'],
        opacity=0.6
    )

    fig_scatter.update_layout(
        height=450,
        xaxis_title=x_col,
        yaxis_title=y_col
    )

    st.plotly_chart(fig_scatter, use_container_width=True)

    # Mostrar correlacion entre las dos variables
    corr_value = df[x_col].corr(df[y_col])
    st.metric("Correlacion de Pearson", f"{corr_value:.4f}")


def _render_multivariate_analysis(df: pd.DataFrame):
    """5. Renderiza analisis multivariado con Heatmap de correlaciones."""
    st.subheader("5. Multivariado - Heatmap de Correlaciones")

    fig_heatmap = plot_correlation_heatmap(df, "Matriz de Correlacion del Dataset")
    st.plotly_chart(fig_heatmap, use_container_width=True)

    # Informacion adicional
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    st.caption(f"Se muestran las correlaciones entre las {len(numeric_cols)} variables numericas del dataset.")
