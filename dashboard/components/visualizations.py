"""
Funciones de visualizacion con Plotly.
"""
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Optional, Tuple


def plot_feature_importance(importance_df: pd.DataFrame, title: str = "Importancia de Variables"):
    """
    Genera grafico de barras horizontal de importancia de caracteristicas.

    Parameters:
    -----------
    importance_df : DataFrame con columnas 'Feature' e 'Importance'
    title : str - Titulo del grafico

    Returns:
    --------
    plotly Figure
    """
    fig = px.bar(
        importance_df.tail(15),
        x='Importance',
        y='Feature',
        orientation='h',
        title=title,
        color='Importance',
        color_continuous_scale='Blues'
    )
    fig.update_layout(height=400, showlegend=False)
    return fig


def plot_prediction_vs_real(y_test, y_pred, title: str = "Prediccion vs Real"):
    """
    Genera scatter plot de prediccion vs valores reales.

    Parameters:
    -----------
    y_test : array-like - Valores reales
    y_pred : array-like - Valores predichos
    title : str - Titulo del grafico

    Returns:
    --------
    plotly Figure
    """
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=y_test,
        y=y_pred,
        mode='markers',
        name='Predicciones',
        marker=dict(color='steelblue', opacity=0.6)
    ))

    # Linea de referencia (prediccion perfecta)
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    fig.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode='lines',
        name='Prediccion Perfecta',
        line=dict(color='red', dash='dash')
    ))

    fig.update_layout(
        title=title,
        xaxis_title='Valor Real',
        yaxis_title='Valor Predicho',
        height=400
    )

    return fig


def plot_distribution(y_test: pd.Series, y_pred: pd.Series, spec_range: Optional[Tuple[float, float]] = None, title: str = "Distribución de Valores (Real vs Predicho)"):
    """
    Genera un histograma superpuesto (con densidad) de valores reales y predichos.

    Parameters:
    -----------
    y_test : array-like - Valores reales
    y_pred : array-like - Valores predichos
    spec_range : tuple (min, max) - Rango de especificacion para sombrear
    title : str - Titulo del grafico

    Returns:
    --------
    plotly Figure
    """
    # Combinar datos en un DataFrame para Plotly Express
    df_plot = pd.DataFrame({
        'Valor': pd.concat([pd.Series(y_test), pd.Series(y_pred)]),
        'Tipo': ['Real'] * len(y_test) + ['Predicho'] * len(y_pred)
    })

    # Crear histograma superpuesto (usando densidad de probabilidad)
    fig = px.histogram(
        df_plot,
        x="Valor",
        color="Tipo",
        marginal="box",
        barmode="overlay",
        histnorm='probability density',
        opacity=0.6,
        color_discrete_map={
            'Real': 'red',
            'Predicho': 'blue'
        }
    )

    # Añadir área sombreada para la especificación
    if spec_range and len(spec_range) == 2 and all(v is not None for v in spec_range):
        spec_min, spec_max = spec_range

        # Sombreado para la zona de especificación
        fig.add_vrect(
            x0=spec_min,
            x1=spec_max,
            fillcolor="green",
            opacity=0.1,
            layer="below",
            line_width=0,
            name="Especificación"
        )

        # Línea para el límite mínimo de especificación
        fig.add_vline(x=spec_min, line_width=1, line_dash="dash", line_color="green", name="Mín. Espec.")

        # Línea para el límite máximo de especificación
        fig.add_vline(x=spec_max, line_width=1, line_dash="dash", line_color="green", name="Máx. Espec.")

    fig.update_layout(
        title=title,
        height=400,
        yaxis_title="Densidad de Probabilidad"
    )

    return fig


def plot_correlation_bar(corr_df: pd.DataFrame, target: str):
    """
    Genera grafico de barras de correlaciones con variable objetivo.

    Parameters:
    -----------
    corr_df : DataFrame con columnas 'Variable' y 'Correlacion'
    target : str - Nombre de la variable objetivo

    Returns:
    --------
    plotly Figure
    """
    # Altura dinamica: 25px por variable, minimo 400px, maximo 1200px
    num_vars = len(corr_df)
    dynamic_height = max(400, min(1200, num_vars * 25))

    fig = px.bar(
        corr_df,
        x='Correlacion',
        y='Variable',
        orientation='h',
        title=f'Correlacion con {target}',
        color='Correlacion',
        color_continuous_scale='RdBu_r',
        range_color=[-1, 1]
    )
    fig.update_layout(height=dynamic_height)
    return fig


def plot_histogram(df: pd.DataFrame, column: str):
    """
    Genera histograma de una variable.

    Parameters:
    -----------
    df : DataFrame con los datos
    column : str - Nombre de la columna a graficar

    Returns:
    --------
    plotly Figure
    """
    fig = px.histogram(
        df,
        x=column,
        nbins=50,
        title=f'Distribucion de {column}',
        color_discrete_sequence=['steelblue']
    )
    fig.update_layout(height=400)
    return fig


def plot_boxplot(df: pd.DataFrame, column: str):
    """
    Genera un Box Plot (Diagrama de Caja) de una variable.

    Parameters:
    -----------
    df : DataFrame con los datos
    column : str - Nombre de la columna a graficar

    Returns:
    --------
    plotly Figure
    """
    fig = px.box(
        df,
        y=column,
        title=f'Box Plot de {column}',
        color_discrete_sequence=['steelblue']
    )
    fig.update_layout(height=400)
    return fig


def plot_correlation_heatmap(df: pd.DataFrame, title: str = "Matriz de Correlacion"):
    """
    Genera un Mapa de Calor de la matriz de correlacion.

    Parameters:
    -----------
    df : DataFrame con los datos
    title : str - Titulo del grafico

    Returns:
    --------
    plotly Figure
    """
    # Seleccionar solo columnas numericas
    numeric_df = df.select_dtypes(include=['number'])

    # Calcular matriz de correlacion
    corr_matrix = numeric_df.corr()

    # Crear heatmap
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns.tolist(),
        y=corr_matrix.index.tolist(),
        colorscale='RdBu_r',
        zmin=-1,
        zmax=1,
        colorbar=dict(title='Correlacion')
    ))

    fig.update_layout(
        title=title,
        height=600,
        width=800,
        xaxis=dict(
            tickangle=45,
            tickfont=dict(size=10)
        ),
        yaxis=dict(
            tickfont=dict(size=10)
        ),
        margin=dict(l=100, r=50, t=80, b=120)
    )

    return fig