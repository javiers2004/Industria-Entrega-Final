"""
Funciones de visualizacion con Plotly.
"""
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


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
    fig.update_layout(height=500)
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
