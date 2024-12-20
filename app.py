import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller, kpss
from hurst import compute_Hc
import yfinance as yf
import streamlit as st
import matplotlib.pyplot as plt

# Función para descargar datos históricos
def descargar_datos_oro(interval="15m", period="5d"):
    simbolo = "GC=F"  # Oro en Yahoo Finance (futuros)
    data = yf.download(tickers=simbolo, interval=interval, period=period)
    return data

# Función para pruebas de estacionariedad
def pruebas_estacionariedad(data):
    resultados = {}
    # Prueba ADF
    adf_result = adfuller(data)
    resultados['ADF'] = {'p-value': adf_result[1], 'estadístico': adf_result[0]}
    # Prueba KPSS
    try:
        kpss_result = kpss(data, regression="c", nlags="auto")
        resultados['KPSS'] = {'p-value': kpss_result[1], 'estadístico': kpss_result[0]}
    except ValueError as e:
        resultados['KPSS'] = {'p-value': None, 'error': str(e)}
    return resultados

# Función para calcular exponente de Hurst
def calcular_hurst(data):
    if len(data) < 100:
        return None
    try:
        H, _, _ = compute_Hc(data, kind="price", simplified=True)
    except FloatingPointError:
        return None
    return H

# Análisis por ventanas deslizantes
def analizar_estacionariedad(data, ventana_dias=2):
    resultados = []
    data['date'] = data.index.date
    fechas_unicas = sorted(data['date'].unique())
    for i in range(len(fechas_unicas) - ventana_dias + 1):
        ventana = fechas_unicas[i:i + ventana_dias]
        subset = data[data['date'].isin(ventana)]
        precios = subset['Close'][subset['Close'] > 0].dropna()
        if len(precios) < 2:
            continue
        retornos = precios.pct_change().dropna()
        estacionariedad = pruebas_estacionariedad(retornos)
        hurst = calcular_hurst(retornos)
        resultados.append({
            "Inicio Ventana": ventana[0],
            "Fin Ventana": ventana[-1],
            "ADF p-value": estacionariedad['ADF']['p-value'],
            "KPSS p-value": estacionariedad['KPSS'].get('p-value'),
            "Hurst Exponent": hurst,
            "Volatilidad": retornos.std()
        })
    return pd.DataFrame(resultados)

# Generar reporte simplificado para patrón
def generar_reporte(df):
    if df.empty:
        return "No hay suficiente información para realizar el análisis."
    estacionario_adf = (df['ADF p-value'] < 0.05).mean() * 100
    estacionario_kpss = (df['KPSS p-value'] > 0.05).mean() * 100
    hurst_estacionario = ((df['Hurst Exponent'] > 0.4) & (df['Hurst Exponent'] < 0.6)).mean() * 100
    return f"""
    **Resumen del análisis**:
    - Probabilidad de que el oro sea estacionario (ADF): {estacionario_adf:.2f}%
    - Probabilidad de que el oro sea estacionario (KPSS): {estacionario_kpss:.2f}%
    - Exponente de Hurst indicando estacionariedad: {hurst_estacionario:.2f}%
    """

# Visualización con gráficos
def graficar(df):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df['Inicio Ventana'], df['ADF p-value'], label='ADF p-value', marker='o')
    ax.plot(df['Inicio Ventana'], df['KPSS p-value'], label='KPSS p-value', marker='o')
    ax.set_title("Evolución de las pruebas de estacionariedad")
    ax.axhline(0.05, color='red', linestyle='--', label='Umbral de estacionariedad (0.05)')
    ax.set_xlabel("Fecha")
    ax.set_ylabel("p-value")
    ax.legend()
    st.pyplot(fig)

# Página principal de Streamlit
st.title("Análisis de Estacionariedad del Oro")
st.write("Este análisis evalúa la probabilidad de que el oro se comporte de forma estacionaria en el próximo día.")

# Descargar datos históricos
st.write("### Descargando datos...")
datos = descargar_datos_oro(interval="15m", period="5d")
if datos.empty:
    st.error("No se pudieron descargar los datos del oro. Intenta nuevamente más tarde.")
else:
    st.success("Datos descargados exitosamente.")
    st.write(datos.head())

    # Análisis de estacionariedad
    st.write("### Analizando datos...")
    resultados = analizar_estacionariedad(datos, ventana_dias=2)
    st.write(resultados)

    # Generar reporte
    reporte = generar_reporte(resultados)
    st.write(reporte)

    # Gráficos
    st.write("### Visualización de resultados")
    graficar(resultados)
