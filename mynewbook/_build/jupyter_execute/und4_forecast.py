#!/usr/bin/env python
# coding: utf-8

# # Unidad 4: Introducción a los modelos de pronóstico
# 
# ## 2.1 Introducción
# 
# Abordaremos métodos de suavizamiento, buscando la aproximación cuantitativa del pronóstico de series temporales.
# 
# ## 2.2 Objetivo
# 
# Conocer los modelos básicos de series de tiempo usando las técnicas apropiadas y comunes para ejemplificar las relaciones entre este tipo de datos. 
# 
# ## 2.3 Acción
# 
# En esta ocasión, se debe aplicar la metodología Holter-Winter y de suavizamiento a la variable tiempo.  Se invita al estudiante en ser claro en sus procedimientos, ya que la forma de justificarlos será evaluada
# 
# ### 2.3.1 Preparación de los datos
# 
# Previo al análisis detallado, es imperativo comprender el proceso de preparación de datos temporales para garantizar su idoneidad y coherencia en el estudio.

# In[1]:


# importando librerias

# librerias para la transformación de datos
import pandas as pd
import numpy as np
from datetime import datetime
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# libreria para visualización de datos
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


# recopilación de datos
df_data = pd.read_csv('./dataset.csv',sep=';')
df_data.head()


# In[3]:


def trf_data(df_data):
    
    # copia del dataframe
    df = df_data.copy()
    
    # transformación de fechas
    df['fechaoperacion'] = pd.to_datetime(df['fechaoperacion'], format='%d/%m/%Y')
    
    # agregando las columnas de fechas
    df['ano'] = df.apply(lambda x: x['fechaoperacion'].year ,axis=1)
    df['mes'] = df.apply(lambda x: x['fechaoperacion'].month ,axis=1)
    df['dia'] = df.apply(lambda x: x['fechaoperacion'].day ,axis=1)
    
    # selección de columnas
    df = df[['fechaoperacion','ano','mes','dia', 'hora1', 'hora2', 'hora3','hora4', 'hora5', 'hora6', 'hora7', 'hora8', 'hora9', 'hora10','hora11', 'hora12', 'hora13', 'hora14', 'hora15', 'hora16', 'hora17','hora18', 'hora19', 'hora20', 'hora21', 'hora22', 'hora23', 'hora24']]
    
    # Convertir la tabla
    list_id = [i.lower() for i in df.columns if not 'hora' in i]
    list_value = [i.lower() for i in df.columns if 'hora' in i]
    
    # pivotear la tabla
    df = df.melt(id_vars=list_id,value_vars=list_value,var_name='periodo',value_name='valor')
    
    return df


# In[4]:


# dataset trasnformado
df = trf_data(df_data)
df.head()


# In[5]:


# remuestreando la serie de tiempo a valores mensuales

# creando una copia del dataframe
df_vcm = df.copy()

#  remuestreando el dataframe a mensual
df_vcm = df_vcm.resample('M',on='fechaoperacion').mean().reset_index()
df_vcm = df_vcm[['fechaoperacion','valor']]

# mostrando el dataframe transformado
df_vcm.head()


# In[6]:


# Graficar la serie de tiempo y su tendencia estimada
plt.figure(figsize=(12, 8))
plt.plot(df_vcm['fechaoperacion'], df_vcm['valor'], label='Valor')
lowess = sm.nonparametric.lowess
smoothed = lowess(df_vcm['valor'], df_vcm['fechaoperacion'])
plt.plot(df_vcm['fechaoperacion'], smoothed[:, 1], label='Tendencia estimada')
plt.xlabel('Fecha')
plt.ylabel('Precio de bolsa mensual')
plt.legend()
plt.show()

# Crear una serie temporal
Ventas_ts = pd.Series(df_vcm['valor'].values, index=pd.date_range(start='2006-01', periods=len(df_vcm), freq='M'))
print(Ventas_ts)


# In[7]:


# Crear un boxplot de los datos
plt.figure(figsize=(10, 8))
sns.boxplot(x=Ventas_ts.index.month, y=Ventas_ts.values,color='skyblue')
plt.xlabel('Mes')
plt.ylabel('Valor')
plt.title('Boxplot de los valores mensuales')
plt.show()


# In[8]:


# Estabilización de la variabilidad
VentasLog = np.log(Ventas_ts)

# Aplicación del método Holt-Winters
model = ExponentialSmoothing(VentasLog, seasonal_periods=12, trend='add', seasonal='add')
fit = model.fit()

# Imprimir los parámetros del modelo
print(fit.params)

# Graficar los resultados
plt.figure(figsize=(12, 8))
plt.plot(VentasLog, label='Observado')
plt.plot(fit.fittedvalues, label='Predicho')
plt.title("Estimación Holt-Winters")
plt.xlabel('Año')
plt.ylabel('Observado/predicho')
plt.legend()
plt.show()


# In[9]:


# Predicción del siguiente año
forecast_next_year = fit.forecast(steps=24)

# Graficar la predicción
plt.figure(figsize=(12, 8))
plt.plot(VentasLog, label='Observado')
plt.plot(fit.fittedvalues, label='Entrenado')
plt.plot(pd.date_range(start='2024-01', periods=24, freq='M'), forecast_next_year, label='Predicción')
plt.title("Predicción del siguiente año")
plt.xlabel('Año')
plt.ylabel('Observado/predicho')
plt.legend()
plt.show()


# In[10]:


forecast_next_year


# In[11]:


predict_pbna = np.exp(forecast_next_year)
predict_pbna


# In[12]:


from sklearn.metrics import mean_squared_error, mean_absolute_error

# Selecciona un conjunto de datos de prueba
test_data = VentasLog[-12:]  # Supongamos que los últimos 12 meses son el conjunto de prueba

# Realiza la predicción para estos datos de prueba
forecast_next_year = fit.forecast(steps=12)

# Calcula el Error Cuadrático Medio (MSE)
mse = mean_squared_error(test_data, forecast_next_year)

# Calcula el Error Absoluto Medio (MAE)
mae = mean_absolute_error(test_data, forecast_next_year)

print("Error Cuadrático Medio (MSE):", mse)
print("Error Absoluto Medio (MAE):", mae)

