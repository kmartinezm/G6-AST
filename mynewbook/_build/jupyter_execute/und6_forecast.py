#!/usr/bin/env python
# coding: utf-8

# # Unidad 6: Regresión en series de tiempo
# 
# ## 6.1 Introducción
# 
# Combinaremos modelos causales con datos de series de tiempo, a través de algoritmos de aprendizaje estadístico.
# 
# ## 6.2 Objetivo
# 
# Combinar métodos autorregresivos y de aprendizaje estadístico mediante el enfoque de la clasificación para analizar datos de series de tiempo.
# 
# ## 6.3 Acción
# 
# Esta vez deberás aplicar el algoritmo Facebook´s Prophet, y si es viable la justificación para la variable en serie de tiempo vista como una regresión. Esto último, complementa los modelos planteados anteriormente y el ajujste a un modelo lineal y estacionario.
# 
# ### 6.3.1 Preparación de los datos
# 
# Previo al análisis detallado, es imperativo comprender el proceso de preparación de datos temporales para garantizar su idoneidad y coherencia en el estudio.
# cia en el estudio.

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
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA

# libreria para visualización de datos
import matplotlib.pyplot as plt
import seaborn as sns
from prophet import Prophet


# #### 6.3.1.1 Recopilación de los datos
# 
# En este bloque, se carga el conjunto de datos desde un archivo CSV utilizando la biblioteca pandas. Se especifica el delimitador de campos y se visualizan las primeras filas del dataframe para verificar que los datos se han cargado correctamente.

# In[76]:


# recopilación de datos
df_data = pd.read_csv('./dataset.csv',sep=';')
df_data.head()


# #### 6.3.1.2 Transformación de Datos
# 
# Este bloque define una función para transformar el dataframe original. Se realiza una copia del dataframe, se convierte la columna de fechas a un formato de fecha adecuado, y se agregan nuevas columnas para el año, mes y día. Luego, se seleccionan y reorganizan las columnas, y finalmente, se pivotea la tabla para facilitar su uso en análisis posteriores.

# In[77]:


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


# In[78]:


# dataset trasnformado
df = trf_data(df_data)
df.head()


# In[79]:


df.info()


# #### 6.3.1.3 Remuestreo de la serie temporal
# 
# En este bloque, se crea una copia del dataframe transformado y se remuestrean los datos a valores mensuales. Se agrupan los datos por fecha y se calcula la media mensual. Luego, se seleccionan las columnas relevantes y se visualiza el nuevo dataframe remuestreado.

# In[80]:


# remuestreando la serie de tiempo a valores mensuales

# creando una copia del dataframe
df_vcm = df.copy()

df_vcm = df_vcm[['ano','mes','valor']]

# creando el groupby
df_vcm = df_vcm.groupby(by=['ano','mes']).mean().reset_index()

# 
df_vcm['fechaoperacion'] = df_vcm.apply(lambda x: datetime(int(x['ano']),int(x['mes']),1),axis=1)

#  remuestreando el dataframe a mensual
df_vcm = df_vcm[['fechaoperacion','valor']]

# mostrando el dataframe transformado
df_vcm


# ### 6.3.2 Modelado Prophet
# 
# En esta sección, se realizará el modelado de series de tiempo utilizando el algoritmo Facebook's Prophet. A continuación se describen los pasos que se llevarán a cabo para ajustar y predecir los datos de la serie temporal.
# 
# #### 6.3.2.1 Preparación del DataFrame
# 
# El primer paso crucial en nuestro análisis es la preparación adecuada de los datos. Para ello:
# 
# - Creación de una copia del DataFrame: Se generará una copia del DataFrame original. Este paso es esencial para mantener la integridad de los datos originales, evitando modificaciones accidentales durante el proceso de análisis.
# - Renombrar columnas: Adaptaremos el formato del DataFrame para que sea compatible con los requisitos específicos del modelo Prophet. Esto implica renombrar la columna que contiene las fechas (fechaoperacion) a ds y la columna de valores (valor) a y. Este cambio es fundamental para que el modelo pueda interpretar correctamente los datos.

# In[81]:


# creando copia del dataframe
df_ph = df_vcm.copy()

# Formatear el DataFrame de acuerdo a los requisitos de Prophet
df_ph.rename(columns={'fechaoperacion': 'ds', 'valor': 'y'}, inplace=True)


# #### 6.3.2.2 Inicialización y ajuste del modelo Prophet
# Una vez preparados los datos, procederemos a:
# 
# - Inicialización del modelo: Se inicializará una instancia del modelo Prophet. Prophet es conocido por su capacidad para manejar datos con estacionalidades fuertes y tendencias múltiples, lo que lo convierte en una herramienta robusta para el análisis de series de tiempo.
# 
# - Ajuste del modelo: Ajustaremos el modelo a los datos formateados. Este proceso permite que Prophet aprenda de los datos históricos, capturando patrones y tendencias subyacentes.

# In[82]:


# Inicializar y ajustar el modelo Prophet
model_ph = Prophet()
model_ph.fit(df_ph)


# #### 6.3.2.3 Generación de predicciones futuras
# Con el modelo ajustado, el siguiente paso es la predicción:
# 
# - Extensión del horizonte temporal: Crearemos un nuevo DataFrame que extienda el horizonte temporal de nuestros datos en 12 meses hacia adelante. Este DataFrame es esencial para que el modelo pueda generar predicciones para un período futuro específico.
# - Generación de predicciones: Utilizaremos el modelo ajustado para realizar predicciones sobre el período extendido. Esto nos permitirá visualizar posibles tendencias y comportamientos futuros de la serie temporal.

# In[83]:


# Hacer una predicción para el futuro
future = model_ph.make_future_dataframe(periods=12, freq='M')  # Predecir 12 meses hacia adelante

forecast_ph = model_ph.predict(future)


# #### 6.3.2.4 Visualización de resultados
# Para interpretar y comunicar eficazmente los resultados del modelo, se procederá a:
# 
# - Gráfico de predicciones: Se generará un gráfico que muestre las predicciones realizadas por el modelo. Esta visualización facilitará la comprensión de las proyecciones futuras y permitirá identificar posibles tendencias y puntos de interés.
# - Análisis de componentes: Adicionalmente, se graficarán los componentes de la predicción, tales como la tendencia y la estacionalidad. Esta descomposición es vital para entender las contribuciones individuales de diferentes factores a la serie temporal, proporcionando una visión más detallada y completa.

# In[84]:


# Graficar los resultados
model_ph.plot(forecast_ph)
plt.show()


# In[85]:


# Graficar los componentes de la predicción
model_ph.plot_components(forecast_ph)
plt.show()


# #### 6.3.2.5 Conclusión
# 
# El uso del modelo Prophet nos permite no solo ajustar nuestros datos históricos de manera precisa, sino también proyectar tendencias futuras con un alto grado de confianza. Este enfoque integral facilita la toma de decisiones informadas basadas en datos, permitiendo a la organización anticiparse a cambios y planificar estratégicamente.

# In[86]:


# transformación de la predicción para gráfico de modelos
forecast_ph_pred = forecast_ph.copy()

cond = forecast_ph_pred['ds'] > datetime(2023,12,1)

forecast_ph_pred = forecast_ph_pred[cond]


# ### 6.3.4 Comparación de modelos
# 
# En esta sección, se llevará a cabo una comparación detallada entre tres modelos de series de tiempo: Prophet, ARIMA y Holt-Winters (ETS). El objetivo es evaluar el rendimiento de cada modelo en la predicción de los últimos 12 meses de la serie temporal.
# 
# #### 6.3.4.1 Modelo ARIMA
# 
# El modelo ARIMA (AutoRegressive Integrated Moving Average) es una técnica ampliamente utilizada para el análisis y predicción de series de tiempo. Este modelo combina tres componentes: autorregresivo (AR), diferenciación (I) y media móvil (MA), permitiendo capturar diversas características de la serie temporal como tendencias y patrones estacionales. A continuación, se describe el proceso de ajuste y predicción utilizando el modelo ARIMA.
# 
# Ajuste del modelo ARIMA:
# 
# - Se ajusta un modelo ARIMA configurado con parámetros específicos, determinados a partir de análisis previos de los datos.
# - Se generan predicciones para los próximos 12 meses, incluyendo intervalos de confianza para evaluar la precisión de las predicciones.

# In[87]:


# Ajustar un modelo ARIMA(4,1,3) a la serie temporal, usando los parámetros sugeridos por los gráficos ACF y PACF
model_ar = ARIMA(df_vcm['valor'].dropna(), order=(4, 1, 3))
model_ar = model_ar.fit()

# Mostrar el resumen del modelo ajustado
model_ar.summary()


# In[88]:


# Hacer predicciones con el modelo ARIMA ajustado para los próximos 12 períodos
forecast_12 = model_ar.get_forecast(steps=12)
forecast_conf_int_12 = forecast_12.conf_int()

# Crear un DataFrame para visualizar las predicciones y los intervalos de confianza para 24 meses
forecast_ar = pd.DataFrame({
    'Predicted': forecast_12.predicted_mean,
    'Lower CI': forecast_conf_int_12.iloc[:, 0],
    'Upper CI': forecast_conf_int_12.iloc[:, 1]
})

forecast_ar


# #### 6.3.4.2 Modelo Holt-Winters (ETS)
# 
# El modelo Holt-Winters, también conocido como ETS (Error, Trend, Seasonal), es una metodología eficaz para la predicción de series de tiempo que presentan tendencias y estacionalidades. Este modelo utiliza componentes aditivos o multiplicativos para capturar la variabilidad en los datos, proporcionando una estructura flexible y precisa para el análisis temporal. A continuación, se describe el proceso de ajuste y predicción utilizando el modelo Holt-Winters.
# 
# Ajuste del modelo Holt-Winters (ETS):
# 
# - Se aplica el método Holt-Winters a los datos, utilizando componentes aditivos para capturar tanto la tendencia como la estacionalidad.
# - Se generan predicciones para los próximos 12 meses, ofreciendo una visión anticipada del comportamiento futuro de la serie temporal.

# In[89]:


# Crear una serie temporal
precio_ts = pd.Series(df_vcm['valor'].values, index=pd.date_range(start='2006-01', periods=len(df_vcm), freq='M'))

# Estabilización de la variabilidad
preciolog = np.log(precio_ts)

# Aplicación del método Holt-Winters
model_hw = ExponentialSmoothing(preciolog, seasonal_periods=12, trend='add', seasonal='add')
fit_hw = model_hw.fit()

# Predicción del siguiente año
forecast_hw = fit_hw.forecast(steps=12)
forecast_hw = np.exp(forecast_hw)

forecast_hw


# #### 6.3.4.3 Visualización de Pronósticos de los modelos
# 
# En esta etapa, se procederá a la visualización comparativa de las predicciones obtenidas de los modelos Prophet, ARIMA y Holt-Winters (ETS). Esta visualización es para evaluar la precisión y el desempeño de cada modelo en relación con los datos reales. Además, se incluirán intervalos de confianza para las predicciones de Prophet y ARIMA, permitiendo así una comprensión más completa de la incertidumbre asociada a estas estimaciones.
# 
# - Se crea un gráfico que compara las predicciones de los tres modelos con los datos reales.
# - Se incluyen intervalos de confianza para las predicciones de Prophet y ARIMA para evaluar la incertidumbre de las predicciones.

# In[90]:


# Crear el gráfico comparativo
plt.figure(figsize=(14, 7))

prediction_dates_12 = pd.date_range(start=df_vcm['fechaoperacion'].iloc[-1] + pd.DateOffset(months=1), periods=12, freq='M')

# Datos reales
plt.plot(df_vcm['fechaoperacion'], df_vcm['valor'], label='Datos Reales', color='black')

# Predicciones Prophet
plt.plot(pd.date_range(start='2024-01', periods=12, freq='M'), forecast_ph_pred['yhat'], label='Modelo Prophet', color='blue')
plt.fill_between(pd.date_range(start='2024-01', periods=12, freq='M'), forecast_ph_pred['yhat_lower'], forecast_ph_pred['yhat_upper'], color='blue', alpha=0.2)

# Predicciones ARIMA
plt.plot(prediction_dates_12, forecast_ar['Predicted'], label='Modelo ARIMA', color='green')
plt.fill_between(prediction_dates_12, forecast_ar['Lower CI'], forecast_ar['Upper CI'], color='green', alpha=0.2)

# Predicciones ETS
plt.plot(pd.date_range(start='2024-01', periods=12, freq='M'), forecast_hw, label='Modelo ETS', color='orange')

# Formatear el gráfico
plt.xlabel('Fecha')
plt.ylabel('Valor')
plt.title('Comparación de Predicciones: Prophet vs ARIMA vs ETS')
plt.legend()
plt.show()

