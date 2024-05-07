#!/usr/bin/env python
# coding: utf-8

# # Unidad 2: Estructura de los datos en series de tiempo
# 
# ## 2.1 Introducción
# 
# Dado que una serie de tiempo es un conjunto de observaciones sobre los valores que toma una variable (cuantitativa) a través del tiempo, exploraremos las tendencias o cambios que se reflejan y afectan su comportamiento.
# 
# ## 2.2 Acción
# 
# Ahora, en esta Unidad 2, se debe continuar con los datos presentados en dicho entregable y se debe evidenciar, en una de las variables en el tiempo, la aproximación en promedio móvil, en rezagos y en estacionalidad. Todo lo anterior, a través de funciones y gráficas que permitan detectar patrones y ciclos de la variable.
# 
# ### 2.2.1 Preparación de los datos

# In[1]:


# importando librerias

# librerias para la transformación de datos
import pandas as pd
import numpy as np
from datetime import datetime
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller

# libreria para visualización de datos
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


# recopilación de datos
df_data = pd.read_csv('./dataset.csv',sep=';')
df_data.head()


# Se realiza una transformacion de los datos separandoos por año,mes,dia y periodo, esto con el fin de poder graficar y analizar mas detalladamente el dataset.

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


# Se crea el grafico original donde en el eje x esta representando las fechas y en el eje y los valores de bolsa nacional del kWh.

# In[6]:


# gráficando la serie de tiempo

# creando lienzo
plt.figure(figsize=(10,8))

# creando gráfico de serie de tiempo
plt.plot(df_vcm['fechaoperacion'],df_vcm['valor'],label='pbna')

# creando los titulos
plt.xlabel('fecha')
plt.ylabel('Precio de Bolsa Nacional ($/kWh)')
plt.title('Serie de tiempo - Precio de Bolsa Nacional Mensual')
plt.legend(loc='upper left')

# mostrando gráfico
plt.show()


# Para el analisis de Media movil, estacionalidad, Rezago se analizara para el caso mensual, trimestral y anual, con el fin de determinar si existen patrones en cada uno de estos intervalos de tiempo.
# 
# Para el caso de media movil se realizan 4 medias moviles: MV3,MV6,MV9,MV12. Ya que al tener tantos datos en el intervalo de tiempo mensual puede ser interesante revisar como varia la media movil conforme mas datos antiguos se analizan.

# In[7]:


# realizando el promedio movil de la serie de tiempo

#  realizando el promedio movil de los ultimos tres meses
df_vcm['ma3'] = df_vcm['valor'].rolling(window=3).mean().shift(1)

# monstrando el dataframe con el promedio movil
df_vcm.head(10)


# In[8]:


# gráficando la serie de tiempo

# creando lienzo
plt.figure(figsize=(10,8))

# creando gráfico de serie de tiempo
plt.plot(df_vcm['fechaoperacion'],df_vcm['valor'],label='pbna')
plt.plot(df_vcm['fechaoperacion'],df_vcm['ma3'],label='MA3')

# creando los titulos
plt.xlabel('fecha')
plt.ylabel('Precio de Bolsa Nacional ($/kWh)')
plt.title('Serie de tiempo - Precio de Bolsa Nacional Mensual')
plt.legend(loc='upper left')

# mostrando gráfico
plt.show()


# Como se puede observar la media movil con n=3 es muy similar a la grafica original, por lo que esta media esta fuertemente influenciada por datos aleatorios, por lo que es necesario que se analicen valores de medias moviles mas altos.

# In[9]:


# realizando el promedio movil de la serie de tiempo

#  realizando el promedio movil de los ultimos tres meses
df_vcm['ma6'] = df_vcm['valor'].rolling(window=6).mean().shift(1)
df_vcm['ma9'] = df_vcm['valor'].rolling(window=9).mean().shift(1)
df_vcm['ma12'] = df_vcm['valor'].rolling(window=12).mean().shift(1)

# monstrando el dataframe con el promedio movil
df_vcm.head(15)


# Se realizan las graficas de cada una de las medias moviles:

# In[10]:


# gráficando la serie de tiempo

# creando lienzo
plt.figure(figsize=(12,10))

# creando gráfico de serie de tiempo
plt.subplot(2,2,1)
plt.plot(df_vcm['fechaoperacion'],df_vcm['valor'],label='original')
plt.plot(df_vcm['fechaoperacion'],df_vcm['ma3'],label='MA3')
plt.legend(loc='upper left')

plt.subplot(2,2,2)
plt.plot(df_vcm['fechaoperacion'],df_vcm['valor'],label='original')
plt.plot(df_vcm['fechaoperacion'],df_vcm['ma6'],label='MA6',color='red')
plt.legend(loc='upper left')

plt.subplot(2,2,3)
plt.plot(df_vcm['fechaoperacion'],df_vcm['valor'],label='original')
plt.plot(df_vcm['fechaoperacion'],df_vcm['ma9'],label='MA9',color='yellow')
plt.legend(loc='upper left')

plt.subplot(2,2,4)
plt.plot(df_vcm['fechaoperacion'],df_vcm['valor'],label='original')
plt.plot(df_vcm['fechaoperacion'],df_vcm['ma12'],label='MA12',color='purple')
plt.legend(loc='upper left')

# mostrando gráfico
plt.show()


# - Como se puede observar la grafica todas las media moviles tienen picos pronunciados en los años 2015 y 2017 y en el año 2024
# - Conforme a la media movil tiene mayor valor de n las lineas se vuelven mas suaves.
# - Conforme a la media movil tiene mayor valor de n los picos son mas bajos
# - Valores de medias moviles con un n mas alto no seran sensibles a los efectos de datos aleatorios, por lo que la prediccion tendra una fluctuacion lenta ante periodos recientes
# 
# Se realiza un grafico de rezagos, para este caso se haran graficos de rezagos de hasta 9.

# In[11]:


# Creando función para gráficar los rezagos del dataset

# Crear un panel de gráficos 3x3
fig, axes = plt.subplots(3, 3, figsize=(12, 12))

# Iterar sobre las filas y columnas del panel
for i in range(3):
    for j in range(3):
        lag = i * 3 + j + 1  # Calcular el lag correspondiente
        ax = axes[i, j]  # Obtener el eje actual
        
        # Calcular y graficar el lag plot con el lag actual
        ax.plot(df_vcm['valor'][:-lag], df_vcm['valor'][lag:], 'o', alpha=0.5)
        ax.set_title(f'Lag = {lag}')
        ax.set_xlabel('y(t)')
        ax.set_ylabel(f'y(t + {lag})')

# Ajustar el espaciado entre los subplots
plt.tight_layout()
plt.show()


# - De las graficas de rezagos no existe un patron de distancia que pueda determinar que exista estacionalidad. Los puntos siempre se mueven de alguna manera en cada uno de los rezagos.
# - Es necesario contar con analisis de disintos intervalos de tiempo para determinar si presenta estacionalidad o no.

# In[12]:


# Establecer 'fechaoperacion' como el índice del DataFrame
df_vcm.set_index('fechaoperacion', inplace=True)

# Asegúrate de que la frecuencia esté definida (por ejemplo, mensual 'M')
df_vcm.index.freq = 'M'

# Realizar la descomposición estacional de la serie temporal
result = seasonal_decompose(df_vcm['valor'], model='additive')

# Graficar la serie original, la tendencia, la estacionalidad y el residuo
plt.figure(figsize=(12, 8))
plt.subplot(2, 1, 1)
plt.plot(df_vcm['valor'], label='Original')
plt.legend(loc='upper left')

plt.subplot(2, 1, 2)
plt.plot(result.seasonal, label='Estacionalidad')
plt.legend(loc='upper left')

plt.tight_layout()
plt.show()


# - El grafico de estacionalidad sugiere que existe un patron que se repite mensualmente, esto puede ser contrario a lo que se puede ver en el grafico de rezagos y al revisar la grafica de tendencia, en los datos no se observa que se tenga un patron o ciclo en el tiempo.
# - Es necesario contar con analisis de disintos intervalos de tiempo para determinar si presenta estacionalidad o no.
