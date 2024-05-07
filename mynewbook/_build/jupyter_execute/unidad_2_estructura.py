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


# In[3]:


# información del dataset
df_data.info()


# In[4]:


# Nombre de las columnas
df_data.columns


# In[5]:


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


# In[6]:


# dataset trasnformado
df = trf_data(df_data)
df.head()


# In[7]:


dic_fec = {'fec_ini': df['fechaoperacion'].min(),
           'fec_fin': df['fechaoperacion'].max()}
dic_fec


# In[8]:


# creando copia del dataframe original
# df_vm = df.copy() 

# # seleccionando las columnas
# df_vm = df_vm[['ano','mes','valor']]

# # agrupando los precios de oferta por mes
# df_vm = df_vm.groupby(['ano','mes']).mean().reset_index()

# # creando columna fechaoperacion ano-mes
# df_vm['fechaoperacion'] = df_vm.apply(lambda x: datetime(int(x['ano']),int(x['mes']),1),axis=1)

# df_vm = df_vm[['fechaoperacion','ano','mes','valor']]
# df_vm.head()


# In[9]:


# creando grafico
plt.figure(figsize=(10,8))

# creando gráfico de linea
sns.lineplot(data=df_vm,x='fechaoperacion',y='valor')

# monstrando gráfico
plt.show()


# ### 2.2.2 Funciones para detectar patrones y ciclos de la variable

# #### Caso Base (Mensual): Promedio Movil

# In[9]:


# creando función para remuestrar movilmente el dataset, usando la función resample
def df_resample(data,target='fechaoperacion',type='M'):
    
    # creando copia del dataset original
    df = data.copy()
    
    # remuestrear el dataset
    df = df.resample(type,on=target).mean().reset_index()
    
    # ordenando el dataframe
    df = df[['fechaoperacion','valor']]
    
    return df


# In[28]:


# creando una copia del dataframe transformado
df_vcm = df.copy()

# remuestreando el dataframe en frecuencia mensual
df_vcm = df_resample(df_vcm)

# muestra
df_vcm.head()


# In[29]:


# creando grafico
plt.figure(figsize=(10,8))

# creando gráfico de linea
sns.lineplot(data=df_vcm,x='fechaoperacion',y='valor')

# monstrando gráfico
plt.show()


# In[30]:


# creando promedio movil
df_vcm


# In[33]:


# Creando promedio movil 5
df_vcm['mvl_3'] = df_vcm['valor'].rolling(window=3).mean().shift(1)
# Creando promedio movil 6
df_vcm['mvl_6'] = df_vcm['valor'].rolling(window=6).mean().shift(1)
# Creando promedio movil 10
df_vcm['mvl_10'] = df_vcm['valor'].rolling(window=10).mean().shift(1)
# Creando promedio movil 12
df_vcm['mvl_12'] = df_vcm['valor'].rolling(window=12).mean().shift(1)

df_vcm.head(20)


# In[34]:


# creando grafico
plt.figure(figsize=(10,8))

# creando gráfico de linea
# sns.lineplot(data=df_vcm,x='fechaoperacion',y='valor')
# sns.lineplot(data=df_vcm,x='fechaoperacion',y='mvl_5')
plt.plot(df_vcm['fechaoperacion'],df_vcm['valor'],label='original')
plt.plot(df_vcm['fechaoperacion'],df_vcm['mvl_3'],label='MA3')
plt.plot(df_vcm['fechaoperacion'],df_vcm['mvl_6'],label='MA6')
plt.plot(df_vcm['fechaoperacion'],df_vcm['mvl_10'],label='MA10')
plt.plot(df_vcm['fechaoperacion'],df_vcm['mvl_12'],label='MA12')

plt.xlabel('fecha')
plt.ylabel('Precio de Bolsa ($/kWh)')
plt.title('Series de tiempo - Precio de bolsa nacional')


# leyenda
plt.legend(loc='upper left')

# monstrando gráfico
plt.show()


# In[38]:


# estacionario
adf = adfuller(df_vcm['valor'],maxlag=1)
print('El T-Test es: ',adf[0])
print('El p-value es: ',adf[1])
print('Valores criticos: ',adf[4])


# In[37]:


print('El p-value es: ',adf[1])
print('Valores criticos: ',adf[4])


# In[39]:


df_vcm['diff'] = df_vcm['valor'].diff()

df_vcm.head(20)


# In[40]:


get_ipython().run_cell_magic('capture', '', "\nprint('Prueba de mensaje')")


# In[ ]:





# #### Caso Base (Mensual): Rezagos

# In[12]:


pd.plotting.lag_plot(df_vcm['valor'], lag=1)


# In[13]:


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


# #### Caso Base (Mensual): Estacionalidad

# In[15]:


# Establecer 'fechaoperacion' como el índice del DataFrame
df_vcm.set_index('fechaoperacion', inplace=True)

# Asegúrate de que la frecuencia esté definida (por ejemplo, mensual 'M')
df_vcm.index.freq = 'M'

# Realizar la descomposición estacional de la serie temporal
result = seasonal_decompose(df_vcm['valor'], model='additive')

# Graficar la serie original, la tendencia, la estacionalidad y el residuo
plt.figure(figsize=(12, 8))
plt.subplot(4, 1, 1)
plt.plot(df_vcm['valor'], label='Original')
plt.legend(loc='upper left')

plt.subplot(4, 1, 2)
plt.plot(result.trend, label='Tendencia')
plt.legend(loc='upper left')

plt.subplot(4, 1, 3)
plt.plot(result.seasonal, label='Estacionalidad')
plt.legend(loc='upper left')

plt.subplot(4, 1, 4)
plt.plot(result.resid, label='Residuo')
plt.legend(loc='upper left')

plt.tight_layout()
plt.show()


# #### Caso 1 (Trimestre): Promedio Movil

# In[17]:


# creando una copia 
df_vct = df.copy()

# remuestreando a trimestre
df_vct = df_resample(df_vct,type='Q')

# muestra
df_vct.head()


# In[18]:


# creando grafico
plt.figure(figsize=(10,8))

# creando gráfico de linea
sns.lineplot(data=df_vct,x='fechaoperacion',y='valor')

# monstrando gráfico
plt.show()


# #### Caso 1 (Trimestre): Rezagos

# In[19]:



# Crear un panel de gráficos 3x3
fig, axes = plt.subplots(3, 3, figsize=(12, 12))

# Iterar sobre las filas y columnas del panel
for i in range(3):
    for j in range(3):
        lag = i * 3 + j + 1  # Calcular el lag correspondiente
        ax = axes[i, j]  # Obtener el eje actual
        
        # Calcular y graficar el lag plot con el lag actual
        ax.plot(df_vct['valor'][:-lag], df_vct['valor'][lag:], 'o', alpha=0.5)
        ax.set_title(f'Lag = {lag}')
        ax.set_xlabel('y(t)')
        ax.set_ylabel(f'y(t + {lag})')

# Ajustar el espaciado entre los subplots
plt.tight_layout()
plt.show()


# #### Caso 1 (Trimestre): Estacionalidad

# In[25]:


# Establece 'fechaoperacion' como el índice del DataFrame
df_vct.set_index('fechaoperacion', inplace=True)

# Asegúrate de que la frecuencia esté definida (por ejemplo, mensual 'M')
df_vct.index.freq = 'Q'

# Realizar la descomposición estacional de la serie temporal
result = seasonal_decompose(df_vct['valor'], model='additive')

# Graficar la serie original, la tendencia, la estacionalidad y el residuo
plt.figure(figsize=(12, 8))
plt.subplot(4, 1, 1)
plt.plot(df_vct['valor'], label='Original')
plt.legend(loc='upper left')

plt.subplot(4, 1, 2)
plt.plot(result.trend, label='Tendencia')
plt.legend(loc='upper left')

plt.subplot(4, 1, 3)
plt.plot(result.seasonal, label='Estacionalidad')
plt.legend(loc='upper left')

plt.subplot(4, 1, 4)
plt.plot(result.resid, label='Residuo')
plt.legend(loc='upper left')

plt.tight_layout()
plt.show()


# #### Caso 2 (Semestre): Promedio Movil

# In[26]:


# creando una copia 
df_vcs = df.copy()

# remuestreando de acuerdo a lo que se quiere trimestre
df_vcs = df_resample(df_vcs,type='2Q')

# muestra
df_vcs.head()


# In[27]:


# creando grafico
plt.figure(figsize=(10,8))

# creando gráfico de linea
sns.lineplot(data=df_vcs,x='fechaoperacion',y='valor')

# monstrando gráfico
plt.show()


# #### Caso 2 (Semestre): Rezagos

# In[28]:


# Crear un panel de gráficos 3x3
fig, axes = plt.subplots(3, 3, figsize=(12, 12))

# Iterar sobre las filas y columnas del panel
for i in range(3):
    for j in range(3):
        lag = i * 3 + j + 1  # Calcular el lag correspondiente
        ax = axes[i, j]  # Obtener el eje actual
        
        # Calcular y graficar el lag plot con el lag actual
        ax.plot(df_vcs['valor'][:-lag], df_vcs['valor'][lag:], 'o', alpha=0.5)
        ax.set_title(f'Lag = {lag}')
        ax.set_xlabel('y(t)')
        ax.set_ylabel(f'y(t + {lag})')

# Ajustar el espaciado entre los subplots
plt.tight_layout()
plt.show()


# In[29]:


df_vcs.head()


# #### Caso 2 (Semestre): Estacionalidad

# In[30]:


# Establece 'fechaoperacion' como el índice del DataFrame
df_vcs.set_index('fechaoperacion', inplace=True)

# Asegúrate de que la frecuencia esté definida (por ejemplo, mensual 'M')
df_vcs.index.freq = '2Q'

# Realizar la descomposición estacional de la serie temporal
result = seasonal_decompose(df_vcs['valor'], model='additive')

# Graficar la serie original, la tendencia, la estacionalidad y el residuo
plt.figure(figsize=(12, 8))
plt.subplot(4, 1, 1)
plt.plot(df_vcs['valor'], label='Original')
plt.legend(loc='upper left')

plt.subplot(4, 1, 2)
plt.plot(result.trend, label='Tendencia')
plt.legend(loc='upper left')

plt.subplot(4, 1, 3)
plt.plot(result.seasonal, label='Estacionalidad')
plt.legend(loc='upper left')

plt.subplot(4, 1, 4)
plt.plot(result.resid, label='Residuo')
plt.legend(loc='upper left')

plt.tight_layout()
plt.show()


# #### Caso 3 (Anual): Promedio Movil

# In[31]:


# creando una copia 
df_vca = df.copy()

# remuestreando de acuerdo a lo que se quiere trimestre
df_vca = df_resample(df_vca,type='Y')

# muestra
df_vca.head()


# In[33]:


# creando grafico
plt.figure(figsize=(10,8))

# creando gráfico de linea
sns.lineplot(data=df_vca,x='fechaoperacion',y='valor')

# monstrando gráfico
plt.show()


# #### Caso 3 (Anual): Rezagos

# In[34]:


# Crear un panel de gráficos 3x3
fig, axes = plt.subplots(3, 3, figsize=(12, 12))

# Iterar sobre las filas y columnas del panel
for i in range(3):
    for j in range(3):
        lag = i * 3 + j + 1  # Calcular el lag correspondiente
        ax = axes[i, j]  # Obtener el eje actual
        
        # Calcular y graficar el lag plot con el lag actual
        ax.plot(df_vca['valor'][:-lag], df_vca['valor'][lag:], 'o', alpha=0.5)
        ax.set_title(f'Lag = {lag}')
        ax.set_xlabel('y(t)')
        ax.set_ylabel(f'y(t + {lag})')

# Ajustar el espaciado entre los subplots
plt.tight_layout()
plt.show()


# #### Caso 3 (Anual): Estacionalidad

# In[37]:


# Establece 'fechaoperacion' como el índice del DataFrame
df_vca.set_index('fechaoperacion', inplace=True)

# Asegúrate de que la frecuencia esté definida (por ejemplo, mensual 'M')
df_vca.index.freq = 'Y'

# Realizar la descomposición estacional de la serie temporal
result = seasonal_decompose(df_vca['valor'], model='additive')

# Graficar la serie original, la tendencia, la estacionalidad y el residuo
plt.figure(figsize=(12, 8))
plt.subplot(4, 1, 1)
plt.plot(df_vca['valor'], label='Original')
plt.legend(loc='upper left')

plt.subplot(4, 1, 2)
plt.plot(result.trend, label='Tendencia')
plt.legend(loc='upper left')

plt.subplot(4, 1, 3)
plt.plot(result.seasonal, label='Estacionalidad')
plt.legend(loc='upper left')

plt.subplot(4, 1, 4)
plt.plot(result.resid, label='Residuo')
plt.legend(loc='upper left')

plt.tight_layout()
plt.show()

