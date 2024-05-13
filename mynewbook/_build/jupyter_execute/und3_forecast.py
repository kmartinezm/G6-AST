#!/usr/bin/env python
# coding: utf-8

# # Unidad 3: Preprocesamiento y visualización
# 
# ## 2.1 Introducción
# 
# A través de gráficos y algunas funciones para preprocesar bases de datos, describiremos el comportamiento subyacente dentro de la relación entre los datos.
# 
# ## 2.2 Objetivo
# 
# Manipular las bases de datos que involucran datos en series de tempo por medio de funciones para explicar el comportamiento regular en un intervalo de tiempo.
# 
# ## 2.3 Acción
# 
# Continuando con la dinámica de la construcción del documento con repositorio en github, se debe incluir en el documento la descomposición, la estacionariedad y la diferenciación, en caso de ser necesarias, de la variable y/o variables seleccionadas con estructura a través del tiempo. Además, si es necesario, se debe implementar alguna transformación con el fin de controlar la tendencia y la variabilidad, de la misma. Debes justificar, el por qué son o no necesarios dichos procedimientos.
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


# ### 2.2.2 Descomposición de la serie de tiempo
# 
# Adentrándonos en la estructura de las series temporales, exploramos el proceso de descomposición. A través de este análisis, separamos las series temporales en sus componentes fundamentales: tendencia, estacionalidad y residuo. Esta descomposición nos permite entender mejor la dinámica subyacente de los datos y revela patrones importantes que pueden ser cruciales para la toma de decisiones y la planificación estratégica. Para ello, llevamos a cabo los siguientes pasos: 
# 
# **1. Remuestreo de la Serie Temporal:**
# 
# En el siguiente código, realizaremos el proceso de remuestreo de una serie temporal con el fin de transformarla en valores mensuales. Comenzaremos creando una copia del dataframe original para preservar los datos originales. Luego, emplearemos la función resample para agrupar los datos en intervalos mensuales y calcular la media de cada mes. Una vez completado el remuestreo, mostraremos el nuevo dataframe transformado que ahora contiene los datos mensuales. Este procedimiento es importante para la preparación de los datos previa a la aplicación del análisis de promedio móvil, lo que nos permitirá identificar tendencias y patrones a lo largo del tiempo de manera más precisa y clara.

# In[5]:


# remuestreando la serie de tiempo a valores mensuales

# creando una copia del dataframe
df_vcm = df.copy()

#  remuestreando el dataframe a mensual
df_vcm = df_vcm.resample('M',on='fechaoperacion').mean().reset_index()
df_vcm = df_vcm[['fechaoperacion','valor']]

# mostrando el dataframe transformado
df_vcm.head()


# **2. Descomposición:**
# 
# En primer lugar, organizamos los datos temporalmente al establecer la columna 'fechaoperacion' como el índice del DataFrame. Posteriormente, nos aseguramos de que la frecuencia de los datos estuviera correctamente definida, optando por configurarla como mensual ('M').
# 
# Luego, procedimos a descomponer la serie temporal en sus componentes fundamentales utilizando el método seasonal_decompose con el modelo 'additive'. Este enfoque considera que la serie temporal es la suma de tres componentes principales: tendencia, estacionalidad y residuo.
# 
# Finalmente, generamos una visualización que mostraba la serie original junto con sus componentes descompuestos: la tendencia, la estacionalidad y el residuo.

# In[6]:


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


# **3. Resultados**
# 
# La descomposición de la serie temporal en sus componentes principales —tendencia, estacionalidad y residuo— proporciona una visión detallada de la estructura subyacente de los datos. A continuación, se presenta una interpretación exhaustiva de cada uno de estos componentes:
# 
# - Serie Original
# La serie original exhibe fluctuaciones constantes, con un aumento notable en la amplitud hacia los años más recientes. Este fenómeno sugiere una mayor volatilidad o cambios en el comportamiento subyacente de los datos a lo largo del tiempo.
# 
# - Tendencia
# Patrón General: La tendencia muestra una estabilidad relativa hasta aproximadamente 2016, seguida de una caída y un posterior aumento significativo hacia el final del período observado.
# Interpretación: Esta evolución en la tendencia podría estar influenciada por factores macroeconómicos, cambios en políticas o desarrollos específicos del mercado, dependiendo del contexto de los datos.
# 
# - Estacionalidad
# Patrón General: Se observan fluctuaciones claras y consistentes, indicativas de una estacionalidad pronunciada.
# Interpretación: La presencia de estacionalidad regular puede asociarse con ciclos comerciales anuales, patrones de consumo estacionales o influencias climáticas, según la naturaleza específica de los datos analizados.
# 
# - Residuo
# Patrón General: Los residuos exhiben un comportamiento errático, con la presencia de algunos picos significativos.
# Interpretación: Al no presentar un patrón discernible, los residuos indican la ausencia de tendencias y estacionalidades residuales. La presencia de picos podría señalar eventos atípicos o factores externos no capturados por los componentes de tendencia y estacionalidad.
# 
# ### 2.3.3 Análisis de Estacionariedad
# 
# En esta sección, nos sumergimos en el análisis de estacionariedad de las series temporales. Exploramos métodos y técnicas para evaluar la estacionariedad de los datos, lo que nos proporciona información crucial sobre la estabilidad de las propiedades estadísticas a lo largo del tiempo. Este análisis es fundamental para garantizar la fiabilidad de los modelos y las predicciones basadas en series temporales.
# 
# Los resultados del test de ADF se interpretan de la siguiente manera:
# 
# - El T-Test: Es el valor del estadístico de prueba. Cuanto más negativo sea este valor, más fuerte será la evidencia en contra de la hipótesis nula de no estacionariedad.
# - El p-value: Es la probabilidad asociada al estadístico de prueba. Un valor de p pequeño (por ejemplo, p < 0.05) indica que podemos rechazar la hipótesis nula y concluir que la serie temporal es estacionaria.
# - Valores Críticos: Estos son los valores críticos del estadístico de prueba para diferentes niveles de significancia. Comparar el valor del estadístico de prueba con estos valores críticos nos permite determinar si la serie temporal es estacionaria o no.

# In[7]:


# estacionario
adf = adfuller(df_vcm['valor'],maxlag=1)
print('El T-Test es: ',adf[0])
print('El p-value es: ',adf[1])
print('Valores criticos: ',adf[4])


# **2. Resultados**
# 
# La prueba de Dickey-Fuller aumentada (ADF) es una herramienta fundamental en el análisis de series temporales que se utiliza para evaluar la estacionariedad de los datos. Esta prueba nos permite determinar si una serie temporal exhibe un comportamiento estacionario, es decir, si sus propiedades estadísticas permanecen constantes a lo largo del tiempo.
# 
# Resultados de la prueba ADF:
# 
# - **El T-Test es**: `-4.283316754925861`
#   Este valor es el estadístico de la prueba y es más negativo que todos los valores críticos proporcionados. Esto indica una fuerte evidencia contra la hipótesis nula de que existe una raíz unitaria en la serie temporal.
# 
# - **El p-value es**: `0.0004572081128972734`
#   El valor p indica la probabilidad de obtener un resultado al menos tan extremo como el observado, bajo la hipótesis nula. Un valor p bajo (típicamente menor que 0.05) sugiere que puedes rechazar la hipótesis nula de que la serie tiene una raíz unitaria.
# 
# - **Valores críticos**:
#   `{'1%': -3.4311647822282243, '5%': -2.8578788898898638, '10%': -2.5739861161899027}`
#   Estos valores críticos corresponden a los umbrales para los niveles de significancia del 1%, 5% y 10%. Si el estadístico de prueba es más negativo #que uno de estos valores críticos, puedes rechazar la hipótesis nula con ese nivel de confianza.
# 
# **3. Conclusión**
# 
# En resumen, los resultados de la prueba ADF sugieren que puedes considerar la serie temporal como estacionaria, lo que implica que no tiene raíz unitaria y muestra un comportamiento constante en términos de media y varianza a lo largo del tiempo. Esto es crucial para muchos modelos de análisis y predicción de series temporales.
# 
# ### 2.3.4 Diferenciación
# 
# Finalmente, examinamos el análisis de diferenciación como una herramienta para abordar la no estacionariedad en las series temporales. Exploramos cómo aplicar diferenciación para transformar los datos y hacerlos estacionarios, lo que facilita un análisis más preciso y confiable de la serie temporal. Este enfoque es crucial para mitigar los efectos de la tendencia y la estacionalidad, permitiendo una interpretación más precisa de los datos y una toma de decisiones informada.
# 
# **1. Cálculo de la diferenciación**
# 
# Después de calcular la diferencia entre los valores sucesivos de la serie temporal, se agregó una nueva columna llamada 'diff' al DataFrame 'df_vcm'. Esto nos permite analizar cómo cambian los valores de la serie de un período a otro. A continuación se presenta una vista previa de los primeros 20 registros del DataFrame después de calcular y agregar las diferencias:

# In[8]:


df_vcm['diff'] = df_vcm['valor'].diff()

df_vcm.head(20)


# In[9]:


df_vcm = df_vcm.reset_index()


# Esta salida mostrará los primeros 20 registros del DataFrame 'df_vcm', donde la columna 'diff' ahora contiene las diferencias entre los valores sucesivos de la serie temporal. 

# In[10]:


# Crear lienzo
plt.figure(figsize=(10, 6))

# Gráfico de líneas para la serie original
plt.plot(df_vcm['fechaoperacion'], df_vcm['valor'], label='Original', color='blue')

# Gráfico de líneas para la serie diferenciada
plt.plot(df_vcm['fechaoperacion'], df_vcm['diff'], label='Diferenciada', color='red')

# Línea horizontal en y = 0
plt.axhline(y=0, color='black', linestyle='--', linewidth=0.5)

# Configuración de título y etiquetas
plt.title('Diferenciación de serie de tiempo')
plt.xlabel('Fecha')
plt.ylabel('Valor')
plt.legend()

# Mostrar gráfico
plt.show()


# **2. Resultados**
# 
# - Comportamiento de la Serie Original: La serie temporal en azul exhibe una tendencia inicialmente estable que se ve alterada por una variación más pronunciada hacia la segunda mitad del período analizado. Este cambio abrupto en la dinámica de la serie sugiere la presencia de factores externos o internos que influyen en su comportamiento. Esta variabilidad podría estar asociada con cambios en las condiciones del mercado, fluctuaciones económicas u otros eventos que afectan la serie a lo largo del tiempo.
# 
# - Análisis de la Serie Diferenciada:La serie diferenciada en rojo muestra **fluctuaciones alrededor de cero**, con algunos picos pronunciados. Estos picos representan cambios grandes en los valores de un periodo a otro. Las **diferencias grandes y abruptas** (picos altos tanto positivos como negativos) indican momentos de cambio significativo en la serie temporal, lo cual puede ser útil para detectar anomalías o cambios importantes en la dinámica de los datos.
# 
# La serie original exhibe una tendencia inicial seguida de una variabilidad creciente hacia la segunda mitad del período analizado. Esta variación puede indicar cambios en las condiciones del entorno o el surgimiento de patrones estacionales.
# 
# Por otro lado, la serie diferenciada muestra una estabilización alrededor de cero con fluctuaciones pronunciadas. Estas diferencias resaltan momentos de cambio significativo en los valores de la serie temporal, que pueden ser cruciales para comprender su dinámica subyacente.
# 
# **Implicaciones para el Análisis y Modelado**
# 
# 1. **Estacionariedad**: La serie diferenciada sugiere un esfuerzo exitoso por estabilizar la media, lo cual es crucial para muchos modelos estadísticos y de machine learning que asumen estacionariedad en la serie temporal.
# 
# 2. **Detección de Anomalías**: Los picos en la serie diferenciada pueden ser útiles para identificar puntos anómalos o eventos extremos que podrían requerir una investigación más profunda o el desarrollo de modelos específicos para predecir o manejar estos cambios.
# 
# 3. **Desarrollo de Modelos**: Modelar la serie original sin considerar la diferenciación podría llevar a modelos que no capturan adecuadamente la dinámica subyacente de la serie. Utilizar la serie diferenciada puede proporcionar una base más sólida para la predicción y el análisis, ya que resalta los cambios críticos en la serie temporal.
# 
# En conclusión, la comparación entre la serie original y la serie diferenciada revela la importancia de la diferenciación en el análisis de series temporales. La diferenciación proporciona una perspectiva más clara de los cambios y la dinámica de los datos, lo que facilita la toma de decisiones informadas y la planificación estratégica.
