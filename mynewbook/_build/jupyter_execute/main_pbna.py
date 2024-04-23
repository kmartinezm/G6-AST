#!/usr/bin/env python
# coding: utf-8

# # Pronóstico de precio de bolsa de la energía
# 
# ## Introducción
# 
# El pronóstico del precio de bolsa de la energía en el mercado colombiano es una herramienta crucial para diversos actores en el sector energético y económico del país. Esta importancia radica en varios aspectos fundamentales que impactan tanto en la estabilidad del suministro eléctrico como en la viabilidad financiera de las empresas y el bienestar de los consumidores, como se detalla a continuación:
# 
# 1.	**Planificación del suministro eléctrico:** El pronóstico del precio de bolsa de la energía permite a las empresas generadoras, distribuidoras y consumidores planificar sus actividades y recursos de manera eficiente. Conociendo de antemano las tendencias esperadas en los precios de la energía, las empresas pueden tomar decisiones informadas sobre inversiones en infraestructura, contratos de suministro y estrategias de gestión de la demanda. Esto contribuye a garantizar un suministro eléctrico confiable y estable en el país.
# 2.	**Optimización de costos y mitigación de riesgos:** Para las empresas del sector energético, el precio de bolsa de la energía es un componente clave en la determinación de sus costos operativos y de producción. Un pronóstico preciso del precio de bolsa les permite optimizar sus operaciones, reducir costos y gestionar de manera efectiva los riesgos asociados a la volatilidad del mercado eléctrico. Además, les brinda la oportunidad de tomar medidas preventivas para mitigar los impactos adversos en caso de fluctuaciones inesperadas en los precios.
# 3.	**Competitividad y atracción de inversión:** Un mercado eléctrico con precios estables y predecibles fomenta un entorno favorable para la inversión tanto nacional como extranjera en el sector energético colombiano. Las empresas e inversores requieren de certidumbre en cuanto a los precios futuros de la energía para realizar evaluaciones de viabilidad económica y tomar decisiones de inversión a largo plazo. El pronóstico del precio de bolsa contribuye a generar confianza en el mercado, lo que a su vez impulsa el desarrollo de infraestructura y la creación de empleo en el sector.
# 4.	**Impacto socioeconómico:** Los precios de la energía tienen un impacto directo en la economía y el bienestar de los ciudadanos colombianos. Un aumento significativo en el precio de bolsa puede resultar en mayores costos de energía para los consumidores finales, lo que afecta su capacidad adquisitiva y el costo de vida en general. Por lo tanto, un pronóstico preciso del precio de bolsa permite a los hogares y empresas anticipar y adaptarse a posibles variaciones en sus facturas de energía, mitigando así su impacto negativo en el presupuesto familiar y la competitividad empresarial.
# 5.	**Gestión de crisis y emergencias:** En situaciones de crisis o emergencias, como la actual coyuntura de embalses en mínimos históricos en Colombia, el pronóstico del precio de bolsa de la energía adquiere una relevancia aún mayor. Conocer con anticipación los posibles incrementos en los precios de la energía permite a las autoridades y actores del sector implementar medidas de contingencia y políticas regulatorias adecuadas para garantizar la estabilidad del sistema eléctrico y minimizar el impacto socioeconómico de la crisis.
# 
# ## Recopilación de datos
# 
# La recopilación de datos relacionados con el precio de bolsa de energía se realizó mediante la extracción de información de fuentes públicas proporcionadas por el administirador del mercado XM, [Sinergox XM](https://sinergox.xm.com.co/Paginas/Home.aspx):
# 
# ![Aplicativo Sinergox XM](sinergox.PNG)
# 
# Dentro del aplicativo SINERGOX de XM, que se presenta en la sección de transacción y precio, se encuentran la base de datos del precio de bolsa de energía nacional.Este aplicativo ofrece un registro de los precios de energía en bolsa nacional para cada hora y día de operación del sistema eléctrico nacional, lo que proporciona una visión completa y actualizada de la dinámica del mercado eléctrico en tiempo real, [base de datos pública de los precios de bolsa de energía nacional](https://sinergox.xm.com.co/trpr/Paginas/Historicos/Historicos.aspx?RootFolder=%2Ftrpr%2FHistricos%2FPrecios&FolderCTID=0x012000394993FA303733428C33EC91D1DFA6DB&View=%7B5CA2173E%2D1541%2D4EC7%2D9D1C%2DE145E3DFFAE3%7D#InplviewHash5ca2173e-1541-4ec7-9d1c-e145e3dffae3=Paged%3DTRUE-p_SortBehavior%3D0-p_FileLeafRef%3DPrecio%255fBolsa%255fNacional%255f%2528%2524kwh%2529%255f1996%252exlsx-p_ID%3D294-PageFirstRow%3D91).
# 
# La disponibilidad de esta información pública es fundamental para diversos actores en el sector energético y económico de Colombia. Permite a las empresas del sector energético, así como a los reguladores y tomadores de decisiones gubernamentales, acceder a datos precisos y oportunos que les ayudan a comprender las tendencias del mercado, evaluar el rendimiento de sus operaciones y diseñar estrategias para optimizar sus actividades. Por lo anterior, se procedió a recopilar el dataset:
# 

# In[1]:


# importando librerias

# librerias para la transformación de datos
import pandas as pd
import numpy as np
from datetime import datetime

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


df.info()

