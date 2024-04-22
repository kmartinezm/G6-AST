#!/usr/bin/env python
# coding: utf-8

# ## Introdución
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
# La información proviene del aplicativo del administrador del mercado XM, una fuente de acceso público que ofrece datos detallados sobre los precios de bolsa de energía en el mercado colombiano. Este aplicativo ofrece un registro de los precios de energía para cada hora y día de operación del sistema eléctrico nacional, lo que proporciona una visión completa y actualizada de la dinámica del mercado eléctrico en tiempo real.
# 
# La disponibilidad de esta información pública es fundamental para diversos actores en el sector energético y económico de Colombia. Permite a las empresas del sector energético, así como a los reguladores y tomadores de decisiones gubernamentales, acceder a datos precisos y oportunos que les ayudan a comprender las tendencias del mercado, evaluar el rendimiento de sus operaciones y diseñar estrategias para optimizar sus actividades.
# 

# In[1]:


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from datetime import datetime


# In[2]:


# recopilación de datos
df_data = pd.read_csv('./dataset.csv',sep=',')
df_data.head()


# In[3]:


df_data.info()


# In[4]:


# hola
df_data.columns


# In[ ]:




