# -*- coding: utf-8 -*-
"""
Proyecto Beauchef PV Forecasting

Cristian Herrera

v1.0 - update, Aug 01 2019
"""

import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn import preprocessing

from math import (pi, cos, sin, tan, acos)
from datetime import (datetime, timedelta)

from solarpv.database import (reshape_radiation)
from solarpv import (validate_date, get_date_index, extraterrestrial_irrad)

#------------------------------------------------------------------------------
# obtener índice de claridad diario
def day_clear_sky_index(day_data, **kargs):
    """
    -> float
    
    Calcula el índice de claridad del día (radiación global) respecto a la
    radiación extraterrestre.
    
    :param DataFrame day_data:
        resgistro temporal de la radiación en el día.
    """
    
    # obtener radiación total en el día
    data_sum = sum(day_data.values)
    
    # lista de timestamps en el registro
    date_format = '%d-%m-%Y %H:%M'
    date = day_data.name
    timestamps = [date + ' ' + h for h in day_data.index]
    
    # calcular radiación extraterrestre
    timestep = (  datetime.strptime(timestamps[1], date_format)
                - datetime.strptime(timestamps[0], date_format)  )
    secs = timestep.seconds/3600
    
    ext_rad = [extraterrestrial_irrad(t, **kargs)*secs for t in timestamps]
    ext_sum = sum(ext_rad)
    
    # retornar cuociente
    return data_sum/ext_sum

#------------------------------------------------------------------------------
# obtener fracción difusa diaria total
def day_cloudless_index(day_global, day_diffuse):
    """
    -> float
    
    Calcula la fracción difusa del día (radiación difusa respecto a la 
    radiación global).
    
    :param DataFrame day_global:
        resgistro temporal de la radiación global en el día.
        
    :param DataFrame day_diffuse:
        resgistro temporal de la radiación difusa en el día.
    """
    
    # obtener radiación total en el día
    global_sum = sum(day_global.values)
    diffuse_sum = sum(day_diffuse.values)
    
    # retornar cuociente
    return (1.0 - diffuse_sum/global_sum)


#------------------------------------------------------------------------------
# clasificar días en el dataset ('soleado', 'nublado', 'nubosidad parcial')
def classify_by_solar_weather(database):
    """
    -> list
    
    Realiza un cluster de los datos de radiación en el dataset en base a
    transformaciones matemáticas para clasificar cada día en:
        'Sunny', 'Partially Cloudy', 'Cloudy'
        
    :param DataFrame database:
        base de datos a procesar.
        
    :returns:
        lista con las etiquetas de cada día.
    """
    
    
    # reordenar dataset en días
    date_format = '%d-%m-%Y %H:%M'
    timestamps = database['Timestamp'].values
    
    initial_date = datetime.strptime(timestamps[0], date_format)
    final_date = datetime.strptime(timestamps[-1], date_format) + timedelta(days=1)
    
    initial_date = datetime.strftime(initial_date, '%d-%m-%Y')
    final_date = datetime.strftime(final_date, '%d-%m-%Y')
    
    dg = reshape_radiation(database, 'Global', initial_date, final_date)
    dd = reshape_radiation(database, 'Diffuse', initial_date, final_date)
    
    # crear dataframe de features
    df = pd.DataFrame(0.0, index = dg.columns,
                      columns = ['clear_sky', 'cloudless','TOD_sum'])
    
    for date in dg.columns:
        global_data = dg[date]
        diffuse_data = dd[date]
        
        # indice de claridad diario
        clear_sky = day_clear_sky_index(global_data)
        
        # indice de nubosidad (1- fracción difusa)
        cloudless = day_cloudless_index(global_data, diffuse_data)
        
        # suma de la tercera derivada
        TOD =  np.diff(abs(np.diff(global_data)))
        TOD_sum = sum(abs(TOD))
        
        # asignar al dataframe
        df.at[date, 'clear_sky'] = clear_sky
        df.at[date, 'cloudless'] = cloudless
        df.at[date, 'TOD_sum'] = TOD_sum
        
    
    #normalizar features
    features = df.values
    
    min_max_scaler = preprocessing.MinMaxScaler()
    features_minmax = min_max_scaler.fit_transform(features)
    
    
    kmeans = KMeans(n_clusters=3, random_state=217).fit(features_minmax)
    
    plt.figure()
    plt.scatter(df['clear_sky'].values, df['cloudless'].values, cmap = 'brg', c=kmeans.labels_, s=1.5)
    
    return kmeans.labels_
    

    
    
    
    
    
        