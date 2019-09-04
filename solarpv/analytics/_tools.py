# -*- coding: utf-8 -*-
"""
Proyecto Beauchef PV Forecasting

Cristian Herrera

v1.0 - update, Aug 01 2019
"""

import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE
from sklearn import preprocessing

from datetime import (datetime, timedelta)

from solarpv.database import reshape_by_day
from solarpv import ext_irradiation

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
    secs = timestep.seconds/3600.0
    
    ext_rad = [ext_irradiation(t, secs, **kargs) for t in timestamps]
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
    cloudness = (1.0 - diffuse_sum/global_sum)
    
    if np.isnan(cloudness):
        return 0.0
    
    return (1.0 - diffuse_sum/global_sum)


#------------------------------------------------------------------------------
# realizar clustering sobre los datos díarios de radiación
def cluster_daily_radiation(database, eps=0.07, min_samples=9):
    """
    -> list
    
    Realiza un cluster de los datos de radiación en el dataset en base a
    transformaciones matemáticas para clasificar cada día en:
        'Sunny', 'Partially Cloudy', 'Cloudy'
        
    :param DataFrame database:
        base de datos a procesar.
    :param float eps:
        La distancia máxima entre dos puntos para que se consideren dentro de
        la vecindad del otro.
    :param int min_samples:
        La cantidad mínima de puntos dentro de la vecindad de un punto para que
        este último se considere un core-point.
        
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
    
    dg = reshape_by_day(database, 'Global', initial_date, final_date)
    dd = reshape_by_day(database, 'Diffuse', initial_date, final_date)
    
    # crear dataframe de features
    df = pd.DataFrame(0.0, index = dg.columns,
                      columns = ['clear_sky', 'cloudless','smoothness'])
    
    # preparar filtro gaussiano
    x = np.linspace(-1,1,15)
    sigma, mu = 1.0, 0.0
    gaussian = np.exp(-( (x-mu)**2 / ( 2.0 * sigma**2 ) ) )
    
    for date in dg.columns:
        global_data = dg[date]
        diffuse_data = dd[date]
        
        # indice de claridad diario
        clear_sky = day_clear_sky_index(global_data)
        
        # indice de nubosidad (1- fracción difusa)
        cloudless = day_cloudless_index(global_data, diffuse_data)
        
        # suavidad del perfil
        smoothed_data =  np.convolve(global_data, gaussian, mode='same')
        smoothed_data = smoothed_data/np.sum( gaussian )
        
        smooth_error = np.sum( np.abs(global_data - smoothed_data) )
        
        # asignar al dataframe
        df.at[date, 'clear_sky'] = clear_sky
        df.at[date, 'cloudless'] = cloudless
        df.at[date, 'smoothness'] = smooth_error
        
    # -------------------------------------------------------------------------
    # clustering
    
    #normalizar features
    features = df.values
    
    min_max_scaler = preprocessing.MinMaxScaler()
    df_minmax = min_max_scaler.fit_transform(features)

    clusterer = DBSCAN(eps=eps, min_samples=min_samples).fit(df_minmax)
    
    data_embedded = TSNE(n_components=2).fit_transform(df_minmax)
    plt.figure()
    plt.scatter(data_embedded[:,0], data_embedded[:,1], cmap='plasma',
                c=clusterer.labels_, s=10.0)
    
    return clusterer.labels_
    
#------------------------------------------------------------------------------
# obtener los n días más soleados en el dataset
def get_n_sunniest_days(database, n_days=5):
    """
    -> list
    
    Obtiene los n días más soleados en el dataset. Retorna una lista con las
    fechas de los n días en formato %d-%m-%Y.
        
    :param DataFrame database:
        base de datos a procesar.
    :param int n_days:
        La cantidad de días a retornar del dataset.
        
    :returns:
        lista con la fecha de los n días más soleados.
    """
    
    # reordenar dataset en días
    date_format = '%d-%m-%Y %H:%M'
    timestamps = database['Timestamp'].values
    
    initial_date = datetime.strptime(timestamps[0], date_format)
    final_date = datetime.strptime(timestamps[-1], date_format) + timedelta(days=1)
    
    initial_date = datetime.strftime(initial_date, '%d-%m-%Y')
    final_date = datetime.strftime(final_date, '%d-%m-%Y')
    
    dg = reshape_by_day(database, 'Global', initial_date, final_date)
    dd = reshape_by_day(database, 'Diffuse', initial_date, final_date)
    
    # crear dataframe de features
    df = pd.DataFrame(0.0, index = dg.columns,
                      columns = ['clear_sky', 'cloudless','smoothness'])
    
    # preparar filtro gaussiano
    x = np.linspace(-1,1,15)
    sigma, mu = 1.0, 0.0
    gaussian = np.exp(-( (x-mu)**2 / ( 2.0 * sigma**2 ) ) )
    
    for date in dg.columns:
        global_data = dg[date]
        diffuse_data = dd[date]
        
        # indice de claridad diario
        clear_sky = day_clear_sky_index(global_data)
        
        # indice de nubosidad (1- fracción difusa)
        cloudless = day_cloudless_index(global_data, diffuse_data)
        
        # suavidad del perfil
        smoothed_data =  np.convolve(global_data, gaussian, mode='same')
        smoothed_data = smoothed_data/np.sum( gaussian )
        
        smooth_error = np.sum( np.abs(global_data - smoothed_data) )
        
        # asignar al dataframe
        df.at[date, 'clear_sky'] = clear_sky
        df.at[date, 'cloudless'] = cloudless
        df.at[date, 'smoothness'] = 1.0 - smooth_error
        
    # normalizar features
    features = df.values
    
    min_max_scaler = preprocessing.MinMaxScaler()
    df_minmax = min_max_scaler.fit_transform(features)
    
    # obtener norma de cada día
    norms = list(np.linalg.norm(df_minmax, axis=1))
    
    data_points = list( zip( norms, df_minmax[:,2], list(dg.columns) ) )
    data_points.sort(reverse=True)
    
    # obtener los n puntos con mayor norma
    n_sunniest = [data_points[i][1:3] for i in range(n_days)]
    
    # ordenar por suavidad
    n_sunniest.sort(reverse=True)
    n_sunniest = [n_sunniest[i][1] for i in range(n_days)]

    return n_sunniest
    
    
    
    
    
    
    
    
    
    
    
    
        