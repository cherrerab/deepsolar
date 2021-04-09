# -*- coding: utf-8 -*-
"""
Proyecto Beauchef PV Forecasting

Cristian Herrera

v1.0 - update, Aug 01 2019
"""

import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib

import numpy as np
import pandas as pd

from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn import preprocessing

from datetime import (datetime, timedelta)

from deepsolar.database import reshape_by_day
from deepsolar._tools import validate_date, get_date_index, ext_irradiation, get_timestep

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
    secs = get_timestep(timestamps, date_format)
    
    ext_rad = [ext_irradiation(t, secs, **kargs) for t in timestamps]
    ext_sum = sum(ext_rad)
    
    # retornar cuociente
    return data_sum/ext_sum if ext_sum!=0.0 else 0.0

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

# -----------------------------------------------------------------------------
# clearsky variability
def clearsky_variability(database, timestamp, timesteps, side='backward', **kargs):
    """
    -> float
    
    Calcula la desviacion estandar del clearsky index dentro del periodo
    especificado.
    
    :param DataFrame database:
        base de datos que contenga el resgistro temporal de la radiación.
    :param str timestamp:
        string que contiene la hora y fecha del comienzo del periodo (UTC).
    :param int timesteps:
        cantidad de puntos/timesteps a considerar en el periodo.
    :param str side: ('backward' or 'forward')
        define si el timestamp corresponde al comienzo o al fin del periodo.
        
    :returns:
        (float) desviacion estandar del clearsky index en el periodo.   
    """
    
    # -------------------------------------------------------------------------
    # obtener indice del timestamp en la base de datos
    timestamp = validate_date(timestamp)
    idx = get_date_index( database['Timestamp'], timestamp )
    
    assert (side=='backward') or (side=='forward')
    
    # definir indices del periodo
    if side=='backward':
        data_index = np.arange( (idx - timesteps), idx )
    else:
        data_index = np.arange( idx, (idx + timesteps) )
        
    # obtener radiacion global en el periodo
    timestamps = database.loc[data_index, 'Timestamp'].values
    global_rad = database.loc[data_index, 'Global'].values
    
    # -------------------------------------------------------------------------
    # calcular clearsky index dentro del periodo
    date_format = '%d-%m-%Y %H:%M'
    secs = get_timestep(database['Timestamp'], date_format)
    
    # calcular radiacion extraterrestre
    ext_rad = [ext_irradiation(t, secs, **kargs) for t in timestamps]
    # calcular clearsky index
    clearsky = np.divide( global_rad, ext_rad, np.zeros_like(global_rad), where=ext_rad!=0.0)
    clearsky[np.isinf(clearsky)] = 0.0
    
    # obtener desviacion estandar
    v = np.std( np.diff(clearsky) )
    
    return v
        
#------------------------------------------------------------------------------
# realizar clustering sobre los datos díarios de radiación
def cluster_daily_radiation(database, eps=0.09, min_samples=9, plot_clusters=True,
                            **kargs):
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
    :param bool plot_clusters:
        Especifica si se desea realizar los gráficos resultantes del clustering.
        
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
    x = np.linspace(-1,1,60)
    sigma, mu = 30.0, 0.0
    gaussian = np.exp(-( (x-mu)**2 / ( 2.0 * sigma**2 ) ) )
    
    for date in dg.columns:
        global_data = dg[date]
        diffuse_data = dd[date]
        
        # indice de claridad diario
        clear_sky = day_clear_sky_index(global_data, **kargs)
        
        # indice de nubosidad (1- fracción difusa)
        cloudless = day_cloudless_index(global_data, diffuse_data)
        
        # suavidad del perfil
        smoothed_data =  np.convolve(global_data, gaussian, mode='same')
        smoothed_data = smoothed_data/np.sum( gaussian )
        
        smooth_error = np.mean( np.power(global_data - smoothed_data, 2) )
        smooth_error = np.sqrt(smooth_error)
        
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
    
    # realizar clustering
    clusterer = DBSCAN(eps=eps, min_samples=min_samples).fit(df_minmax)
    cluster_labels = clusterer.labels_
    
    # corregir labels
    num_labels = np.max(cluster_labels) + 2
    cluster_labels = np.where(cluster_labels==-1, num_labels-1, cluster_labels)
    cluster_labels[np.where(cluster_labels==3)] = 2
    cluster_labels[0] = 3
    matplotlib.rc('font', family='Times New Roman')
    # plotear resultado
    if plot_clusters:
        fig, axs = plt.subplots(2, 2)
        fig.set_size_inches(6, 4)
        
        # plotear TSNE
#        axs[0, 0].scatter(data_embedded[:,0], data_embedded[:,1], cmap='plasma',
#                          marker=markers)
#        axs[0, 0].set_title('PCA embedding')
#        axs[0, 0].style.use(['seaborn-white', 'seaborn-paper'])
        
        # plotear clustering
        axs[0, 0].scatter(df_minmax[:,0], df_minmax[:,1], cmap='plasma',
                          c=cluster_labels, s=4.0)
        axs[0, 0].set_ylabel('índice de nubosidad')
        axs[0, 0].set_xlabel('índice de claridad')
        
        axs[0, 1].scatter(df_minmax[:,0], df_minmax[:,2], cmap='plasma',
                          c=cluster_labels, s=4.0)
        axs[0, 1].set_ylabel('índice de suavidad')
        axs[0, 1].set_xlabel('índice de claridad')
        
        axs[1, 0].scatter(df_minmax[:,1], df_minmax[:,2], cmap='plasma',
                          c=cluster_labels, s=4.0)
        axs[1, 0].set_ylabel('índice de suavidad')
        axs[1, 0].set_xlabel('índice de nubosidad')
        
        plt.style.use(['seaborn-white', 'seaborn-paper'])
        plt.tight_layout()
        
        
        # crear nueva figura
        matplotlib.rc('font', family='Times New Roman')
        fig, axs = plt.subplots(3, 1)
        fig.set_size_inches(6, 5)
        # plotear conjunto de días de un mismo cluster:     
        for label in range(3):
            # obtener fechas correspondientes a la etiqueta
            cluster_dates = dg.columns[cluster_labels==label]
            
            for date in cluster_dates:
                # obtener valores de irradiancia global 
                irrad_values = dg[date].values
                # plotear
                axs[label].plot(irrad_values, c=(0.050383, 0.029803, 0.527975, 0.15))
                axs[label].set_ylim([0.0, 1300.0])
                
                # agregar elementos del gráfico
                axs[label].set_xticklabels([])
                axs[label].set_ylabel('Irradiancia global, W/m2')
                
        plt.style.use(['seaborn-white', 'seaborn-paper'])
        plt.tight_layout()                     
            
    return cluster_labels
    
#------------------------------------------------------------------------------
# obtener los n días más soleados en el dataset
def get_n_sunniest_days(database, n_days=5, **kargs):
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
    x = np.linspace(-1,1,60)
    sigma, mu = 30.0, 0.0
    gaussian = np.exp(-( (x-mu)**2 / ( 2.0 * sigma**2 ) ) )
    
    for date in dg.columns:
        global_data = dg[date]
        diffuse_data = dd[date]
        
        # indice de claridad diario
        clear_sky = day_clear_sky_index(global_data, **kargs)
        
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
    
    
    
    
    
    
    
    
    
    
    
    
        