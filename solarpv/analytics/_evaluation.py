# -*- coding: utf-8 -*-
"""
Proyecto Beauchef PV Forecasting

Cristian Herrera

v1.0 - update, Aug 01 2019
"""


import matplotlib.pyplot as plt
import random

from datetime import datetime, timedelta

import pandas as pd

from solarpv import validate_date
from solarpv.analytics import cluster_daily_radiation
from solarpv.database import time_window_dataset

import numpy as np

#------------------------------------------------------------------------------
# evaluar modelo en cierto periodo de tiempo
def forecasting_test(database, output_name, model, forecasting_date):
    """
    -> (np.array, np.array)
    
    realiza una predicción de generación (forecasting) a partir del
    forecasting_date especificado, utilizando el modelo entregado.
    
    :param DataFrame database:
        base de datos que contiene el registro de generación fotovoltaica en
        el tiempo.
    :param str output_name:
        nombre de la columna que contiene los datos del set Y
    :param keras.model model:
        modelo de forecasting a evaluar.
    :param str forecasting_date:
        fecha y hora en la que realizar el pronóstico %d-%m-%Y %H:%M.
        
    :returns:
        tupla de np.arrays de la forma (Y_data, Y_pred)
    """
    
    # obtener index correspondiente a forecasting_date
    forecasting_date = validate_date(forecasting_date)
    date_index = np.where( database['Timestamp']==forecasting_date)[0][0]
    
    # obtener ventana de tiempo
    n_input = int(model.input.shape[1])
    n_output = int(model.output.shape[1])
    
    start_index = date_index - n_input
    end_index = date_index + n_output
    
    # checkear ventana de tiempo
    try:
        assert start_index >= 0
        assert end_index < database.shape[0]
    except AssertionError:
        print('InputError: no es posible aplicar el modelo en la ventana'+
              'de tiempo especificada')
        return None
    
    # definir datos
    Y_data = database[output_name].iloc[date_index:end_index].values
    Y_data = np.reshape(Y_data, (n_output,-1))
    
    X_data = database.iloc[start_index:end_index]
    X, _ = time_window_dataset(X_data, output_name, n_input, n_output, n_input+n_output, ['Timestamp'])
    
    # aplicar modelo
    Y_pred = model.predict(X)
    Y_pred = np.reshape(Y_pred, (n_output,-1))
    
    # retornar
    assert Y_data.shape[0] == Y_pred.shape[0]
    return (Y_data, Y_pred)

#------------------------------------------------------------------------------
# obtener gráficos de predicción en cuatro días distintos al azar
def plot_forecasting_model(database, output_name, model, forecasting_date, 
                           hours=['10:00','13:00','16:00','19:00'],
                           time_margin=12, title=''):
    """
    -> None
    
    realiza el gráfico de forecasting en el día forecasting_date, utilizando el
    modelo especificado, en 4 horas a lo largo del día especificado.
    
    :param DataFrame database:
        base de datos que contiene el registro de generación fotovoltaica en
        el tiempo.
    :param str output_name:
        nombre de la columna que contiene los datos del set Y.
    :param keras.model model:
        modelo de forecasting a evaluar.
    :param str forecasting_date:
        fecha y hora en la que realizar el pronóstico %d-%m-%Y.
    :param list(str) hours:
        lista de 4 horas sobre las que aplicar el modelo.
    :param int time_margin:
        cantidad de datos a considerar como margen de la ventana temporal.
        
    ;returns:
        None
    """
    
    # incializar plot
    fig, axs = plt.subplots(2, 2)
    fig.suptitle(title)
    # obtener input/output del modelo
    n_input = int(model.input.shape[1])
    n_output = int(model.output.shape[1])
    
    # obtener timestamps de forecasting
    test_times = [forecasting_date + ' ' + h for h in hours]
    
    plot_axs = [(0, 0), (0, 1), (1, 0), (1, 1)]
    
    for test_time, ix in zip(test_times, plot_axs):
        # obtener index correspondiente a forecasting_date
        date_index = np.where( database['Timestamp']==test_time)[0][0]
        
        # obtener ventana de tiempo
        start_index = date_index - n_input
        end_index = date_index + n_output
        
        # checkear ventana de tiempo
        try:
            assert start_index >= 0
            assert end_index < database.shape[0]
            
        except AssertionError:
            return None
        
        # definir datos
        T_X = database.index[start_index:date_index]
        T_Y = database.index[date_index:end_index]
        
        X_data = database.iloc[start_index:end_index]
        X, _ = time_window_dataset(X_data, output_name, n_input, n_output, n_input+n_output, ['Timestamp'])
        
        # aplicar modelo
        Y_pred = model.predict(X)
        Y_pred = np.reshape(Y_pred, (n_output,-1))
        
        # datos base
        start_index = max([0, start_index - time_margin])
        end_index = min([database.shape[0], end_index + time_margin])
        T_data = database.index[start_index:end_index]
        Y_data = database[output_name].iloc[start_index:end_index]
    
        # plotear
        axs[ix[0], ix[1]].plot(T_data, Y_data, ls='--', c='k')
        
        axs[ix[0], ix[1]].plot(T_X, X[0,:,0], 'o-', c=[0.050383, 0.029803, 0.527975])
        axs[ix[0], ix[1]].plot(T_Y, Y_pred,'o-', c=[0.798216, 0.280197, 0.469538])
        
        axs[ix[0], ix[1]].set_ylabel('PV Yield kW_avg')
        axs[ix[0], ix[1]].set_xlabel('data points')
        
        axs[ix[0], ix[1]].set_ylim([0.0, 1.0])
        axs[ix[0], ix[1]].set_ylim([0.0, 1.0])
    
    return None

#------------------------------------------------------------------------------
# obtener plot de estimación Y_test vs Y_pred
def plot_forecasting_accuracy(Y_data, Y_pred, title='', **kargs):
    """
    -> None
    
    grafica la correlación entre Y_data e Y_pred (este último correspondiente
    al forecasting realizado por el modelo) con el fin de visualizar el 
    desempeño del modelo.
    
    :param DataFrame Y_data:
        set de datos reales con los cuales comparar el forecasting.
    :param DataFrame Y_pred:
        set de datos obtenidos con el modelo.
    :param str title:
        titulo a poner en el plot.
        
    :returns:
        None
    """
    
    # reordenar datos
    Y_data = np.reshape(np.array(Y_data), (-1, 1))
    Y_pred = np.reshape(np.array(Y_pred), (-1, 1))
    
    # inicializar plot
    plt.figure()
    
    # plotear Y_data vs Y_pred
    plt.scatter(Y_data, Y_pred, c=[0.050383, 0.029803, 0.527975],**kargs)
    
    # plotear linea 1:1
    plt.plot([0, 1], [0, 1], c='k')
    
    # añadir limites
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    
    plt.title(title)
    plt.xlabel('Y_data')
    plt.ylabel('Y_pred')

#------------------------------------------------------------------------------
# realizar evaluación respecto a clusters de días
def cluster_evaluation(solar_database, pv_database, output_name, model, 
                       plot_clusters=True, random_state=0):
    """
    -> None
    
    retorna indicadores de evaluación del modelo entregado en base a su
    desempeño en distintos tipos de días obtenidos mediante clustering de los
    datos de radiación.
    
    :param DataFrame solar_database:
        resgistro temporal de la radiación global en el día.
    :param DataFrame pv_database:
        registro temporal de generación en el día.
    :param str output_name:
        nombre de la columna que contiene los datos del set Y.
    :param keras.model model:
        modelo de forecasting a evaluar.
    :param bool plot_clusters:
        Especifica si se desea realizar los gráficos resultantes del clustering.
        
    :returns:
        DataFrame
    """
    
    # realizar clustering sobre los datos
    cluster_labels = np.array( cluster_daily_radiation(solar_database, plot_clusters=plot_clusters) )
    
    # obtener lista de días
    date_format = '%d-%m-%Y %H:%M'
    timestamps = solar_database['Timestamp'].values
    
    initial_date = datetime.strptime(timestamps[0], date_format)
    final_date = datetime.strptime(timestamps[-1], date_format)
    
    dates = pd.date_range(initial_date, final_date, freq='1D')
    dates = dates.strftime('%d-%m-%Y')
    
    # -------------------------------------------------------------------------
    # evaluar modelo en los clusters
    print('\n' + '='*80)
    print('Cluster Evaluation Summary')
    print('='*80 + '\n')
    
    random.seed(random_state)
    
    num_labels = np.max(cluster_labels) + 1
    
    # inicializar cluster_metrics
    cluster_metrics = pd.DataFrame(index=np.arange(num_labels),
                                   columns=['MAE', 'RMSE'])
    

    # para cada etiqueta resultante del clustering
    for label in np.arange(num_labels):
        
        # obtener fechas correspondientes a la etiqueta
        cluster_dates = dates[cluster_labels==label]
        # escoger una fecha de muestra al azar
        cluster_sample = random.choice( list(cluster_dates) )
        
        # inicializar datos cluster
        cluster_data = []
        cluster_pred = []
        
        # por cada una de las fechas del cluster
        for date in cluster_dates:
            # obtener forecasting_times de testing
            initial_hour = datetime.strptime('00:00','%H:%M')
            hours = [initial_hour + timedelta(seconds=3600*i) for i in range(24)]
            test_times = [date + ' ' + datetime.strftime(h,'%H:%M') for h in hours]
            
            # calcular mse de predicción en cada una de las horas
            for test_time in test_times:
                try:
                    Y_data, Y_pred = forecasting_test(pv_database, output_name, model, test_time)
                except TypeError:
                    continue
                
                # agregar datos a cluster data
                cluster_data.append(Y_data)
                cluster_pred.append(Y_pred)
        
        # calcular rmse y mae del cluster
        Y_data = np.concatenate(cluster_data)
        Y_pred = np.concatenate(cluster_pred)
        
        cluster_metrics.at[label, 'RMSE'] = np.sqrt(np.mean(np.power(Y_data - Y_pred, 2), axis = 0)[0])
        cluster_metrics.at[label, 'MAE'] = np.mean(np.abs(Y_data - Y_pred))
        
        # plot cluster_sample
        plot_title = 'cluster '+ str(label)
        plot_forecasting_model(pv_database, output_name, model, cluster_sample, title=plot_title)
        
        # plot gráfico estimación
        plot_forecasting_accuracy(Y_data, Y_pred, title=plot_title, s=0.1)
    
    print(cluster_metrics)    
    return cluster_metrics
                
                
            
            
            
            
            
            
            
        
        
        
        
        
    
    


    
    
    
    
    
    
    
    
    
