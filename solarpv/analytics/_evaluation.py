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

from solarpv._tools import validate_date, get_timestep, get_date_index

from solarpv.analytics import cluster_daily_radiation, clearsky_variability
from solarpv.analytics import persistence_predict
from solarpv.analytics import mean_squared_error, mean_absolute_error, mean_bias_error
from solarpv.analytics import skew_error, kurtosis_error, forecast_skill
from solarpv.analytics import plot_error_dist, plot_error_variability


from solarpv.database import time_window_dataset, img_sequence_dataset

import numpy as np

#------------------------------------------------------------------------------
# evaluar modelo en cierto periodo de tiempo
def forecast_pred(datasets, output_name, model, forecast_date,
                  img_sequence=None, verbose=True):
    """
    -> (np.array, np.array)
    
    realiza una predicción de generación (forecasting) a partir del
    forecast_date especificado, utilizando el modelo entregado.
    
    :param list(DataFrame) datasets:
        lista que contiene los datasets con los atributos correspondientes a
        los inputs del model de pronóstico.
    :param str output_name:
        nombre de la columna que contiene los datos del set Y
    :param keras.model model:
        modelo de forecasting a evaluar.
    :param str forecasting_date:
        fecha y hora en la que realizar el pronóstico %d-%m-%Y %H:%M (UTC).
    :param int img_sequence:
        indice del dataset que consiste en una sequencia de imagenes.
        
    :returns:
        tupla de np.arrays de la forma (Y_true, Y_pred)
    """
    
    # obtener dataset con datos de potencias PV (Y)
    system_ds = datasets[0]
    
    # obtener index correspondiente a forecasting_date
    forecast_date = validate_date(forecast_date)
    date_index = get_date_index( system_ds['Timestamp'], forecast_date)
    
    # obtener ventana de tiempo
    n_input = int(model.input.shape[1])
    n_output = int(model.output.shape[1])
    
    start_index = date_index - n_input
    end_index = date_index + n_output
    
    # checkear ventana de tiempo
    try:
        assert start_index >= 0
        assert end_index < system_ds.shape[0]
    except AssertionError:
        if verbose:
            print('InputError: no es posible aplicar el modelo en la ventana'+
                  'de tiempo especificada')
        return None
    
    # definir datos
    Y_true = system_ds[output_name].iloc[date_index:end_index].values
    Y_true = np.reshape(Y_true, (n_output,-1))
    
    X = []
    for i, ds in enumerate(datasets):
        X_data = ds.iloc[start_index:end_index]
        if i == img_sequence:
            X_ds, _ = img_sequence_dataset(X_data, n_input, n_output, n_input+n_output)
        else:
            X_ds, _ = time_window_dataset(X_data, output_name, n_input, n_output, n_input+n_output, ['Timestamp'])
        X.append(X_ds)
    
    # aplicar modelo
    Y_pred = model.predict(X)
    Y_pred = np.reshape(Y_pred, (n_output,-1))
    
    # retornar
    assert Y_true.shape[0] == Y_pred.shape[0]
    return (Y_true, Y_pred)

#------------------------------------------------------------------------------
# obtener gráficos de predicción en cuatro días distintos al azar
def plot_forecast_pred(datasets, output_name, model, forecast_date,
                       img_sequence=-1,
                       hours=['10:00','13:00','16:00','19:00'],
                       time_margin=12, title=''):
    """
    -> None
    
    realiza el gráfico de forecasting en el día forecasting_date, utilizando el
    modelo especificado, en 4 horas a lo largo del día especificado.
    
    :param list(DataFrame) datasets:
        lista que contiene los datasets con los atributos correspondientes a
        los inputs del model de pronóstico.
    :param str output_name:
        nombre de la columna del primer dataset que contiene los datos del set Y.
    :param keras.model model:
        modelo de forecasting a evaluar.
    :param str forecasting_date:
        fecha y hora en la que realizar el pronóstico %d-%m-%Y.
    :param int img_sequence:
        indice del dataset que consiste en una sequencia de imagenes.
    :param list(str) hours:
        lista de 4 horas sobre las que aplicar el modelo.
    :param int time_margin:
        cantidad de datos a considerar como margen de la ventana temporal.
        
    ;returns:
        None
    """
    
    # obtener dataset con datos de Y
    system_ds = datasets[0]
    
    # incializar plot
    fig, axs = plt.subplots(2, 2)
    fig.suptitle(title)
    # obtener input/output del modelo
    n_input = int(model.input.shape[1])
    n_output = int(model.output.shape[1])
    
    # obtener timestamps de forecasting
    pred_times = [forecast_date + ' ' + h for h in hours]
    
    plot_axs = [(0, 0), (0, 1), (1, 0), (1, 1)]
    
    for test_time, ix in zip(pred_times, plot_axs):
        # obtener index correspondiente a forecasting_date
        date_index = np.where( system_ds['Timestamp']==test_time )[0][0]
        
        # obtener ventana de tiempo
        start_index = date_index - n_input
        end_index = date_index + n_output
        
        # checkear ventana de tiempo
        try:
            assert start_index >= 0
            assert end_index < system_ds.shape[0]
            
        except AssertionError:
            return None
        
        # definir datos de input
        T_X = system_ds.index[start_index:date_index]
        T_Y = system_ds.index[date_index:end_index]
        
        X = []
        for i, ds in enumerate(datasets):
            X_data = ds.iloc[start_index:end_index]
            if i == img_sequence:
                X_ds, _ = img_sequence_dataset(X_data, n_input, n_output, n_input+n_output)
            else:
                X_ds, _ = time_window_dataset(X_data, output_name, n_input, n_output, n_input+n_output, ['Timestamp'])
            X.append(X_ds)
        
        # aplicar modelo
        Y_pred = model.predict(X)
        Y_pred = np.reshape(Y_pred, (n_output,-1))
        
        # datos base
        start_index = max([0, start_index - time_margin])
        end_index = min([system_ds.shape[0], end_index + time_margin])
        T_data = system_ds.index[start_index:end_index]
        Y_true = system_ds[output_name].iloc[start_index:end_index]
    
        # plotear
        axs[ix[0], ix[1]].plot(T_data, Y_true, ls='--', c='k')
        
        axs[ix[0], ix[1]].plot(T_X, X[0][0,:,0], 'o-', c=[0.050383, 0.029803, 0.527975])
        axs[ix[0], ix[1]].plot(T_Y, Y_pred,'o-', c=[0.798216, 0.280197, 0.469538])
        
        axs[ix[0], ix[1]].set_ylabel('PV Yield kW_avg')
        axs[ix[0], ix[1]].set_xlabel('data points')
        
        axs[ix[0], ix[1]].set_ylim([0.0, 1.0])
        axs[ix[0], ix[1]].set_ylim([0.0, 1.0])
    
    return None

#------------------------------------------------------------------------------
# obtener plot de estimación Y_test vs Y_pred
def plot_forecast_accuracy(Y_true, Y_pred, title='', **kargs):
    """
    -> None
    
    grafica la correlación entre Y_true e Y_pred (este último correspondiente
    al forecasting realizado por el modelo) con el fin de visualizar el 
    desempeño del modelo.
    
    :param DataFrame Y_true:
        set de datos reales con los cuales comparar el forecasting.
    :param DataFrame Y_pred:
        set de datos obtenidos con el modelo.
    :param str title:
        titulo a poner en el plot.
        
    :returns:
        None
    """
    
    # reordenar datos
    Y_true = np.reshape(np.array(Y_true), (-1, 1))
    Y_pred = np.reshape(np.array(Y_pred), (-1, 1))
    
    # inicializar plot
    plt.figure()
    
    # plotear Y_true vs Y_pred
    plt.scatter(Y_true, Y_pred, c=[0.050383, 0.029803, 0.527975],**kargs)
    
    # plotear linea 1:1
    plt.plot([0, 1], [0, 1], c='k')
    
    # añadir limites
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    
    plt.title(title)
    plt.xlabel('Y_true')
    plt.ylabel('Y_pred')
    
    
#------------------------------------------------------------------------------
# evaluar forecast_model
def forecast_model_evaluation(datasets, output_name, model, data_leap,
                              cluster_labels=[], img_sequence=None,
                              plot_results=True, random_state=0,
                              save_path=None):
    """
    -> DataFrame
    
    retorna métricas de evaluación del modelo de forecast entregado en base a
    su desempeño en el dataset entregado.
    
    :param list(DataFrame) datasets:
        lista que contiene los datasets con los atributos correspondientes a
        los inputs del model de pronóstico.
    :param str output_name:
        nombre de la columna que contiene los datos del set Y.
    :param keras.model model:
        modelo de forecasting a evaluar.
    :param int data_leap:
        define el intervalo de tiempo entre cada pronóstico a realizar.
    :param list or array_like cluster_labels:
        contiene la etiqueta que identifica el cluster da cada día en el dataset.
    :param int img_sequence: (default None)
        indice del dataset que consiste en una sequencia de imagenes.
    :param bool plot_results:
        determina si se muestran los gráficos de la evaluación.
    :param int random_state:
        permite definir el random_state en la evaluación.
    :param str save_path:
        ubicación del directorio en donde guardar los resultados.
        
    :returns:
        DataFrame
    """
    
    random.seed(random_state)

    # obtener lista de días
    system_ds = datasets[0]
    
    date_format = '%d-%m-%Y %H:%M'
    timestamps = system_ds['Timestamp'].values
    
    timestep = get_timestep(system_ds['Timestamp'], date_format)
    
    initial_date = datetime.strptime(timestamps[0], date_format)
    final_date = datetime.strptime(timestamps[-1], date_format)
    
    dates = pd.date_range(initial_date, final_date, freq='1D')
    dates = dates.strftime('%d-%m-%Y')
    
    # -------------------------------------------------------------------------
    # evaluar modelo
    print('\n' + '-'*80)
    print('cluster evaluation summary')
    print('-'*80 + '\n')
    
    n_output = int(model.output.shape[1])
    
    # checkear cluster_labels
    if  not cluster_labels:
        cluster_labels = np.zeros([1, dates.size]).flatten()
        num_labels = np.max(cluster_labels) + 1
        
    num_labels = np.max(cluster_labels) + 1
    
    # inicializar cluster_metrics
    metrics = ['mbe', 'mae', 'rmse', 'fs', 'std', 'skw', 'kts']
    eval_metrics = pd.DataFrame(index=np.arange(num_labels), columns=metrics)
    
    # para cada etiqueta resultante del clustering
    for label in np.arange(num_labels):
        # obtener fechas correspondientes a la etiqueta
        cluster_dates = dates[cluster_labels==label]
        
        # escoger una fecha de muestra al azar
        cluster_sample = random.choice( list(cluster_dates) )
        
        # inicializar datos cluster
        cluster_data = []
        cluster_pred = []
        cluster_pers = []
        
        # por cada una de las fechas del cluster
        for date in cluster_dates:
            # obtener forecasting_times de testing
            timeleap = timestep*data_leap
            
            initial_hour = datetime.strptime(date, '%d-%m-%Y')
            hours = [initial_hour + timedelta(seconds=timeleap*i) for i in range(int(24*3600/timeleap))]
            pred_times = [datetime.strftime(h, date_format) for h in hours]
            
            # calcular predicción en cada una de las horas
            for pred_time in pred_times:
                try:
                    # prediccion del modelo
                    Y_true, Y_pred = forecast_pred(datasets, output_name, model, pred_time, img_sequence=img_sequence)
                    
                    # prediccion del persistence model
                    Y_pers = persistence_predict(system_ds, pred_time, n_output)
                    
                    # print progress
                    print('\revaluating cluster ' + str(int(label)) + ': ' + pred_time, end='')
                    
                except TypeError:
                    continue
                
                # agregar datos a cluster data
                cluster_data.append(Y_true)
                cluster_pred.append(Y_pred)
                cluster_pers.append(Y_pers)
        
        # calcular metricas del cluster
        Y_true = np.concatenate(cluster_data, axis=None)
        Y_pred = np.concatenate(cluster_pred, axis=None)
        Y_pers = np.concatenate(cluster_pers, axis=None)
        
        eval_metrics.at[label, 'mbe'] = mean_bias_error(Y_true, Y_pred)
        eval_metrics.at[label, 'mae'] = mean_absolute_error(Y_true, Y_pred)
        eval_metrics.at[label, 'rmse'] = np.sqrt(mean_squared_error(Y_true, Y_pred))
        eval_metrics.at[label, 'fs'] = forecast_skill(Y_true, Y_pred, Y_pers)
        eval_metrics.at[label, 'skw'] = skew_error(Y_true, Y_pred)
        eval_metrics.at[label, 'kts'] = kurtosis_error(Y_true, Y_pred)
        
        # plot cluster_sample
        plot_title = 'cluster '+ str(label)
        plot_forecast_pred(datasets, output_name, model, cluster_sample, img_sequence=img_sequence, title=plot_title)
        
        # plot gráfico estimación
        plot_forecast_accuracy(Y_true, Y_pred, title=plot_title, s=0.1)
      
    return eval_metrics

#------------------------------------------------------------------------------
# realizar analisis de error sobre modelo
def forecast_error_evaluation(datasets, output_name, model, data_leap,
                              cluster_labels=[], img_sequence=None,
                              plot_results=True, random_state=0,
                              save_path=None):
    """
    -> DataFrame
    
    retorna métricas respecto al error del modelo de forecast en base a su
    desempeño sobre el dataset entregado.
    
    :param list(DataFrame) datasets:
        lista que contiene los datasets con los atributos correspondientes a
        los inputs del model de pronóstico.
    :param str output_name:
        nombre de la columna que contiene los datos del set Y.
    :param keras.model model:
        modelo de forecasting a evaluar.
    :param int data_leap:
        define el intervalo de tiempo entre cada pronóstico a realizar.
    :param list or array_like cluster_labels:
        contiene la etiqueta que identifica el cluster da cada día en el dataset.
    :param int img_sequence: (default None)
        indice del dataset que consiste en una sequencia de imagenes.
    :param bool plot_results:
        determina si se muestran los gráficos de la evaluación.
    :param int random_state:
        permite definir el random_state en la evaluación.
    :param str save_path:
        ubicación del directorio en donde guardar los resultados.
        
    :returns:
        DataFrame
    """
    
    random.seed(random_state)

    # obtener lista de días
    system_ds = datasets[0]
    
    date_format = '%d-%m-%Y %H:%M'
    timestamps = system_ds['Timestamp'].values
    
    timestep = get_timestep(system_ds['Timestamp'], date_format)
    
    initial_date = datetime.strptime(timestamps[0], date_format)
    final_date = datetime.strptime(timestamps[-1], date_format)
    
    dates = pd.date_range(initial_date, final_date, freq='1D')
    dates = dates.strftime('%d-%m-%Y')
    
    # -------------------------------------------------------------------------
    # evaluar modelo
    print('\n' + '-'*80)
    print('cluster evaluation summary')
    print('-'*80 + '\n')
    
    n_input = int(model.input.shape[1])
    n_output = int(model.output.shape[1])
    
    # checkear cluster_labels
    if  not cluster_labels:
        cluster_labels = np.zeros([1, dates.size]).flatten()
        num_labels = np.max(cluster_labels) + 1
        
    num_labels = np.max(cluster_labels) + 1
    
    # inicializar metrics dataframe
    cols = []
    
    horizons = [str( (h + 1)*timestep/60.0 ) + ' min' for h in range(n_output)]
    metrics = ['mbe', 'mae', 'rmse', 'fs', 'std', 'skw', 'kts']
    for m in metrics:
        cols += [m + ' ' + h for h in horizons]
        
    eval_metrics = pd.DataFrame(index=np.arange(num_labels), columns=cols)
    
    # para cada etiqueta resultante del clustering
    for label in np.arange(num_labels):
        # obtener fechas correspondientes a la etiqueta
        cluster_dates = dates[cluster_labels==label]
        
        # inicializar datos cluster
        cluster_data = []
        cluster_pred = []
        cluster_pers = []
        cluster_var = []
        
        # por cada una de las fechas del cluster
        for date in cluster_dates:
            # obtener forecasting_times de testing
            timeleap = timestep*data_leap
            
            initial_hour = datetime.strptime(date, '%d-%m-%Y')
            hours = [initial_hour + timedelta(seconds=timeleap*i) for i in range(int(24*3600/timeleap))]
            pred_times = [datetime.strftime(h, date_format) for h in hours]
            
            # calcular predicción en cada una de las horas
            for pred_time in pred_times:
                try:
                    # prediccion del modelo
                    Y_true, Y_pred = forecast_pred(datasets, output_name, model,
                                                   pred_time, img_sequence=img_sequence,
                                                   verbose=False)
                    
                    # prediccion del persistence model
                    Y_pers = persistence_predict(system_ds, pred_time, n_output, ['Inversor 1'])
                    
                    # clearsky variability
                    cs_var = clearsky_variability(system_ds, pred_time, n_input)
                    
                    # print progress
                    print('\revaluating cluster ' + str(int(label)) + ': ' + pred_time, end='')
                    
                except TypeError:
                    continue
                
                # agregar datos a cluster data
                cluster_data.append(Y_true)
                cluster_pred.append(Y_pred)
                cluster_pers.append(Y_pers)
                cluster_var.append(cs_var)
        
        # calcular metricas del cluster
        Y_true = np.concatenate(cluster_data, axis=1)
        Y_pred = np.concatenate(cluster_pred, axis=1)
        Y_pers = np.concatenate(cluster_pers, axis=1)
        cs_var = np.array(cluster_var)
        
        # por cada horizonte de pronóstico
        for i, h in enumerate(horizons):
            eval_metrics.at[label, 'mbe ' + h] = mean_bias_error(Y_true[i,:], Y_pred[i,:])
            eval_metrics.at[label, 'mae ' + h] = mean_absolute_error(Y_true[i,:], Y_pred[i,:])
            eval_metrics.at[label, 'rmse ' + h] = np.sqrt(mean_squared_error(Y_true[i,:], Y_pred[i,:]))
            eval_metrics.at[label, 'fs ' + h] = forecast_skill(Y_true[i,:], Y_pred[i,:], Y_pers[i,:])
            eval_metrics.at[label, 'skw ' + h] = skew_error(Y_true[i,:], Y_pred[i,:])
            eval_metrics.at[label, 'kts ' + h] = kurtosis_error(Y_true[i,:], Y_pred[i,:])
            
        # plot distribución de error
        plot_error_dist(Y_true, Y_pred, Y_pers, horizons, bins=40, log=True, range=(-0.5,0.5))
        plot_error_variability(Y_true, Y_pred, cs_var, horizons, s=0.08)
      
    return eval_metrics
    
    
#------------------------------------------------------------------------------
# realizar evaluación respecto a clusters de días
def cluster_evaluation(solar_database, datasets, output_name, model, 
                       img_sequence=-1, plot_clusters=True, random_state=0):
    """
    -> None
    
    retorna indicadores de evaluación del modelo entregado en base a su
    desempeño en distintos tipos de días obtenidos mediante clustering de los
    datos de radiación.
    
    :param DataFrame solar_database:
        resgistro temporal de la radiación global en el día.
    :param list(DataFrame) datasets:
        lista que contiene los datasets con los atributos correspondientes a
        los inputs del model de pronóstico.
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
    print('\n' + '-'*80)
    print('cluster evaluation summary')
    print('-'*80 + '\n')
    
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
            pred_times = [date + ' ' + datetime.strftime(h,'%H:%M') for h in hours]
            
            # calcular mse de predicción en cada una de las horas
            for pred_time in pred_times:
                try:
                    Y_true, Y_pred = forecast_pred(datasets, output_name, model, pred_time, img_sequence=img_sequence)
                except TypeError:
                    continue
                
                # agregar datos a cluster data
                cluster_data.append(Y_true)
                cluster_pred.append(Y_pred)
        
        # calcular rmse y mae del cluster
        Y_true = np.concatenate(cluster_data)
        Y_pred = np.concatenate(cluster_pred)
        
        cluster_metrics.at[label, 'RMSE'] = np.sqrt(np.mean(np.power(Y_true - Y_pred, 2), axis = 0)[0])
        cluster_metrics.at[label, 'MAE'] = np.mean(np.abs(Y_true - Y_pred))
        
        # plot cluster_sample
        plot_title = 'cluster '+ str(label)
        plot_forecast_pred(datasets, output_name, model, cluster_sample, img_sequence=img_sequence, title=plot_title)
        
        # plot gráfico estimación
        plot_forecast_accuracy(Y_true, Y_pred, title=plot_title, s=0.1)
    
    print(cluster_metrics)    
    return cluster_metrics



                
                