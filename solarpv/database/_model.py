# -*- coding: utf-8 -*-
"""
Proyecto Beauchef PV Forecasting

Cristian Herrera

v1.0 - update, Aug 01 2019
"""

import pandas as pd
pd.options.mode.chained_assignment = None

from datetime import datetime, timedelta

import numpy as np

from math import floor
from progressbar import ProgressBar

from time import sleep


#------------------------------------------------------------------------------
# generar database de ventanas temporales
def time_window_dataset(dataset, output_name, n_input, n_output, overlap,
                        avoid_columns=[]):
    """
    -> (np.array, np.array)
    
    Crea dos datasets (X,Y) a partir del database, tomando ventanas de tiempo
    adyacentes, el desfase entre estas ventanas puede ser definido mediante el
    parámetro overlap.
    
    :param DataFrame dataset:
        base de datos que contiene el registro temporal a procesar.
    :param str output_name:
        nombre de la columna que contiene los datos del set Y
    :param int n_input:
        largo de la ventana temporal de X.
    :param int n_output:
        largo de la ventana temporal de Y.
    :param int overlap:
        desfase entre las ventanas temporales.
    :param list(str) avoid_columns:
        lista con el nombre de las columnas que no se desean procesar.
    
    :returns:
        tupla de np.arrays (X, Y). 
    """
    # obtener cantidad de datos en database
    n_data = dataset.shape[0]
    
    # remover columnas especificadas en avoid_columns
    colnames = list(dataset.columns.values)
    for c in avoid_columns:
        colnames.remove(c)
        
    # obtener cantidad de atributos en database
    n_features = len(colnames)
    
    # -------------------------------------------------------------------------
    # obtener cantidad de ventanas temporales que se construiran
    k = overlap
    n_windows = floor( (n_data-n_input-n_output)/k ) + 1
    
    # inicializar datasets
    X = np.zeros((n_windows, n_input, n_features))
    Y = np.zeros((n_windows, n_output))
    
    # -------------------------------------------------------------------------
    # procesar datasets
    
    # por cada atributo
    for i in range(n_features):
        
        # obtener serie de datos de la columna
        column_i = colnames[i]
        feature_data = dataset[column_i]
        
        # obtener ventanas temporales
        for j in np.arange( n_windows ):
            X[j, :, i] = feature_data[k*j:k*j+n_input].values
            
            # si el atributo corresponde al output
            if column_i == output_name:
                Y[j, :] = feature_data[k*j+n_input:k*j+n_input+n_output].values
    
    # retornar sets    
    return (X, Y)

# -----------------------------------------------------------------------------
# contruir set de input y output para entrenamiento, (X, Y)
def setup_lstm_dataset(dataset, output_name, days_train, days_test,
                       n_input, n_output, overlap, avoid_columns=[]):
    """
    -> numpy.array, numpy.array, numpy.array, numpy.array
    
    construye el par de datasets X, Y necesarios para el entrenamiento de
    un modelo LSTM, asociando a cada 3-dimension una columna del dataset
    entregado. el dataset Y se contruye a partir de la columna output_name.
    
    :param pd.DataFrame dataset:
        set de datos a partir del cual se obtienen los sets X e Y.
    :param str output_name:
        nombre de la columna que contiene los datos de Y.
    :param int days_train:
        periodo de días a agregar al set X_train en cada iteración.
    :param int days_test:
        periodo de días a agregar al set X_test en cada iteración.
    :param int n_input:
        largo de la ventana temporal de X.
    :param int n_output:
        largo de la ventana temporal de Y.
    :param int overlap:
        desfase entre las ventanas temporales.
    :param list(str) avoid_columns:
        lista con el nombre de las columnas que no se desean procesar.
        
    :returns:
        tupla de datasets (X_train, X_test, Y_train, Y_test)
    """
    
    print('\n' + '='*80)
    print('Spliting Training Data')
    print('='*80 + '\n')
    sleep(0.25)
    # -------------------------------------------------------------------------
    # inicialización
    
    # obtener fecha inicial del dataset e inicializar current_date
    date_format = '%d-%m-%Y %H:%M'
    current_date = datetime.strptime(dataset.at[0, 'Timestamp'], date_format)
    
    # inicializar sets de datos
    train_dataset = []
    
    # datos de entrenamiento
    X_train_wd = []; Y_train_wd = []
    
    # datos de validación
    X_test_wd = []; Y_test_wd = []
    
    # -------------------------------------------------------------------------
    # procesamiento
    train_date = current_date + timedelta(days=days_train)
    test_date = current_date + timedelta(days=days_train+days_test)
    
    # inicializar sub-sets
    train_i = []; test_i = []
    
    bar = ProgressBar()
    for i in bar( dataset.index ):
        # obtener timestamp
        timestamp = datetime.strptime(dataset.at[i, 'Timestamp'], date_format)
        #print(timestamp)
        
        # si debe ser agregado al X_train
        if (timestamp < train_date):
            train_i.append( dataset.iloc[[i]] )
          
        # si debe ser agregado al X_test
        elif (train_date <= timestamp) and (timestamp < test_date):
            test_i.append( dataset.iloc[[i]] )
            
        # si se entra en un nuevo periodo de split
        elif (test_date <= timestamp):
            # concatenar datos
            train_data = pd.concat(train_i, axis=0)
            test_data = pd.concat(test_i, axis=0)
            
            # agregar train_data a list
            train_dataset.append(train_data)
            
            # obtener ventanas temporales
            X_train_wd_i, Y_train_wd_i = time_window_dataset(train_data, output_name, n_input, n_output, overlap, avoid_columns)
            X_test_wd_i, Y_test_wd_i = time_window_dataset(test_data, output_name, n_input, n_output, overlap, avoid_columns)
        
            # agregar ventanas a listas
            X_train_wd.append(X_train_wd_i)
            Y_train_wd.append(Y_train_wd_i)
            
            X_test_wd.append(X_test_wd_i)
            Y_test_wd.append(Y_test_wd_i)
            
            # reinicializar sub-sets
            train_i = []; test_i = []
            
            # re-establecer fechas
            current_date = timestamp
            train_date = current_date + timedelta(days=days_train)
            test_date = current_date + timedelta(days=days_train+days_test)
            
    # -------------------------------------------------------------------------
    # finalizacion
    
    # concatenar ventanas temporales
    X_train = np.concatenate(X_train_wd, axis=0)
    Y_train = np.concatenate(Y_train_wd, axis=0)
    
    X_test = np.concatenate(X_test_wd, axis=0)
    Y_test = np.concatenate(Y_test_wd, axis=0)
    
    # retornar
    return X_train, X_test, Y_train, Y_test

# -----------------------------------------------------------------------------
# normalizar sets de datos de entrenamiento y validación (X_train, X_test)
def lstm_standard_scaling(X_train, X_test):
    """
    -> numpy.array, numpy.array
    
    normaliza los datos de X_train y X_test en base al máximo y mínimo en el
    set X_train.
    
    :param numpy.array X_train:
        set de datos de entrenamiento. se supone una estructura de ventanas
        temporales donde cada feature se presenta en la tercera dimensión.
        (n_time_windows, n_inputs, n_features).
    :param numpy.array X_test:
        set de datos de validación. se supone una estructura de ventanas
        temporales donde cada feature se presenta en la tercera dimensión.
        (n_time_windows, n_inputs, n_features).
        
    :returns:
        los sets de datos X_train, X_test normalizados entre 0 y 1.
    """
    
    # inicializar set de parámetros para normalizar
    std_scaler = np.zeros((X_train.shape[2], 2))
    
    # por cada feature del dataset de training
    for i in range(X_train.shape[2]):
        # obtener el mínimo
        feature_min = np.min(X_train[:,:,i], axis=None)
        # obtener el máximo
        feature_max = np.max(X_train[:,:,i], axis=None)
        
        # normalizar atributo
        X_train[:,:,i] = (X_train[:,:,i] - feature_min)/(feature_max - feature_min)
        
        # agregar parámetros al scaler
        std_scaler[i, 0] = feature_min
        std_scaler[i, 1] = feature_max
    
    # por cada feature del dataset de testing
    for i in range(X_train.shape[2]):
        # obtener parámetros del scaler
        feature_min = std_scaler[i, 0]
        feature_max = std_scaler[i, 1]
        
        # normalizar atributo
        X_test[:,:,i] = (X_test[:,:,i] - feature_min)/(feature_max - feature_min)
            
    # retornar
    return (X_train, X_test, std_scaler)