# -*- coding: utf-8 -*-
"""
Proyecto Beauchef PV Forecasting

Cristian Herrera

v1.0 - update, Aug 01 2019
"""


import os
import inspect
from time import sleep

from datetime import datetime, timedelta

import pandas as pd

from solarpv._tools import validate_date, get_date_index, ext_irradiation, get_timestep, solar_incidence
from solarpv.database import read_csv_file
from solarpv.analytics import mean_squared_error

import numpy as np
from scipy.optimize import minimize
from math import (pi, cos, sin, tan, acos)


#------------------------------------------------------------------------------
# AR model predict
def ar_predict(X, ar_param):
    """
    -> float
    
    realiza una predicción mediante un modelo AR(ar_param) sobre la serie
    de tiempo entregada (X).
    
    :param np.array X:
        serie de tiempo sobre la cual realizar la predicción.
    :param np.array ar_param:
        parámetros del modelo AR.
    
    :returns:
        valor resultante de la autoregresión.
    """
    
    # checkear serie de tiempo y parámetros
    assert X.shape == ar_param.shape
    
    # calcular
    prediction = np.sum(np.multiply(X, ar_param))
    
    return prediction

#------------------------------------------------------------------------------
# AR model fit
def ar_fit(X_train, Y_train, val_set=None, solver='Nelder-Mead'):
    """
    -> numpy.array(floats)
    
    estima los parámetros del modelo de autoregresion que minimicen el mse
    entre las predicciones y los valores de entrenamiento Y_train.
    
    :param numpy.array X_train:
        serie de datos input de entrenamiento. (num_samples, num_input)
    :param numpy.array Y_train:
        serie de datos output de entrenamiento. (num_samples, num_output)
    :param tuple(numpy.array) val_set:
        serie de datos (X_val, Y_val) para validación.
    :param str solver:
        método/sover a utlizar en la minimización (scipy.optimize.minimize)
        default='Nelder-Mead'
        
    :returns: los parámetros del modelo AR.
        
    """
    print('\n' + '-'*80)
    print('fitting autoregressive model parameters')
    print('-'*80 + '\n')
    sleep(0.25)
    
    # checkear que el solver especificado sea admisible
    assert solver in ['Nelder-Mead','Powell','CG','BFGS','L-BFGS-B','TNC','SLSQP']
    
    # obtener características de los datos
    n_time_series, n_input = X_train.shape
    n_output = Y_train.shape[1]
    
    # inicializar parámetros (normal)
    parameters = np.random.uniform(low=-1, high=1, size=n_input)
    
    # optimización ------------------------------------------------------------
    num_steps = 0
    
    # inicializar registro de scores
    score_hist = []
    
    # definir función score
    def optimization_step(ar_param):
        nonlocal num_steps
        nonlocal score_hist
        
        # inicializar matriz de predicción
        Y_pred = np.zeros_like(Y_train)
        
        # por cada serie de tiempo
        for i in range(n_time_series):
            
            # inicializar serie de tiempo auxiliar
            x = np.zeros([1, n_input+n_output]).flatten()
            x[0:n_input] = X_train[i, :]
            
            for j in range(n_input, n_input+n_output):
                # predecir usando ar
                x[j] = ar_predict(x[j-n_input:j], ar_param)
            
            # agregar predicciones
            Y_pred[i,:] = x[n_input:n_input+n_output]
            
        # calcular errror
        score = mean_squared_error(Y_train, Y_pred)
        score_hist.append(score)
        
        # print progress
        if num_steps%10 == 0:
            print('\rstep: %5d, score: %3.3f' %(num_steps, score), end='')
        
        num_steps += 1
        return score
    
    # minimizar
    res = minimize(fun=optimization_step, x0=parameters, method=solver,
                   options={'maxiter': 10000, 'disp': False})
    params = np.reshape(res.x, (1, n_input))
    
    # validación
    if val_set:
        print('\n\nevaluating on validation set: ', end='')
        
        X_val, Y_val = val_set
        assert X_val.shape[1] == params.shape[1]
        
        num_time_series = X_val.shape[0]
        Y_pred = np.zeros_like(Y_val)
        
        # por cada serie de tiempo
        for i in range(num_time_series):
            # inicializar serie de tiempo auxiliar
            x = np.zeros([1, n_input+n_output]).flatten()
            x[0:n_input] = X_val[i, :]
            
            for j in range(n_input, n_input+n_output):
                # predecir usando ar
                x[j] = ar_predict(x[j-n_input:j], params.flatten())
            
            # agregar predicciones
            Y_pred[i,:] = x[n_input:n_input+n_output]
        
        # calcular error
        val_score = mean_squared_error(Y_val, Y_pred)
        print('done')
        
        print('\nAR fit completed:')
        print('fit_score: %3.3f, val_score: %3.3f' %(score_hist[-1], val_score))
    
    else:
        print('\nAR fit completed:')
        print('fit_score: %3.3f' %(score_hist[-1]))
    
    # retornar
    print('parameters: ', params.flatten())
    
    return params, res, score_hist

#------------------------------------------------------------------------------
# KNN model predict
def knn_predict(X, k, X_train, Y_train):
    """
    -> numpy.array(float)
    
    realiza una predicción mediante el modelo KNN sobre los datos train.
    
    :param np.array X:
        serie de tiempo sobre la cual realizar la predicción.
    :param int k:
        k nearest neighbours a considerar en la predicción.
    :param numpy.array X_train:
        serie de datos input de entrenamiento. (num_samples, num_input)
    :param numpy.array Y_train:
        serie de datos output de entrenamiento. (num_samples, num_output)
        
    :returns:
        serie de datos estiamdos (1, num_output)
    """
    
    # generar matriz de distancias
    dist_X = np.zeros((1, X_train.shape[0]))
    
    for i in range(X_train.shape[0]):
        # calcular distancia
        dist_X[0, i] = np.linalg.norm(X - X_train[i,:])
        
    # sort
    index = np.arange(X_train.shape[0])
    sorted_dist = index[ np.argsort(dist_X) ].flatten()
    
    # promediar primeras k series Y_train
    Y_pred = np.mean( Y_train[sorted_dist[0:k], :], axis=0 )
    
    return Y_pred

#------------------------------------------------------------------------------
# persistence model forecast
def persistence_predict(database, forecast_date, n_output, power_cols,
                        lat=-33.45775, lon=70.66466111, Bs=0.0, Zs=0.0, rho=0.2,
                        verbose=False):
    """
    -> numpy.array(floats)
    
    genera un pronóstico de generación fotovoltaica dentro del horizonte
    especificado a partir del criterio de persistencia y el modelo de Perez.
    
    Se supone que tanto el índice de claridad como la fracción difusa del
    timestamp especificado se mantienen constantes dentro de todo el horizonte.
    
    :param DataFrame database:
        base de datos que contiene el registro temporal del sistema fotovoltaico.
    :param str forcast_date:
        fecha y hora en la que realizar el pronóstico %d-%m-%Y %H:%M (UTC).
    :param int n_output:
        cantidad de pronósticos a realizar desde el forcast_date especificado.
    :param str or list(str) power_cols:
        nombre de las columnas que tienen la potencia de los equipos a
        pronosticar.
        
    :param float lat:
        latitud del punto geográfico.
    :param float lon:
        longitud del punto geográfico en grados oeste [0,360).
    :param float Bs: (default, 0.0)
        inclinación de la superficie.
        ángulo de inclinación respecto al plano horizontal de la superficie 
        sobre la cual calcular la irradiancia (grados).
    :param float Zs: (default, 0.0)
        azimut de la superficie.
        ángulo entre la proyección de la normal de la superficie en el plano
        horizontal y la dirección hacia el ecuador (grados).
        
    :returns:
        array con las potencias pronosticadas dentro de el horizonte.
    """
    
    # -------------------------------------------------------------------------
    # obtener datos operacionales en el timestamp
    forecast_date = validate_date(forecast_date)
    date_index = get_date_index( database['Timestamp'], forecast_date )
    
    # checkear que los datos estén contenidos en el database
    input_index = date_index - 1
    end_index = date_index + n_output
    
    # checkear ventana de tiempo
    try:
        assert input_index >= 0
        assert end_index < database.shape[0]
    except AssertionError:
        print('InputError: no es posible aplicar el modelo en la ventana'+
              'de tiempo especificada')
        return None
    
    # obtener potencia en el timestamp de input
    power = 0.0
    
    # agregar la potencia de los equipos
    power_cols = list(power_cols)
    for equip in power_cols:
        power += database.at[input_index, equip]
        
    # obtener parámetros de irradiación
    global_rad = database.at[input_index, 'Global']
    diffuse_rad = database.at[input_index, 'Diffuse']
    
    # obtener timestep de la base de datos
    date_format = '%d-%m-%Y %H:%M'
    data_timestep = get_timestep(database['Timestamp'], date_format)
    input_timestamp = database.at[input_index, 'Timestamp']
    
    ext_rad = ext_irradiation(input_timestamp, data_timestep, lat=lat, lon=lon)
    
    clearsky_index = np.min([1.0, global_rad/ext_rad]) if ext_rad!=0.0 else 1.0
    cloudness_index = np.min([1.0, diffuse_rad/global_rad]) if global_rad!=0.0 else 1.0
    
    # -------------------------------------------------------------------------
    # modelo de Perez
    input_timestamp = datetime.strptime(input_timestamp, date_format)
    
    rad_fh = np.zeros((1, n_output + 1)).flatten()
    
    # incializar matriz de coeficientes
    file_dir = os.path.dirname( os.path.abspath(inspect.getfile(persistence_predict)) )
    file_path = os.path.join( file_dir, 'perez_coefs.csv')
    
    coefs = read_csv_file(file_path)
    
    # para cada timestamp dentro del horizonte de pronostico
    for i in range(n_output + 1):
        # obtener timestamp de pronostico
        new_timestamp = input_timestamp + timedelta(seconds=data_timestep*i)
        new_timestamp = datetime.strftime(new_timestamp, date_format)
        
        # radiacion extraterrestre en el plano horizontal
        gho = ext_irradiation(new_timestamp, data_timestep, lat=lat, lon=lon)
        
        # radiacion global en el plano horizontal
        ghi = gho*clearsky_index
        
        # coeficiente de incidencia
        cos_theta = solar_incidence(new_timestamp,
                                    lat=lat, lon=lon, Bs=Bs, Zs=Zs)
        cos_theta_z = solar_incidence(new_timestamp, lat=lat, lon=lon)
        
        rb = cos_theta/cos_theta_z if cos_theta_z !=0.0 else 0.0
        
        # zenith
        theta_z = acos(cos_theta_z)
        
        # estimacion radiaciones
        dif_i = cloudness_index*ghi
        dir_i = (ghi - dif_i)
        
        # irradiacion difusa circumsolar
        a = np.max([0, cos_theta])
        b = np.max([cos(85.0*pi/180.0), cos_theta_z])
        
        # clearness parameter
        if dif_i==0.0:
            eps = 1.0
        else:
            eps = ( (dif_i - dir_i*rb)/dif_i +
                   5.535e-6*(theta_z*180/pi)**3 )/( 1 + 5.535e-6*(theta_z*180/pi)**3 )
        
        # brightness parameter
        gon = gho/cos_theta_z if cos_theta_z !=0.0 else 0.0
        brp = (1/cos_theta_z)*(dif_i/gon) if (cos_theta_z!=0.0) and (gon!=0.0) else 0.0
        
        # obtener coeficientes
        idx = np.searchsorted(coefs['e'], eps, side='right') - 1
        f11, f12, f13 = coefs.iloc[idx, 1:4]
        f21, f22, f23 = coefs.iloc[idx, 4:7]
        
        # calcular brightness coeficients
        F1 = np.max([0, (f11 + f12*brp + theta_z*f13)])
        F2 = (f21 + f22*brp + theta_z*f23)
        
        # corregir angulos de la superfice
        Bs = Bs*pi/180
        Zs = Zs*pi/180
        
        # calcular irradiación total
        gti = ( dir_i*rb + dif_i*(1 - F1)*(1 + cos(Bs))/2 + dif_i*F1*a/b
               + dif_i*F2*sin(Bs) + ghi*rho*(1 - cos(Bs))/2 )
        rad_fh[i] = gti
      
    # calcular estimaciones de potencia
    forecast_output = np.zeros((1, n_output)).flatten()
    
    # si hay información para realizar un pronóstico
    if not(rad_fh[i]==0.0 or power==0.0):
        for i in range(n_output):
            forecast_output[i] = (rad_fh[i+1]/rad_fh[0])*power
    
    # si se desea imprimir información
    if verbose:
        # radiacion global
        global_rad = database['Global'].values[date_index:end_index]
        
        # potencia sistema
        syst_power = np.zeros((1, n_output)).flatten()
        for equip in power_cols:
            syst_power = syst_power + database[equip].values[date_index:end_index]
        
        print('forecasted radiation: ' + str(rad_fh[0:]) )
        print('global radiation: ' + str(global_rad) + '\n')
        print('forecasted power: ' + str(forecast_output) )
        print('system power: ' + str(syst_power) )
    
        
    return np.reshape(forecast_output, (n_output, -1))
        
        
        
        
        
    
    
    
    
    