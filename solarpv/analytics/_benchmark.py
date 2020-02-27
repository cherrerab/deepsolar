# -*- coding: utf-8 -*-
"""
Proyecto Beauchef PV Forecasting

Cristian Herrera

v1.0 - update, Aug 01 2019
"""


import matplotlib.pyplot as plt
import random
import os
import inspect

from datetime import datetime, timedelta

import pandas as pd

from solarpv._tools import validate_date, get_date_index, ext_irradiation, get_timestep, solar_incidence
from solarpv.database import read_csv_file

import numpy as np
from math import (pi, cos, sin, tan, acos)
from scipy.stats import skew, kurtosis


#------------------------------------------------------------------------------
# mean absolute error
def mean_squared_error(Y_true, Y_pred):
    """
    -> float
    
    calcula el error cuadrado medio de la diferencia entre Y_true e Y_pred.
    
    :param np.array Y_true:
        serie de valores reales.
    :param np.array Y_pred:
        serie de valores estimados.
    
    :returns:
        error cuadrado medio de la diferencia.
    """
    
    # calcular error
    error = Y_pred.flatten() - Y_true.flatten()
    
    # calcular error cuadrado medio
    mse = np.sqrt(np.mean(np.power(error, 2)))
    
    return mse

#------------------------------------------------------------------------------
# mean absolute error
def mean_absolute_error(Y_true, Y_pred):
    """
    -> float
    
    calcula el error absoluto medio de la diferencia entre Y_true e Y_pred.
    
    :param np.array Y_true:
        serie de valores reales.
    :param np.array Y_pred:
        serie de valores estimados.
    
    :returns:
        error absoluto medio de la diferencia.
    """
    
    # calcular error
    error = Y_pred.flatten() - Y_true.flatten()
    
    # calcular error absoluto medio
    mae = np.mean( np.abs(error) )
    
    return mae

#------------------------------------------------------------------------------
# mean bias error
def mean_bias_error(Y_true, Y_pred):
    """
    -> float
    
    calcula el error medio de la diferencia entre Y_true e Y_pred.
    
    :param np.array Y_true:
        serie de valores reales.
    :param np.array Y_pred:
        serie de valores estimados.
    
    :returns:
        error medio de la diferencia.
    """
    
    # calcular error
    error = Y_pred.flatten() - Y_true.flatten()
    
    # calcular error absoluto medio
    mae = np.mean( error )
    
    return mae

#------------------------------------------------------------------------------
# skew error
def skew_error(Y_true, Y_pred, **kargs):
    """
    -> float
    
    calcula el skewness de la diferencia entre Y_true e Y_pred.
    
    :param np.array Y_true:
        serie de valores reales.
    :param np.array Y_pred:
        serie de valores estimados.
    
    :returns:
        skewness de la diferencia.
    """
    
    # calcular error
    error = Y_pred.flatten() - Y_true.flatten()
    
    # calcular kurtosis
    skw = skew(error, **kargs)
    
    return skw

#------------------------------------------------------------------------------
# kurtosis
def kurtosis_error(Y_true, Y_pred, **kargs):
    """
    -> float
    
    calcula el kurtosis de la diferencia entre Y_true e Y_pred.
    
    :param np.array Y_true:
        serie de valores reales.
    :param np.array Y_pred:
        serie de valores estimados.
    
    :returns:
        kurtosis de la diferencia.
    """
    
    # calcular error
    error = Y_pred.flatten() - Y_true.flatten()
    
    # calcular kurtosis
    kts = kurtosis(error, **kargs)
    
    return kts

#------------------------------------------------------------------------------
# persistence model forecast
def persistence_forecast(database, forecast_date, n_output, power_cols,
                         lat=-33.45775, lon=70.66466111, Bs=0.0, Zs=0.0, rho=0.2):
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
    clearsky_index = global_rad/ext_rad if ext_rad!=0.0 else 0.0
    cloudness_index = diffuse_rad/global_rad if global_rad!=0.0 else 0.0
    
    # -------------------------------------------------------------------------
    # modelo de Perez
    input_timestamp = datetime.strptime(input_timestamp, date_format)
    
    rad_fh = np.zeros((1, n_output + 1)).flatten()
    
    # incializar matriz de coeficientes
    file_dir = os.path.dirname( os.path.abspath(inspect.getfile(persistence_forecast)) )
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
        dir_i = ghi - dif_i
        
        # irradiacion difusa circumsolar
        a = np.max([0, cos_theta])
        b = np.max([cos(85.0*pi/180.0), cos_theta_z])
        
        # clearness parameter
        eps = ( (dif_i - dir_i*rb)/dif_i +
               5.535e-6*(theta_z*180/pi)**3 )/( 1 + 5.535e-6*(theta_z*180/pi)**3 )
        
        # brightness parameter
        gon = gho/cos_theta_z if cos_theta_z !=0.0 else 0.0
        brp = (1/cos_theta_z)*(dif_i/gon) if cos_theta_z !=0.0 else 0.0
        
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
    for i in range(n_output):
        forecast_output[i] = (rad_fh[i+1]/rad_fh[0])*power
        
    return forecast_output
        
#------------------------------------------------------------------------------
# forecast skill
def forecast_skill(Y_true, Y_pred, Y_base, **kargs):
    """
    -> float
    
    calcula el forecast skill (%) entre los valores Y_pred e Y_base.
    
    :param np.array Y_true:
        serie de valores reales.
    :param np.array Y_pred:
        serie de valores estimados.
    :param np.array Y_pred:
        serie de valores estimados mediante un modelo base.
    
    :returns:
        forecast skill de los valores estimados.
    """
    
    # calcular rmse Y_pred
    rmse = np.sqrt( mean_squared_error(Y_true, Y_pred) )
    
    # calcular rmse Y_base
    rmse_b = np.sqrt( mean_squared_error(Y_true, Y_base) )
    
    # calcular forecast skill
    fs = 1.0 - rmse/rmse_b
    
    return fs
        
        
        
        
    
    
    
    
    