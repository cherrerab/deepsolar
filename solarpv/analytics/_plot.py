# -*- coding: utf-8 -*-
"""
Proyecto Beauchef PV Forecasting

Cristian Herrera

v1.0 - update, Aug 01 2019
"""

import matplotlib.pyplot as plt

import numpy as np

from datetime import datetime, timedelta

from solarpv.database import reshape_by_day
from solarpv.analytics import ext_irradiance, ext_irradiation

from solarpv import (validate_date, get_date_index)

#------------------------------------------------------------------------------
# obtener gráfico fracción difusa vs claridad
def plot_fraccion_difusa_claridad(database, lat=-33.45775, lon=70.66466111):
    """
    -> None
    
    Plotea el gráfico de fracción difusa vs claridad a partir de la base de
    datos entregada.
    
    :param DataFrame database:
        base de datos que contiene el registro temporal de irradiancia a
        procesar.
        
    :return:
        None
    """
    
    diffuse_rad = database['Diffuse'].values
    global_rad = database['Global'].values
    
    frac_dif = np.divide(diffuse_rad, global_rad)
    
    timestamp = database['Timestamp'].values
    ext_rad = [ext_irradiance(t, lat, lon) for t in timestamp]
    claridad = np.divide( global_rad, np.asarray(ext_rad) )
    
    plt.figure()
    plt.scatter(claridad, frac_dif, c='k', s=0.8)
    plt.ylim( (0, 1.0) )
    plt.xlim( (0, 1.0) )
    
    plt.xlabel(u'Índice de Claridad')
    plt.ylabel(u'Fracción Difusa')
    
    return None

#------------------------------------------------------------------------------
# obtener gráfico 2D de radiación en el tiempo
def plot_2D_radiation_data(database, unit='', **kargs):
    """
    -> None
    
    Plotea el gráfico 2D de radiación en el tiempo. Donde el eje X corresponde
    a los días, y el eje Y a los minutos de cada día.
    
    :param DataFrame database:
        base de datos que contiene el registro de radiación en el tiempo.
    :param str unit:
        define la unidad de medición de los datos.
        
    :return:
        None
    """
    
    # transformar base de datos a formato diarioxhorario ----------------------
    df = reshape_by_day(database, **kargs)
    
    # plotear mapa temporal de radiación --------------------------------------
    plt.figure()
    plt.imshow(df.values, cmap='plasma', aspect='auto')
    
    # ajustar ticks en el plot
    initial_hour = datetime.strptime('00:00','%H:%M')
    hours = [initial_hour + timedelta(seconds=7200*x) for x in range(24)]
    hours_ticks = [datetime.strftime(h, '%H:%M') for h in hours]
    
    num_days = len(df.columns)
    num_ticks = min( num_days, 5 )
    delta_days = round( num_days/num_ticks )
    
    initial_day = datetime.strptime(df.columns[0], '%d-%m-%Y')
    
    days = [initial_day + timedelta(days=delta_days*x) for x in range(num_ticks)]
    
    day_ticks = [datetime.strftime(d, '%d-%m-%Y') for d in days]
    
    # set ticks
    data_delta = (  datetime.strptime(df.index[1], '%H:%M')
                  - datetime.strptime(df.index[0], '%H:%M') )
    data_delta = data_delta.seconds
    y_delta = int(3600/data_delta)
    
    plt.yticks( range(0, 24*y_delta, 2*y_delta), hours_ticks)
    plt.xticks( range(0, num_ticks*delta_days, delta_days), day_ticks )
    
    # agregar colorbar        
    cbar = plt.colorbar()
    cbar.set_label(unit)

    plt.show()
    return None
    
#------------------------------------------------------------------------------
# obtener gráfico 1D de radiación en el tiempo
def plot_1D_radiation_data(database, colname, start_date, stop_date, 
                         extraRad=True, lat=-33.45775, lon=70.66466111):
    """
    -> None
    Plotea el gráfico temporal 1D de la radiación en el tiempo. Donde el eje X
    corresponde al tiempo y el eje Y a la intensidad de la radiación.
    
    :param DataFrame database:
        base de datos que contiene el registro de radiación en el tiempo.
    :param str colname:
        nombre de la columna a graficar.
    :param str start_date:
        fecha desde la cual empezar el plot.
    :param str stop_date:
        fecha en que termina el plot.
    :param bool extraRad:
        si incluir en el plot la radiación extraterreste.
    :param float lat:
        latitud del punto geográfico.
    :param float lon:
        longitud del punto geográfico en grados oeste [0,360).
        
    :return:
        None
    """
    
    # obtener limites del plot ------------------------------------------------
    tmstmp = database['Timestamp'].values
    date_format = '%d-%m-%Y %H:%M'
    
    start_date = datetime.strptime( validate_date(start_date), date_format )
    stop_date = datetime.strptime( validate_date(stop_date), date_format )
    
    start_index = get_date_index(tmstmp, start_date, nearest=True)
    stop_index = get_date_index(tmstmp, stop_date, nearest=True)
    
    if stop_index <= start_index:
        print('InputError: las fechas ingresadas no permiten generar un plot.')
        return None
    
    # plotear -----------------------------------------------------------------
    Y = database[colname][start_index:stop_index]
    timestamps = tmstmp[start_index:stop_index]
    X = range( len(timestamps) )
    
    plt.figure()
    plt.plot(X, Y, c='k', ls='-', lw=0.8, label='Data')
    
    # añadir readiación extraterrestre ----------------------------------------
    if extraRad:
        timestep = (datetime.strptime(timestamps[1], date_format)
                    - datetime.strptime(timestamps[0], date_format))
        
        secs = timestep.seconds
        
        ext_rad = [ext_irradiation(t, secs, lat, lon) for t in timestamps]
        plt.plot(X, ext_rad, c='k', ls='--', lw=0.8, label='Extraterrestrial')   
    
    # colocar etiquetas en el gráfico -----------------------------------------
    plt.xlabel('Data points')
    plt.ylabel('Radiación Wh/m2')
    
    plt.legend(loc='best')
    
    title = tmstmp[start_index] + ' - ' + tmstmp[stop_index]
    plt.title(title)
    return None
    
#------------------------------------------------------------------------------
# obtener gráfico de performance ratio de la planta fotovoltaica
def plot_performance_ratio(db_pv, db_solar, start_date, stop_date,
                           lat=-33.45775, lon=70.66466111):
    """
    -> None
     Plotea el gráfico de performance ratio vs claridad a partir de la base de
     datos entregada.
    
    :param DataFrame db_pv:
        base de datos que contiene el registro de 'Potencia' fotovoltaica.
    :param DataFrame db_solar:
        base de datos que contiene el registro de irradiancia 'Global'.
    :param float lat:
        latitud del punto geográfico.
    :param float lon:
        longitud del punto geográfico en grados oeste [0,360).
        
    :return:
        None
    """
    
    # -------------------------------------------------------------------------
    # verificar bases de datos
    assert ('Timestamp' in db_pv.columns) and ('Potencia' in db_pv.columns)
    assert ('Timestamp' in db_solar.columns) and ('Global' in db_solar.columns)
    
    # verificar que los timesteps se correspondan
    date_format = '%d-%m-%Y %H:%M'
    
    timestep_pv = (datetime.strptime(db_pv.at[1, u'Timestamp'], date_format)
                  -datetime.strptime(db_pv.at[0, u'Timestamp'], date_format) )
    
    timestep_solar = (datetime.strptime(db_solar.at[1, u'Timestamp'], date_format)
                     -datetime.strptime(db_solar.at[0, u'Timestamp'], date_format) )
    
    assert timestep_solar.seconds==timestep_pv.seconds
    
    # -------------------------------------------------------------------------
    # obtener indices
    start_date = datetime.strptime( validate_date(start_date), date_format )
    stop_date = datetime.strptime( validate_date(stop_date), date_format )
    
    # photovoltaic
    pv_index = [-1,-1]
    for i in db_pv.index:
        date = datetime.strptime(db_pv.at[i,u'Timestamp'], date_format)
        
        if (date >= start_date) and (pv_index[0] == -1):
            pv_index[0] = i
            
        if (date >= stop_date) and (pv_index[1] == -1):
            pv_index[1] = i-1
    
    # solarimetric
    solar_index = [-1,-1]
    for i in db_solar.index:
        date = datetime.strptime(db_solar.at[i,u'Timestamp'], date_format)
        
        if (date >= start_date) and (solar_index[0] == -1):
            solar_index[0] = i
            
        if (date >= stop_date) and (solar_index[1] == -1):
            solar_index[1] = i-1
     
    assert (solar_index[1]-solar_index[0]) == (pv_index[1]-pv_index[0])
    
    # -------------------------------------------------------------------------
    
    # obtener performance ratio
    global_rad = db_solar['Global']
    global_rad = global_rad.values[ solar_index[0]:solar_index[1] ]
    
    pv_power = db_pv['Potencia']
    pv_power = pv_power.values[ pv_index[0]:pv_index[1] ]
    
    plant_factor = 1000.0/16.2
    performance_ratio = []
    for i in range( len(pv_power) ):
        if ( global_rad[i] != 0.0 ):
            performance_ratio.append( plant_factor*(pv_power[i]/global_rad[i]) )
        else:
            performance_ratio.append(np.nan)
    
    performance_ratio = np.array(performance_ratio)
        
    # obtener claridad
    timestamp = db_solar['Timestamp'].values[ solar_index[0]:solar_index[1] ]
    ext_rad = [ext_irradiance(t, lat, lon) for t in timestamp]
    
    claridad = []
    for i in range( len(pv_power) ):
        if ( ext_rad[i] != 0.0 ):
            claridad.append( global_rad[i]/ext_rad[i] )
        else:
            claridad.append( np.nan )
            
    claridad = np.array(claridad)
    
    # plotear
    plt.figure()
    plt.scatter(claridad, performance_ratio, c='k', s=0.8)
    
    plt.xlabel(u'Índice de Claridad')
    plt.ylabel(u'Performance Ratio')
    plt.xlim([0, 1.0])

    return None
    
    
    
    
    
    
    
    
    
    
    
    
        