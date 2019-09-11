# -*- coding: utf-8 -*-
"""
Proyecto Beauchef PV Forecasting

Cristian Herrera

v1.0 - update, Aug 01 2019
"""

import pandas as pd
pd.options.mode.chained_assignment = None

from datetime import datetime

from solarpv.noaa._api import get_key_times
from solarpv.noaa._tools import get_bound_indexes, bound_goes16_data

from progressbar import ProgressBar

import matplotlib.pyplot as plt

import numpy as np
import os
from time import sleep

# -----------------------------------------------------------------------------
# base de datos NOAA GOES16
def goes16_dataset(dir_path, size, lat=-33.45775, lon=-70.66466111,
                   adjust_time=0, fix_timestamps=True):
    """
    -> pandas.DataFrame
    
    Procesa los archivos netCDF4 contenidos en el directorio especificado
    contruyendo un DataFrame de imagenes satelitales.
    
    :param str dir_path:
        ubicacion de los archivos netCDF4 (.nc) que se desea procesar.
    :param int size:
        tamaño de las imagenes resultantes en el procesamiento.
    :param float lat:
        latitud central del área de interés.
    :param float lon:
        longitud central del área de interés.
    :param str adjust_time:
        permite corregir en segundos el timestamp asociado a cada dato.
    :param bool fix_timestamps:
        si corregir la serie de timestamps en el dataset.
        
          
    :returns: DataFrame de la base de datos
    """
    print('\n' + '='*80)
    print('Parsing NOAA-GOES16 Data')
    print('='*80 + '\n')
    
    # obtener cantidad de imágenes en el directorio ---------------------------
    files_list = os.listdir(dir_path)
    
    # ordenar los archivos netCDF4
    nc_list = []
    
    for f in files_list:
        file_name, ext = os.path.splitext(f)
        
        # si no es un archivo .nc
        if ext != '.nc':
            continue
        
        # obtener timestamp
        start_date, _ = get_key_times(file_name)
        start_date = datetime.strptime(start_date, '%d-%m-%Y %H:%M:%S')
        nc_list.append( (start_date, f) )
    
    # ordenar
    nc_list.sort()
    
    # inicializar dataset -----------------------------------------------------
    colnames = [u'Timestamp'] + list(np.arange(size*size))
    
    db = pd.DataFrame(0.0, index=np.arange(len(nc_list)), columns=colnames)
    db[u'Timestamp'] = db[u'Timestamp'].astype(str)
    
    # obtener bound indexes ---------------------------------------------------
    print('getting bounds indexes:', end =" ") 
    data_path = os.path.join(dir_path, nc_list[0][1])
    bound_indexes = get_bound_indexes(data_path, lat, lon, size)
    
    print(' done\n')
    sleep(0.25)
    
    # formatear dataset -------------------------------------------------------
    date_format = '%d-%m-%Y %H:%M'
    
    bar = ProgressBar()
    for i in bar( db.index ):
        timestamp, data_key = nc_list[i]
        
        # formatear timestamp
        db.at[i, u'Timestamp'] = datetime.strftime(timestamp, date_format)
        
        # formatear data
        data_path = os.path.join(dir_path, data_key)
        data = bound_goes16_data(data_path, bound_indexes)

        db.at[i, 1:] = data.reshape( (1,-1) )[0]
        
    if not(fix_timestamps):
        return db
    
    # -------------------------------------------------------------------------
    # añadir timestamps faltantes
    
    # obtener timestep del dataset
    timestep = (  datetime.strptime(db.at[1,u'Timestamp'], date_format)
                - datetime.strptime(db.at[0,u'Timestamp'], date_format) )
    timestep = timestep.seconds
    
    
    time_freq = str(timestep) + 'S'
    idx = pd.date_range(db['Timestamp'].iloc[0], db['Timestamp'].iloc[-1],
                        freq=time_freq)
    
    db.index = pd.DatetimeIndex( db[u'Timestamp'].values, dayfirst=True )
    db = db.reindex( idx, fill_value=0.0 )
    db[u'Timestamp'] = idx.strftime(date_format)
    
    # resetear index a enteros
    db = db.reset_index()
    db.drop('index',axis=1, inplace=True)
  
    return db

# -----------------------------------------------------------------------------
# plot base de datos NOAA GOES16
def plot_goes16_dataset(database, index, shape, gamma_correction=False):
    """
    -> None
    
    Plotea las imagenes satelitales contenidas en el dataset entregado que
    correspondan a los índices en index.
    
    :param DataFrame database:
        base de datos que contiene el registro de imagenes satelitales.
    :param list or array-like index:
        indices a considerar en el plot.
    :param bool gama_correction:
        si se desea aplicar gamma correction sobre la imagen.
        
    :returns:
        None
    """
    
    # inicializar figura ------------------------------------------------------
    
    for i in index:
        try:
            # arreglar datos
            rad_img = database.iloc[i, 1:].values.astype(float)
            rad_img = rad_img.reshape( shape )
            
            # aplicar gamma correction
            if gamma_correction:
                rad_img = (rad_img * np.pi * 0.3) / 441.868715
                # Make sure all data is in the valid data range
                rad_img = np.maximum(rad_img, 0.0)
                rad_img = np.minimum(rad_img, 1.0)
                rad_img = np.sqrt(rad_img)
            
            plt.imshow(rad_img)
            
        except KeyError:
            continue
        
    # retornar    
    return None