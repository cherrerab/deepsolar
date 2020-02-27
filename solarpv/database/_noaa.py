# -*- coding: utf-8 -*-
"""
Proyecto Beauchef PV Forecasting

Cristian Herrera

v1.0 - update, Aug 01 2019
"""

import pandas as pd
pd.options.mode.chained_assignment = None

from datetime import datetime, timedelta

from solarpv.noaa._api import get_key_times
from solarpv.noaa._tools import get_bound_indexes, bound_goes16_data
from solarpv._tools import validate_date, get_timestep

from progressbar import ProgressBar

import matplotlib.pyplot as plt

import numpy as np
import os
from time import sleep

import cv2
from skimage.transform import resize


from keras.models import Model

# -----------------------------------------------------------------------------
# base de datos NOAA GOES16
def goes16_dataset(dir_path, timestamps, size, lat=-33.45775, lon=-70.66466111,
                   adjust_time=0, fix_timestamps=True):
    """
    -> pandas.DataFrame
    
    Procesa los archivos netCDF4 contenidos en el directorio especificado
    contruyendo un DataFrame de imagenes satelitales a partir de la serie de
    timestamps entregados.
    
    :param str or list dir_path:
        ubicacion de los archivos netCDF4 (.nc) que se desea procesar.
    :param list or array-like timestamps:
        serie de timestamps de las imágenes que se desean procesar.
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
    print('\n' + '-'*80)
    print('processing noaa-goes16 data')
    print('-'*80 + '\n')
    
    # obtener lista de archivos netCDF4 a procesar ----------------------------
    nc_list = []
    
    # por cada una de las direcciones especificadas
    if type(dir_path) is str:
        dir_path = [dir_path]
        
    for path in list(dir_path):
        files_list = os.listdir(path)
        
        # por cada archivo en el directorio
        for f in files_list:
            file_name, ext = os.path.splitext(f)
            
            # si no es un archivo .nc
            if ext != '.nc':
                continue
            
            # obtener timestamp
            start_date, end_date = get_key_times(file_name)
            
            start_date = datetime.strptime(start_date, '%d-%m-%Y %H:%M:%S')
            end_date = datetime.strptime(end_date, '%d-%m-%Y %H:%M:%S')
            
            nc_list.append( (start_date, end_date, os.path.join(path, f)) )
        
    # ordenar
    nc_list.sort()
    
    # inicializar dataset -----------------------------------------------------
    colnames = [u'Timestamp', 'Start Time', 'End Time'] + list(np.arange(size*size))
    num_data = len(timestamps)
    
    db = pd.DataFrame(0.0, index=np.arange(num_data), columns=colnames)
    db[u'Timestamp'] = db[u'Timestamp'].astype(str)
    
    # obtener bound indexes ---------------------------------------------------
    print('getting bounds indexes:', end =" ") 
    bound_indexes = get_bound_indexes(nc_list[0][1], lat, lon, size)
    
    print(' done\n')
    sleep(0.25)
    
     # formatear dataset -------------------------------------------------------
    date_format = '%d-%m-%Y %H:%M'
    
    bar = ProgressBar()
    for i in bar( db.index ):
        # obtener timestamp a procesar
        timestamp = validate_date(timestamps[i])
        timestamp = datetime.strptime(timestamp, date_format)
        
        # formatear timestamp
        db.at[i, u'Timestamp'] = datetime.strftime(timestamp, date_format)
        
        # obtener imagen satelital correspondiente
        for j in range( len(nc_list) ):
            # obtener timestamp de inicio del scaneo
            data_start, _, data_path = nc_list[j]
            
            # si la imagen fue tomada después del timestamp de  interés
            if timestamp < data_start:
                # la imagen correspondiente es la tomada anteriormente
                data_start, data_end, data_path = nc_list[j-1]
                break
        
        # agregar timestamps de scaneo
        db.at[i, 'Start Time'] = datetime.strftime(data_start, date_format)
        db.at[i, 'End Time'] = datetime.strftime(data_end, date_format)
        
        # formatear data
        data = bound_goes16_data(data_path, bound_indexes)

        db.at[i, 3:] = data.reshape( (1,-1) )[0]
        
    if not(fix_timestamps):
        return db
    
    # -------------------------------------------------------------------------
    # añadir timestamps faltantes
    
    # obtener timestep del dataset
    timestep = get_timestep(db['Timestamp'], date_format)
    
    time_freq = str(timestep) + 'S'
    first_date = datetime.strptime( db['Timestamp'].iloc[0], date_format)
    last_date = datetime.strptime( db['Timestamp'].iloc[-1], date_format)
    
    idx = pd.date_range(first_date, last_date, freq=time_freq)
    
    db.index = pd.DatetimeIndex( db[u'Timestamp'].values, dayfirst=True )
    db = db.reindex( idx, fill_value=0.0 )
    db[u'Timestamp'] = idx.strftime(date_format)
    
    # resetear index a enteros
    db = db.reset_index()
    db.drop('index',axis=1, inplace=True)
  
    return db

# -----------------------------------------------------------------------------
# base de datos NOAA GOES16
def goes16_dataset_v2(dir_path, timestamps, size, lat=-33.45775, lon=-70.66466111,
                   adjust_time=0, fix_timestamps=True):
    """
    -> pandas.DataFrame
    
    Procesa los archivos netCDF4 contenidos en el directorio especificado
    contruyendo un DataFrame de imagenes satelitales a partir de la serie de
    timestamps entregados.
    
    :param str dir_path:
        ubicacion de los archivos netCDF4 (.nc) que se desea procesar.
    :param list or array-like timestamps:
        serie de timestamps de las imágenes que se desean procesar.
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
    first_date = datetime.strptime( db['Timestamp'].iloc[0], date_format)
    last_date = datetime.strptime( db['Timestamp'].iloc[-1], date_format)
    
    idx = pd.date_range(first_date, last_date, freq=time_freq)
    
    db.index = pd.DatetimeIndex( db[u'Timestamp'].values, dayfirst=True )
    db = db.reindex( idx, fill_value=0.0 )
    db[u'Timestamp'] = idx.strftime(date_format)
    
    # resetear index a enteros
    db = db.reset_index()
    db.drop('index',axis=1, inplace=True)
  
    return db

# -----------------------------------------------------------------------------
# resize imágenes satelitales
def resize_goes16_dataset(database, dim, **kargs):
    """
    -> DataFrame
    
    Realiza un cv2.resize sobre cada una de las imágenes en el dataset.
    
    :param DataFrame dataset:
        base de datos que contiene el registro temporal de imagenes satelitales.
    :param tuple dim:
        tamaño final de cada una de las imágenes en el dataset.
        
    :returns:
        Dataframe con las imágenes con el nuevo tamaño.
    """
    
    print('\n' + '-'*80)
    print('resizing noaa-goes16 dataset')
    print('-'*80 + '\n')
    
    # obtener tamaños
    img_size = int( np.sqrt(database.shape[1] - 1) )
    new_height, new_width = dim
    
    # inicializar nuevo dataframe
    num_rows = int( new_height*new_width )
    colnames = [u'Timestamp'] + list(np.arange(num_rows))
    db = pd.DataFrame(0.0, index=database.index, columns=colnames)
    db[u'Timestamp'] = db[u'Timestamp'].astype(str)
    db[u'Timestamp'] = database[u'Timestamp']
    
    # por cada una de las imágenes contenidas en el dataset
    bar = ProgressBar()
    for i in bar( database.index ):
        # reshape imagen
        rad_data = np.nan_to_num( database.iloc[i, 1:].values )
        rad_data = np.reshape(rad_data, (img_size, img_size, 1) )
        
        # resize
        resized_data = resize( np.float32(rad_data) , dim, **kargs )
        
        # agregar al nuevo dataframe
        db.at[i, 1:] = resized_data.reshape((1, new_height*new_width))[0]
        
    return db

# -----------------------------------------------------------------------------
# reducir dimensionalmente base de datos NOAA GOES16
def encode_goes16_dataset(database, autoencoder, latent_space_name):
    """
    -> DataFrame
    
    Realiza una reducción dimensional sobre cada una de las imágenes satelitales
    contenidas en dataset a partir del output del espacio latente del autoencoder.
    
    :param DataFrame dataset:
        base de datos que contiene el registro temporal de imagenes satelitales.
    :param keras.model autoencoder:
        Autoencoder entrenado sobre el mismo tipo de imágenes satelitales.
    :param str latent_space_name:
        Nombre de la capa correspondiente al espacio latente del autoencoder.
        
    :returns:
        Dataframe con el espacio latente de cada imagen del database.
    """
    
    # configurar encoder
    latent_space = autoencoder.get_layer(latent_space_name)
    encoding_dim = int(latent_space.output.shape[1])
    
    encoder = Model(inputs=autoencoder.input, output=latent_space.output)
    
    # comprobar correspondencia entre tamaños de imágenes
    img_size = np.sqrt(database.shape[1] - 1)
    assert int(autoencoder.input.shape[1]) == img_size
    
    # procesar base de datos --------------------------------------------------
    
    # inicializar nuevo dataframe
    colnames = [u'Timestamp'] + list(np.arange(encoding_dim))
    db = pd.DataFrame(0.0, index=database.index, columns=colnames)
    db[u'Timestamp'] = db[u'Timestamp'].astype(str)
    db[u'Timestamp'] = database[u'Timestamp']
    
    bar = ProgressBar()
    for i in bar( database.index ):
        
        # procesar imagen original
        rad_data = database.iloc[i, 1:].values
        rad_data = np.nan_to_num( np.reshape(rad_data, [1, img_size, img_size, 1]) )
        
        # obtener reducción dimensional
        encoded_data = encoder.predict(rad_data)
        
        # añadir datos al nuevo dataset
        db.at[i, 1:] = encoded_data.reshape( (1, encoding_dim) )[0]
        
    # retornar dataset
    return db

# -----------------------------------------------------------------------------
# plot base de datos NOAA GOES16
def show_goes16_dataset(database, shape, index='all', gamma_correction=False):
    """
    -> None
    
    Plotea las imagenes satelitales contenidas en el dataset entregado que
    correspondan a los índices en index.
    
    :param DataFrame database:
        base de datos que contiene el registro de imagenes satelitales.
    :param int shape:
        tamaño en pixeles de la ventana de la imágenes.
    :param list or array-like index:
        indices a considerar en el plot.
    :param bool gama_correction:
        si se desea aplicar gamma correction sobre la imagen.
        
    :returns:
        None
    """
    
    # si se define utilizar todas las imágenes
    if index == 'all':
        index = database.index
        
    # para cada uno de los índices especificados
    for i in index:
        try:
            # arreglar datos
            rad_img = database.iloc[i, 1:].values.astype(float)
            rad_img = rad_img.reshape( (shape,shape) )
            
            # aplicar gamma correction
            if gamma_correction:
                rad_img = (rad_img * np.pi * 0.3) / 441.868715
                # Make sure all data is in the valid data range
                rad_img = np.maximum(rad_img, 0.0)
                rad_img = np.minimum(rad_img, 1.0)
                rad_img = np.sqrt(rad_img)
                
            rad_img = cv2.resize(rad_img, (500, 500))
            
            cv2.imshow('goes-16 dataset', rad_img)
            cv2.waitKey(200)
            
        except KeyError:
            continue
        
        # exit()
        if cv2.waitKey(1) == 27:
            break
        
    # retornar
    cv2.destroyAllWindows()
    return None