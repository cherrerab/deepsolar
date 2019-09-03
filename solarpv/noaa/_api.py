# -*- coding: utf-8 -*-
"""
Proyecto Beauchef PV Forecasting

Cristian Herrera

v1.0 - update, Aug 01 2019
"""

import xarray as xr
import requests
import netCDF4
import boto3
import matplotlib.pyplot as plt

from solarpv import validate_date

from datetime import datetime, timedelta

import os
from time import sleep

from progressbar import ProgressBar

import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# obtener timpos de escaneo
def get_key_times(key):
    """
    -> tuple(str, str)
    
    Retorna una tupla con los timestamps de comienzo y fin de la operación
    de escaneo del key especificado.
    
    :param str key:
        key desde el cual se desea obtener los tiempos de escaeno.
        
    :returns:
        tupla con los timestamps en formato %d-%m-%Y %H:%M:%S.
    """
    
    # obtener tiempos del key -------------------------------------------------
    _, _, _, start, end, creat = key.split('_')
    
    # tiempo de comienzo
    start_year = int(start[1:5])
    start_yday = int(start[5:8])
    start_hour = int(start[8:10])
    start_min = int(start[10:12])
    start_sec = round(int(start[12:15])/10.0)
    
    start_date = datetime(start_year, 1, 1, start_hour, start_min, start_sec)
    start_date += timedelta( days=(start_yday-1) )
    start_date = datetime.strftime(start_date, '%d-%m-%Y %H:%M:%S')
    
    # tiempo de termino
    end_year = int(end[1:5])
    end_yday = int(end[5:8])
    end_hour = int(end[8:10])
    end_min = int(end[10:12])
    end_sec = round(int(end[12:15])/10.0)
    
    end_date = datetime(end_year, 1, 1, end_hour, end_min, end_sec)
    end_date += timedelta( days=(end_yday-1) )
    end_date = datetime.strftime(end_date, '%d-%m-%Y %H:%M:%S')
    
    return (start_date, end_date)
    
# -----------------------------------------------------------------------------
# obtener keys desde el bucket
def get_keys(bucket, prefix=''):
    """
    -> list(str)
    
    Retorna una lista con las keys registradas en el aws s3 bucket que
    comienzan con el prefijo especificado.
    
    :param str bucket:
        nombre del s3 bucket.
    :param str prefix: (default '')
        prefijo de las <Product>/<Year>/<Day of Year>/<Hour>/<Filename> keys.
        
    :returns:
        lista de las keys que comienzan con el prefijo.
    """
    
    # -------------------------------------------------------------------------
    # verificar si el prefijo entregado es un string
    assert isinstance(prefix, str)
    
    # inicializar un cliente con Amazon Simple Storage Service (s3)
    s3 = boto3.client('s3')
    
    kwargs = {'Bucket' : bucket,
              'Prefix' : prefix}
    
    # -------------------------------------------------------------------------
    # comenzar a listar las keys del bucket que comienzan con el prefijo
    KEY_LIST = []
    while True:
        # obtener la lista de objects en el bucket
        response = s3.list_objects_v2(**kwargs)
        
        # filtrar aquellas keys que comienzan con el prefijo
        for obj in response['Contents']:
            key = obj['Key']
            if key.startswith(prefix):
                KEY_LIST.append(key)
        
        # si el bucket está truncado, obtner el token siguiente
        try:
            kwargs['ContinuationToken'] = response['NextContinuationToken']
        except KeyError:
            break
            
    # retornar lista de keys
    return KEY_LIST

# -----------------------------------------------------------------------------
# descargar imagen satelital
def download_goes16_data(timestamp, product='ABI-L1b-RadF', mode='M6',
                         channel='C04'):
    """
    -> netCDF4.Dataset
    
    Descarga desde el bucket aws s3 de la noaa la imagen satelital más cercana
    al timestamp especificado.
    
    :param str timestamp:
        fecha en formato %d-%m-%Y %H:%M de la imagen que se desea descargar.
        el criterio para asociar una imagen a un timestamp es que este último
        esté entre el comienzo y el fin de escaneo.
    :param str product:
        producto generado por uno de los instrumentos en el goes16.
    :param str mode:
        modo de la operación de escaneo realizada por el goes16.
    :param str channel:
        canal o banda que se desea obtener.
        
    :returns:
        netCDF4.Dataset de la imagen satelital.
    """
    
    # estructurar data timestamp ----------------------------------------------
    date_format = '%d-%m-%Y %H:%M'
    
    timestamp = validate_date(timestamp)
    timestamp = datetime.strptime(timestamp, date_format)
    date_tt = timestamp.timetuple()
    
    year = str(date_tt.tm_year)
    yday = str(date_tt.tm_yday)
    hour = date_tt.tm_hour
    
    KEYS = []
    for i in range(-1,2):
        h = str(hour+i)
        # armar data query ----------------------------------------------------
        filename_prefix = 'OR_'+ product + '-' + mode + channel
        data_prefix = '/'.join([product, year, yday, h, filename_prefix])
    
        # obtener keys del bucket ---------------------------------------------
        KEYS += get_keys(bucket = 'noaa-goes16', prefix = data_prefix)
        
    # obtener el key correspondiente a la fecha especificada
    time_format = '%d-%m-%Y %H:%M:%S'
    for k in KEYS:
        
        # obtener tiempos de scanning
        start_time, end_time = get_key_times(k)
        
        start_time = datetime.strptime(start_time, time_format)
        end_time = datetime.strptime(end_time, time_format)
        
        # si timestamp se encuentra dentro del intervalo de scan
        if (timestamp > start_time) and (timestamp <= end_time):
            data_key = k
            break
        # si timestamp no se encuentra en ningun intervalo
        if (timestamp < end_time):
            data_key = k
            break
        
    # armar query de descarga ------------------------------------------------
    query = 'https://noaa-goes16.s3.amazonaws.com/' + data_key
    response = requests.get(query)
    
    file_name = data_key.split('/')[-1].split('.')[0]
    nc4_ds = netCDF4.Dataset(file_name, memory = response.content)
        
    return nc4_ds

# -----------------------------------------------------------------------------
# descargar y guardar imágenes satelitales entre un periodo de tiempo
def download_goes16_data_v2(save_path, start_time, end_time,
                            product='ABI-L1b-RadF', mode='M6', channel='C04'):
    """
    -> None
    
    Descarga desde el bucket aws s3 de la noaa todos los datafiles/imágenes
    satelitales durante el periodo limitado por start_time y end_time.
    
    :param str save_path:
        ubicación de la carpeta donde se guardarán los datasets descargados.
    :param str start_time:
        fecha en formato %d-%m-%Y %H:%M desde la que comienza el periodo.
    :param str end_time:
        fecha en formato %d-%m-%Y %H:%M en la que termina el periodo.
    :param str product:
        producto generado por uno de los instrumentos en el goes16.
    :param str mode:
        modo de la operación de escaneo realizada por el goes16.
    :param str channel:
        canal o banda que se desea obtener.
        
    :returns:
        None
    """
    
    print('\n' + '='*80)
    print('Downloading NOAA GOES16 Datasets')
    print('='*80 + '\n')
    sleep(0.25)
        
    # verificar si el periodo entregado es válido -----------------------------
    date_format = '%d-%m-%Y %H:%M'
    
    start_time = validate_date(start_time)
    start_time = datetime.strptime(start_time, date_format)
    
    end_time = validate_date(end_time)
    end_time = datetime.strptime(end_time, date_format)
    
    assert start_time < end_time
    
    total_hours = (end_time - start_time).seconds
    total_hours = round( total_hours/3600.0 )
    
    # verificar si la carpeta entragada existe --------------------------------
    assert os.path.isdir(save_path)
    
    # descarga ----------------------------------------------------------------
    bar = ProgressBar()
    for i in bar( range(total_hours) ):
        # estructurar key_time
        key_time = start_time + timedelta(hours=i)
        
        date_tt = key_time.timetuple()
        
        year = str(date_tt.tm_year)
        yday = str(date_tt.tm_yday)
        hour = str(date_tt.tm_hour)
        
        # ---------------------------------------------------------------------
        # armar data query
        filename_prefix = 'OR_' + product + '-' + mode + channel
        data_prefix = '/'.join([product, year, yday, hour, filename_prefix])
        
        # obtener keys del bucket
        KEYS = get_keys(bucket = 'noaa-goes16', prefix = data_prefix)
        
        # descargar cada uno de los data_keys
        for k in KEYS:
            data_key = k
            
            # armar query de descarga
            query = 'https://noaa-goes16.s3.amazonaws.com/' + data_key
            response = requests.get(query)
            
            # procesar dataset
            file_name = data_key.split('/')[-1].split('.')[0]
            nc4_ds = netCDF4.Dataset(file_name, memory = response.content)
            store = xr.backends.NetCDF4DataStore(nc4_ds)
            
            dataset = xr.open_dataset(store)
            
            # guardar
            new_file_name = os.path.join(save_path, data_key.split('/')[-1])
            dataset.to_netcdf(path=new_file_name, encoding = {'y':{}, 'x':{}})
            
    # retornanr
    return None

# -----------------------------------------------------------------------------
# plotear noaa dataset - netCDF4 file
def plot_goes16_data(nc4_dataset, **kargs):
    """
    -> None
    
    Plota los datos de radiación contenidos en el dataset entregado.
    
    :param str or netCDF4.Dataset nc4_dataset:
        los strings son interpretados como la ubicación del archivo .nc.
        los archivos netCDF4 son abiertos usando xarray.
        
    :returns:
        None
    """
    
    # parsing del dataset -----------------------------------------------------
    if type(nc4_dataset) is str:
        data = xr.open_dataset(nc4_dataset)
        
    elif type(nc4_dataset) is netCDF4._netCDF4.Dataset:
        store = xr.backends.NetCDF4DataStore(nc4_dataset)
        data = xr.open_dataset(store)
        
    # plotear datos -----------------------------------------------------------
    plt.figure(figsize=(80, 80))
    plt.imshow(data.Rad, cmap='gray')
    plt.axis('off')
        
            
            
            
            
            
            
            
        
        
        
        
    
    
    
    
    
    



