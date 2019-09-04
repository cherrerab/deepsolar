# -*- coding: utf-8 -*-
"""
Proyecto Beauchef PV Forecasting

Cristian Herrera

v1.0 - update, Aug 01 2019
"""

import xarray as xr
import netCDF4

from solarpv import validate_date
from solarpv.noaa._api import get_key_times

from datetime import datetime, timedelta

import os
from time import sleep

from progressbar import ProgressBar

import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# leer datos satelitales guardados en el disco
def read_goes16_data(dir_path, timestamp, product='ABI-L1b-RadF', mode='M6',
                     channel='C04'):
    """
    -> netCDF4.Dataset
    
    Busca en el directorio entregado el archivo .nc más cercano al timestamp
    especificado.
    
    :param str dir_path:
        directorio en el que se encuentran los archivos .nc (netCDF4).
    :param str timestamp:
        fecha en formato %d-%m-%Y %H:%M de la imagen que se desea retornar.
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
    
    # obtener lista de archivos en el directorio ------------------------------
    nc_files = os.listdir(dir_path)
    
    time_format = '%d-%m-%Y %H:%M:%S'
    for f in nc_files:
        key, ext = os.path.splitext(f)
        
        # verificar extension del archivo
        if ext != '.nc':
            continue
        
        # obtener timepo de scanning
        start_time, end_time = get_key_times(key)
        
        start_time = datetime.strptime(start_time, time_format)
        end_time = datetime.strptime(end_time, time_format)
        
        # si timestamp se encuentra dentro del intervalo de scan
        if (timestamp > start_time) and (timestamp < end_time):
            data_key = f
            break
        # si ya no es posible encontrar un intervalo que lo contenga
        if (timestamp < start_time):
            data_key = f
            break
    
    # retornar archivo netCDF
    file_path = os.path.join(dir_path, data_key)
    nc4_ds = netCDF4.Dataset(file_path, mode='r')
    
    return nc4_ds

# -----------------------------------------------------------------------------
# get satellite window
def bound_goes16_data(nc4_dataset, lat, lon, radius):
    """
    -> numpy.array
    
    Retorna los datos de radiación en el área limitada por las coordenadas
    especificadas.
    
    :param str or netCDF4.Dataset nc4_dataset:
        los strings son interpretados como la ubicación del archivo .nc.
        los archivos netCDF4 son abiertos usando xarray.
    :param tuple(float) lower_coord:
        limites inferiores de latitud y longitud.
    :param tuple(float) upper_coord:
        limites superiores de latitud y longitud.
        
    :returns:
        None
    """
    # parsing del dataset -----------------------------------------------------
    if type(nc4_dataset) is str:
        data = xr.open_dataset(nc4_dataset)
        
    elif type(nc4_dataset) is netCDF4._netCDF4.Dataset:
        store = xr.backends.NetCDF4DataStore(nc4_dataset)
        data = xr.open_dataset(store)
        
    # obtener limites en el arreglo -------------------------------------------
    lats = data.variables['latitude'][:] 
    lons = data.variables['longitude'][:]
    
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
    
    