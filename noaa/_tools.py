# -*- coding: utf-8 -*-
"""
Proyecto Beauchef PV Forecasting

Cristian Herrera

v1.0 - update, Aug 01 2019
"""

import xarray as xr
import netCDF4
import numpy as np

from deepsolar._tools import validate_date
from deepsolar.noaa._api import get_key_times

from datetime import datetime

import os

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
def get_bound_indexes(nc4_dataset, lat, lon, size):
    """
    -> Tuple
    
    Retorna los índices correspondientes al área delimitada por las coordenadas
    especificadas en el dataset.
    
    :param str or netCDF4.Dataset nc4_dataset:
        los strings son interpretados como la ubicación del archivo .nc.
        los archivos netCDF4 son abiertos usando xarray.
    :param float lat:
        latitud central del área.
    :param float lon:
        longitud central del área.
    :param int size:
        tamaño de la imagen final que se desea.
        
    :returns:
        None
    """
    # parsing del dataset -----------------------------------------------------
    if type(nc4_dataset) is str:
        goes_nc = netCDF4.Dataset(nc4_dataset, mode='r')
        
    elif type(nc4_dataset) is netCDF4._netCDF4.Dataset:
        goes_nc = nc4_dataset

    # obtener información del GOES-16 y constantes ----------------------------
    proj_info = goes_nc.variables['goes_imager_projection']
    
    lon_origin = proj_info.longitude_of_projection_origin
    H = proj_info.perspective_point_height + proj_info.semi_major_axis
    r_eq = proj_info.semi_major_axis
    r_pol = proj_info.semi_minor_axis
    
    # data meshgrid de angulos radianes de scanning
    lat_rad_1d = goes_nc.variables['x'][:]
    lon_rad_1d = goes_nc.variables['y'][:]

    
    lat_rad,lon_rad = np.meshgrid(lat_rad_1d,lon_rad_1d)
    
    
    # calculo de lat/lon a partir de los angulos radianes

    lambda_0 = (lon_origin*np.pi)/180.0
    
    a_var = np.power(np.sin(lat_rad),2.0) + (np.power(np.cos(lat_rad),2.0)*(np.power(np.cos(lon_rad),2.0)+(((r_eq*r_eq)/(r_pol*r_pol))*np.power(np.sin(lon_rad),2.0))))
    b_var = -2.0*H*np.cos(lat_rad)*np.cos(lon_rad)
    c_var = (H**2.0)-(r_eq**2.0)
    
    r_s = (-1.0*b_var - np.sqrt((b_var**2)-(4.0*a_var*c_var)))/(2.0*a_var)
    
    s_x = r_s*np.cos(lat_rad)*np.cos(lon_rad)
    s_y = - r_s*np.sin(lat_rad)
    s_z = r_s*np.cos(lat_rad)*np.sin(lon_rad)
    
    lats = (180.0/np.pi)*(np.arctan(((r_eq*r_eq)/(r_pol*r_pol))*((s_z/np.sqrt(((H-s_x)*(H-s_x))+(s_y*s_y))))))
    lons = (lambda_0 - np.arctan(s_y/(H-s_x)))*(180.0/np.pi)

    # encontrar indices de (lat, lon)
    
    lat_near = ( np.abs(lats - lat) < 0.01 )
    lon_near = ( np.abs(lons - lon) < 0.01 )
    
    index = np.where( np.bitwise_and(lat_near, lon_near) )
    
    lat_index = index[0][0]
    lon_index = index[1][0]
    
    # obtener indices de la ventana
    lat_start = int( lat_index - size/2 )
    lat_end = int( lat_index + size/2 )
    
    lon_start = int( lon_index - size/2 )
    lon_end = int( lon_index + size/2 )
    
    # cerrar archivo
    goes_nc.close()
    del goes_nc
     
    # retornar límites encontrados
    return (lat_start, lat_end, lon_start, lon_end)

# get satellite window
def bound_goes16_data(nc4_dataset, bound_indexes):
    """
    -> numpy.array
    
    Retorna los datos de radiación en el área limitada por las coordenadas
    especificadas.
    
    :param str or netCDF4.Dataset nc4_dataset:
        los strings son interpretados como la ubicación del archivo .nc.
        los archivos netCDF4 son abiertos usando xarray.
    :param tuple(float) bound_indexes:
        indices en el dataset que corresponden al área que se desea delimitar.
        (lat_start, lat_end, lon_start, lon_end)
        
    :returns:
        None
    """
    
    # parsing del dataset -----------------------------------------------------
    if type(nc4_dataset) is str:
        goes_nc = xr.open_dataset(nc4_dataset)
        
    elif type(nc4_dataset) is netCDF4._netCDF4.Dataset:
        store = xr.backends.NetCDF4DataStore(nc4_dataset)
        goes_nc = xr.open_dataset(store)
    
    # obtener datos de radiación
    rad_data = goes_nc.variables['Rad'][:]
    
    # apicar limites especificados
    lat_start, lat_end, lon_start, lon_end = bound_indexes
    rad_data = rad_data.values[ lat_start:lat_end, lon_start:lon_end  ]
    
    # cerrar archivo
    goes_nc.close()
    del goes_nc
    
    # retornar array
    return np.array(rad_data)

# -----------------------------------------------------------------------------
# plotear noaa dataset - netCDF4 file
def plot_goes16_data(nc4_dataset, gamma_correction=False, **kargs):
    """
    -> None
    
    Plota los datos de radiación contenidos en el dataset entregado.
    
    :param str or netCDF4.Dataset nc4_dataset:
        los strings son interpretados como la ubicación del archivo .nc.
        los archivos netCDF4 son abiertos usando xarray.   
    :param bool gama_correction:
        si se desea aplicar gamma correction sobre la imagen.
        
    :returns:
        None
    """
    
    # parsing del dataset -----------------------------------------------------
    if type(nc4_dataset) is str:
        goes_nc = xr.open_dataset(nc4_dataset)
        
    elif type(nc4_dataset) is netCDF4._netCDF4.Dataset:
        store = xr.backends.NetCDF4DataStore(nc4_dataset)
        goes_nc = xr.open_dataset(store)
    
    # obtener datos de radiación
    rad_data = goes_nc.variables['Rad'][:]
    
    # aplicar gamma correction
    if gamma_correction:
        rad_data = (rad_data * np.pi * 0.3) / 441.868715
        # Make sure all data is in the valid data range
        rad_data = np.maximum(rad_data, 0.0)
        rad_data = np.minimum(rad_data, 1.0)
        rad_data = np.sqrt(rad_data)
    
    # plotear datos -----------------------------------------------------------
    plt.figure(figsize=(80, 80))
    plt.imshow(rad_data, cmap='gray')
    plt.axis('off')
    
    # cerrar archivo
    goes_nc.close()
    del goes_nc
    
    