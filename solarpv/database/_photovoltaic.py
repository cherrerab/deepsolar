# -*- coding: utf-8 -*-
"""
Proyecto Beauchef PV Forecasting

Cristian Herrera

v1.0 - update, Aug 01 2019
"""

import pandas as pd
pd.options.mode.chained_assignment = None

from datetime import datetime, timedelta

from solarpv.database._tools import parse_database

from progressbar import ProgressBar
import os

# -----------------------------------------------------------------------------
# base de datos generación fotovoltaica
def photovoltaic_dataset(dir_path, usecols, all_equipments=False, 
                         keep_negatives=True, adjust_time=0, fix_timestamps=True):
    """
    Lee los archivos que contienen los datos de potencia (kW) del arreglo
    fotovoltaico y construye la base de datos temporal correspondiente.

    :param DataFrame dir_path:
        carpeta que contiene los registros de potencia del arreglo fotovoltaico
        descargados del portal de SMA.
    :param str usecols:
        contiene las columnas que han de utilizarse para procesar la base de
        datos. 
    :param bool all_equipments: (default, False)
        si el archivo entregado contiene las potencias de los equipos en el
        sistema.
    :param bool keep_negatives: (default, True)
        si mantener valores negativos de irradiacion.
    :param str adjust_time:
        permite corregir en segundos el timestamp asociado a cada dato.
    :param bool fix_timestamps: (default, True)
        si corregir la serie de timestamps en el dataset.
        
    :returns:
        DataFrame con los datos procesados.
    """
    print('\n' + '='*80)
    print('Parsing Photovoltaic Plant Data')
    print('='*80 + '\n')
    
    
    # -------------------------------------------------------------------------
    # obtener lista de archivos csv contenidos en el directorio
    files_list = os.listdir(dir_path)
    
    # ordenar los archivos csv
    csv_list = []
    for f in files_list:
        file_name, _ = os.path.splitext(f)
        
        # obtener número
        _, file_num = file_name.split( '(' )
        file_num, _ = file_num.split( ')' )
        file_num = int(file_num)
        
        csv_list.append( (file_num, f) )
    
    # ordenar
    csv_list.sort()
    
    #--------------------------------------------------------------------------
    # crear dataframe
 
    # columnas a usar
    if all_equipments:
        colnames = [u'Timestamp', u'Sistema', u'SB 2500HF-30 273',
                    u'SB 2500HF-30 997', u'SMC 5000A 434', u'SMC 5000A 493']
        
    else:
        colnames = [u'Timestamp', u'Sistema']
    
    # leer datos
    files_data = []
    for _, f in csv_list:
        
        # leer csv
        file_path = os.path.join(dir_path, f)
        db = parse_database(file_path, 'Hoja1', 1, usecols, colnames)
        
        # añadir a la lista de datos
        files_data.append(db)
    
    db = pd.concat(files_data, axis=0, ignore_index = True)
    
    #--------------------------------------------------------------------------
    # formatear dataFrame
    date_start = os.path.basename(dir_path)
    first_hour, first_day = db.at[0, u'Timestamp'].split('/')

    assert int(first_day) == datetime.strptime(date_start, '%d-%m-%Y').day
    
    # primer timestamp
    date_format = '%d-%m-%Y %H:%M'
    
    date_start = ' '.join([date_start, first_hour])
    date_start = datetime.strptime(date_start, date_format)
    
    # obtener base de tiempo del data frame
    timestep = (  datetime.strptime(db.at[1,u'Timestamp'], '%H:%M/ %d')
                  - datetime.strptime(db.at[0,u'Timestamp'], '%H:%M/ %d') )
    timestep = timestep.seconds
    
    # formatear columnas
    bar = ProgressBar()
    for i in bar( db.index ):
        
        # formatear timestamp
        timestamp = date_start + timedelta( seconds = i*timestep )
        new_date = datetime.strftime(timestamp,'%H:%M/ %d')
        
        old_date = datetime.strptime(db.at[i, u'Timestamp'],'%H:%M/ %d')
        old_date = datetime.strftime(old_date,'%H:%M/ %d')
        
        # revisar correspondencia de los datos
        if old_date != new_date:
            print('TimeError: desincronización entre timestamp y datos.')
            print(old_date)
            break
        
        db.at[i, u'Timestamp'] = datetime.strftime(timestamp, date_format)
        
        # formatear potencias
        for c in colnames[1:]:
            potencia = db.at[i, c]
            potencia = '0,000' if potencia=='' else potencia
            potencia = float( potencia.replace(',', '.') )
            db.at[i, c] = potencia
        
    if not(fix_timestamps):
        return db
    
    # -------------------------------------------------------------------------
    # añadir timestamps faltantes
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

#------------------------------------------------------------------------------
# base de datos generación fotovoltaica
def temperature_dataset(dir_path, usecols, keep_negatives=True, adjust_time=0,
                        fix_timestamps=True):
    """
    Lee los archivos que contienen los datos de temperatura (°C) del arreglo
    fotovoltaico y construye la base de datos temporal correspondiente.

    :param DataFrame dir_path:
        carpeta que contiene los registros de potencia del arreglo fotovoltaico
        descargados del portal de SMA.
    :param str usecols:
        contiene las columnas que han de utilizarse para procesar la base de
        datos. 
    :param bool keep_negatives: (default, True)
        si mantener valores negativos de irradiacion.
    :param str adjust_time:
        permite corregir en segundos el timestamp asociado a cada dato.
    :param bool fix_timestamps: (default, True)
        si corregir la serie de timestamps en el dataset.
        
    :returns:
        DataFrame con los datos procesados.
    """
    print('\n' + '='*80)
    print('Parsing Photovoltaic Plant Data')
    print('='*80 + '\n')
    
    # -------------------------------------------------------------------------
    # obtener lista de archivos csv contenidos en el directorio
    files_list = os.listdir(dir_path)
    
    # ordenar los archivos csv
    csv_list = []
    for f in files_list:
        file_name, _ = os.path.splitext(f)
        
        # obtener número
        _, file_num = file_name.split( '(' )
        file_num, _ = file_num.split( ')' )
        file_num = int(file_num)
        
        csv_list.append( (file_num, f) )
    
    # ordenar
    csv_list.sort()
    
    #--------------------------------------------------------------------------
    # crear dataframe
 
    # columnas a usar
    colnames = [u'Exterior', u'Module']
    
    # leer datos
    files_data = []
    for _, f in csv_list:
        
        # leer csv
        file_path = os.path.join(dir_path, f)
        db = parse_database(file_path, 'Hoja1', 1, usecols, colnames)
        
        # añadir a la lista de datos
        files_data.append(db)
    
    db = pd.concat(files_data, axis=0, ignore_index = True)
    
    #--------------------------------------------------------------------------
    # formatear dataFrame
    date_start = os.path.basename(dir_path)
    first_hour, first_day = db.at[0, u'Timestamp'].split('/')

    assert int(first_day) == datetime.strptime(date_start, '%d-%m-%Y').day
    
    # primer timestamp
    date_format = '%d-%m-%Y %H:%M'
    
    date_start = ' '.join([date_start, first_hour])
    date_start = datetime.strptime(date_start, date_format)
    
    # obtener base de tiempo del data frame
    timestep = (  datetime.strptime(db.at[1,u'Timestamp'], '%H:%M/ %d')
                  - datetime.strptime(db.at[0,u'Timestamp'], '%H:%M/ %d') )
    timestep = timestep.seconds
    
    # formatear columnas
    bar = ProgressBar()
    for i in bar( db.index ):
        
        # formatear timestamp
        timestamp = date_start + timedelta( seconds = i*timestep )
        new_date = datetime.strftime(timestamp,'%H:%M/ %d')
        
        old_date = datetime.strptime(db.at[i, u'Timestamp'],'%H:%M/ %d')
        old_date = datetime.strftime(old_date,'%H:%M/ %d')
        
        # revisar correspondencia de los datos
        if old_date != new_date:
            print('TimeError: desincronización entre timestamp y datos.')
            print(old_date)
            break
        
        db.at[i, u'Timestamp'] = datetime.strftime(timestamp, date_format)
        
        # formatear temperaturas
        for c in colnames[1:]:
            temp = db.at[i, c]
            temp = '0,000' if temp=='' else temp
            temp = float( temp.replace(',', '.') )
            db.at[i, c] = temp
        
    if not(fix_timestamps):
        return db
    
    # -------------------------------------------------------------------------
    # añadir timestamps faltantes
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

#------------------------------------------------------------------------------