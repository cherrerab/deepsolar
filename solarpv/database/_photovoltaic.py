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
# renombrar archivos csv descargados del sunny portal
def rename_csv_files(dir_path, first_date, name_sort=False):
    """
    Renombra los archivos contenidos en el directorio especificado, asignando a
    cada archivo un día en formato %d-%m-%Y. 
    
    :param str dir_path:
        carpeta que contiene los registros de potencia del arreglo fotovoltaico
        descargados del portal de SMA.
    :param str first_date:
        fecha correspondiente al primer archivo contenido en el directorio.
        se supone que cada archivo corresponde a un día y que estos pueden ser
        ordenados por el nombre.
    :param bool name_sort:
        especifíca si los archivos pueden ser ordenados por nombre.
        
    :returns:
        None
    """
    
    # -------------------------------------------------------------------------
    # obtener lista de archivos csv contenidos en el directorio
    files_list = os.listdir(dir_path)
    
    # ordenar los archivos csv
    csv_list = []
    for f in files_list:
        file_name, file_ext = os.path.splitext(f)
        
        # si el archivo no es csv, saltar
        if file_ext != '.csv':
            continue
        
        # si los archivos pueden ser ordenados directamente
        if name_sort:
            csv_list.append( (f, 0) )
        
        # si los archicos contienen un número que los puede ordenar
        else:    
            # obtener número
            _, file_num = file_name.split( '(' )
            file_num, _ = file_num.split( ')' )
            file_num = int(file_num)
            
            csv_list.append( (f, file_num) )
    
    # -------------------------------------------------------------------------
    # ordenar
    if name_sort:
        csv_list.sort()
        
    else:
        csv_list.sort(key=lambda x: x[1])
        
    # -------------------------------------------------------------------------
    # renombrar archivos
    date = datetime.strptime(first_date, '%d-%m-%Y')
    
    for f, _ in csv_list:
        
        # obtener dirección original
        file_path = os.path.join(dir_path, f)
        
        # obtener fecha correspondiente
        date_name = datetime.strftime(date, '%Y-%m-%d') + '.csv'
        new_path = os.path.join(dir_path, date_name)

        # renombrar
        os.rename(file_path, new_path)
        
        # aumentar fecha
        date = date + timedelta(days=1)
        
    return

# -----------------------------------------------------------------------------
# base de datos generación fotovoltaica
def photovoltaic_dataset(dir_path, usecols, all_equipments=False, 
                         keep_negatives=True, adjust_time=0, fix_timestamps=True):
    """
    Lee los archivos que contienen los datos de potencia (kW) del arreglo
    fotovoltaico y construye la base de datos temporal correspondiente.

    :param str dir_path:
        carpeta que contiene los registros de potencia del arreglo fotovoltaico
        descargados del portal de SMA.
        se supone que los archivos al ordenarse a partir del nombre, estos
        quedan ordenados cronológicamente.
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
        file_name, file_ext = os.path.splitext(f)
        
        if file_ext != '.csv':
            continue
        
        csv_list.append( f )
    
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
    for f in csv_list:
        file_name, _ = os.path.splitext(f)
        
        # leer csv
        file_path = os.path.join(dir_path, f)
        db = parse_database(file_path, 'Hoja1', 1, usecols, colnames)
        
        # añadir fecha al timestamp de los datos
        db['Timestamp'] = file_name + ' ' + db['Timestamp'].astype(str)
        
        # añadir a la lista de datos
        files_data.append(db)
    
    db = pd.concat(files_data, axis=0, ignore_index = True)
    
    #--------------------------------------------------------------------------
    # formatear dataFrame
    first_hour = db.at[0, u'Timestamp'].split('/')[0]

    # primer timestamp
    date_format = '%d-%m-%Y %H:%M'
    
    date_start = datetime.strptime(first_hour, '%Y-%m-%d %H:%M')
    
    # obtener base de tiempo del data frame
    t0 = datetime.strptime( db.at[0, u'Timestamp'].split('/')[0], '%Y-%m-%d %H:%M')
    t1 = datetime.strptime( db.at[1, u'Timestamp'].split('/')[0], '%Y-%m-%d %H:%M')
    
    timestep = (t1 - t0).seconds

    # formatear columnas
    bar = ProgressBar()
    for i in bar( db.index ):
        
        # formatear timestamp
        timestamp = date_start + timedelta( seconds = i*timestep )
        new_date = datetime.strftime(timestamp,'%H:%M')

        old_date = db.at[i, u'Timestamp'].split('/')[0]
        old_date = datetime.strptime(old_date, '%Y-%m-%d %H:%M')
        old_date = datetime.strftime(old_date, '%H:%M')
        
        
        # revisar correspondencia de los datos
        if old_date != new_date:
            print('TimeError: desincronización entre timestamp y datos.')
            print(old_date)
            print(timestamp)
            break
        
        db.at[i, u'Timestamp'] = datetime.strftime(timestamp, date_format)
        
        # formatear potencias
        for c in colnames[1:]:
            potencia = db.at[i, c]
            potencia = potencia.replace(' ', '')
            potencia = '0,000' if potencia=='' else potencia
            potencia = float( potencia.replace(',', '.') )
            db.at[i, c] = potencia
        
    if not(fix_timestamps):
        return db
    
    # -------------------------------------------------------------------------
    # añadir timestamps faltantes
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
# base de datos generación fotovoltaica
def photovoltaic_dataset_v2(dir_path, usecols, all_equipments=False, 
                            keep_negatives=True, adjust_time=0, fix_timestamps=True):
    """
    Lee los archivos que contienen los datos de potencia (kW) del arreglo
    fotovoltaico y construye la base de datos temporal correspondiente.

    :param str dir_path:
        carpeta que contiene los registros de potencia del arreglo fotovoltaico
        descargados del portal de SMA.
        se supone que los archivos al ordenarse a partir del nombre, estos
        quedan ordenados cronológicamente.
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
        file_name, file_ext = os.path.splitext(f)
        
        if file_ext != '.csv':
            continue
        
        csv_list.append( f )
    
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
    for f in csv_list:
        
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
        file_name, file_ext = os.path.splitext(f)
        
        if file_ext != '.csv':
            continue
        
        csv_list.append( f )
    
    # ordenar
    csv_list.sort()
    
    #--------------------------------------------------------------------------
    # crear dataframe
 
    # columnas a usar
    colnames = [u'Timestamp', u'Exterior', u'Module']
    
    # leer datos
    files_data = []
    for f in csv_list:
        file_name, _ = os.path.splitext(f)
        
        # leer csv
        file_path = os.path.join(dir_path, f)
        db = parse_database(file_path, 'Hoja1', 1, usecols, colnames)
        
        # añadir fecha al timestamp de los datos
        db['Timestamp'] = file_name + ' ' + db['Timestamp'].astype(str)
        
        # añadir a la lista de datos
        files_data.append(db)
    
    db = pd.concat(files_data, axis=0, ignore_index = True)
    
    #--------------------------------------------------------------------------
    # formatear dataFrame
    first_hour = db.at[0, u'Timestamp'].split('/')[0]

    # primer timestamp
    date_format = '%d-%m-%Y %H:%M'
    
    date_start = datetime.strptime(first_hour, '%Y-%m-%d %H:%M')
    
    # obtener base de tiempo del data frame
    t0 = datetime.strptime( db.at[0, u'Timestamp'].split('/')[0], '%Y-%m-%d %H:%M')
    t1 = datetime.strptime( db.at[1, u'Timestamp'].split('/')[0], '%Y-%m-%d %H:%M')
    
    timestep = (t1 - t0).seconds
    
    # formatear columnas
    bar = ProgressBar()
    for i in bar( db.index ):
        
        # formatear timestamp
        timestamp = date_start + timedelta( seconds = i*timestep )
        new_date = datetime.strftime(timestamp,'%H:%M')

        old_date = db.at[i, u'Timestamp'].split('/')[0]
        old_date = datetime.strptime(old_date, '%Y-%m-%d %H:%M')
        old_date = datetime.strftime(old_date, '%H:%M')
        
        
        # revisar correspondencia de los datos
        if old_date != new_date:
            print('TimeError: desincronización entre timestamp y datos.')
            print(old_date)
            print(timestamp)
            break
        
        db.at[i, u'Timestamp'] = datetime.strftime(timestamp, date_format)
        
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

#------------------------------------------------------------------------------