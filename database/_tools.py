# -*- coding: utf-8 -*-
"""
Proyecto Beauchef PV Forecasting

Cristian Herrera

v1.0 - update, Aug 01 2019
"""

import pandas as pd
pd.options.mode.chained_assignment = None

from datetime import datetime, timedelta

from deepsolar._tools import (validate_date, get_date_index, get_timestep, ext_irradiance)

import numpy as np

from math import ceil, floor
from progressbar import ProgressBar
import os

from time import sleep

#------------------------------------------------------------------------------
# abrir archivo csv
def read_csv_file(csv_path, try_header='', return_encoding=False, **kargs):
    """
    -> DataFrame or [DataFrame, str]
    
    
    Método más robusto para la lectura de archivos csv. Itera hasta encontrar
    el enconding correcto para su manipulación.
    
    :param str csv_path:
        ubicacion del archivo csv que se desea leer.
    :param str try_header: (default, '')
        nombre del header/cabecera con la que probar el encoding.
    :param bool return_encoding: (default, False)
        si es True la función retorna además el encoding utilizado para leer
        el archivo.
    :param bool keep_default_na: (default, True)
        si incluir o no los valores NaN (celdas vacias) del archivo en el
        DataFrame.
        
    :returns:
        (return_encoding=False) DataFrame del archivo csv
        (return_encoding=True) [DataFrame, str]
    
    :raises:
        IOError: si no logra abrir el archivo.
    """
    csv_name = os.path.basename(csv_path)
    
    # encodings con los que se probará la apertura
    encodings = iter(['utf_8_sig', 'latin_1', 'utf_8'])
    
    # estado de lectura
    correct_read = False
    
    
    while not correct_read:
        try:
            encode = next(encodings)
            # crear DataFrame a partir del csv
            data = pd.read_csv(csv_path, sep = None, engine = 'python',
                                      encoding = encode, **kargs)
            # probar el header de prueba
            if (try_header != '') and (try_header in data.columns):
                correct_read = True
            
            # si no se quiere utilizar header de prueba
            if try_header == '':
                # si se llega hasta este punto se puede retornar
                correct_read = True
        
        except IOError:
            print('IOError: El archivo ' + csv_name + ' no se encuentra o no ' +
                  'pudo ser abierto en el directorio entregado:')
            print( os.path.dirname(csv_path) )
            return 
        
        except UnicodeDecodeError:
            continue
        
        except StopIteration:
            print('No se logro encontrar el encoding adecuado para leer el'
                  ' archivo correctamente.')
            return
    # si se desea recuperar el encoder utilizado
    if return_encoding:
        return [data, encode]
    
    # si solo se desea el DataFrame
    return data

# -----------------------------------------------------------------------------
# generar lista de columnas de excel
def columns_id(usecols):
    """
    -> list
    
    Genera la lista de id de columnas ('A', 'B', 'C', ... 'AA', 'AB', 'AC',...)
    necesaria para contener todas columnas en usecols.
    
    :param list-like usecols:
        lista de id de columnas a contener (['A', 'F', 'AA', 'BZ']).
        
    :returns:
        lista de id de columnas
    """
    alphabet = [''] + list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
    
    idcols = []
    i = 0
    val = False
    # mientras la lista no contenga todas las columnas
    while not val:
        # agregar ids
        for j in alphabet[1:]: 
            idcols.append( ''.join( (alphabet[i], j) ) )
        i += 1
        
        val = True
        # por cada una de las cols a contener
        for c in usecols:
            # verificar si esta contenida en la lista
            val = val and (c in idcols)
            
    return idcols

# -----------------------------------------------------------------------------
# abrir archivo de base de datos
def parse_database(path, sheet_name, header, usecols, colnames):
    """
    -> DataFrame
    
    Realiza el parsing del archivo que contiene la base de datos con que se
    desea trabajar independiente de si es csv o xlsx. Permite cambiar el nombre
    de las columnas y reordenarlas.
    
    :param str path:
        ubicacion del csv o excel que contiene la base de datos de la empresa.
    :param str sheet_name:
        hoja en el archivo excel en el que se encuentra la base de datos.
        (None si es un csv)
    :param int header:
        fila en el libro de excel o en el csv  que contiene las cabeceras de
        las columnas
    :param str usecols:
        contiene las columnas que han de utilizarse para procesar la base de
        datos.
    
    :returns: DataFrame de la base de datos
    """
    
    # validar extension de base de datos --------------------------------------
    database_name = os.path.basename(path)
    extension = os.path.splitext(database_name)[1]
    formats = ['.xls','.xlsx', '.xlsm', '.xlsb', '.csv']
    try:
        assert (extension in formats)
    except AssertionError:
        print ( 'IOError: La extension del archivo no permite su procesamiento\n'
               'Extensiones validas: .xls, .xlsx, .xlsm, .xlsb, .csv')
        return None
    
    try:
        # remover columnas comodin
        colnames = [colnames[i] for i in range(len(usecols)) if usecols[i] != '*']
        usecols = [i for i in usecols if i != '*']
             
        # obtener indices de las columnas
        column_letters = columns_id(usecols)
        
        column_index = []
        for c in usecols:
            i = column_letters.index(c)
            column_index.append(i)
            
        # parsing del archivo -------------------------------------------------
        # si el archivo corresponde a un csv
        if extension == '.csv':   
            db = read_csv_file(path, keep_default_na=False, index_col=False,
                               header=header-1, skip_blank_lines=True, usecols=column_index)
            
        # si el archivo corresponde a un libro excel
        else:
            db = pd.read_excel(path, sheet_name=sheet_name, header=header-1,
                               index_col=None, usecols=column_index, keep_default_na=False)
            
        # ordenar columnas ----------------------------------------------------
        old_order = [c for _,c in sorted(zip(column_index, usecols))]
        set_order = list(usecols)
        old_cols = list(db.columns)
        for c in set_order:
            i = old_order.index(c)
            j = set_order.index(c)
            db.rename(columns={old_cols[i]:colnames[j]}, inplace=True)
            
    except IndexError:
        print('ColumnError: las columnas ingresadas no permiten el procesamiento.')
        return None

    return db[colnames]

# -----------------------------------------------------------------------------
# seleccionar datos contenidos en un rango temporal
def select_date_range(database, date_start, date_end):
    """
    -> DataFrame
    
    retorna el sub-dataset contenido dentro del margen temporal especificado.
    
    :param DataFrame database:
        base de datos que contiene el registro temporal a procesar.
    :param str date_start:
        fecha inicial del margen temporal.
    :param str date_end:
        fecha final del margen temporal.
        
    :returns:
        DataFrame del sub-dataset
    """
    
    db = database.copy()
    db.index = pd.DatetimeIndex( db[u'Timestamp'].values, dayfirst=True )
    
    # obtener rango temporal
    date_format = '%d-%m-%Y %H:%M'
    date_start = datetime.strptime( validate_date(date_start), date_format )
    date_end = datetime.strptime( validate_date(date_end), date_format )
    
    mask = (db.index >= date_start) & (db.index < date_end)
    
    # modificar dataset
    db = db.loc[mask]
    db = db.reset_index()
    db.drop('index',axis=1, inplace=True)
    
    return db

# -----------------------------------------------------------------------------
# ajustar timestamps de un dataset
def adjust_timestamps(database, delta_time):
    """
    -> DataFrame
    
    Ajusta los timestamps del dataset especificado utilizando un delta de tiempo
    en segundos delta_time.
    
    :param DataFrame database:
        base de datos que contiene el registro temporal a procesar.
    :param int delta_time:
        tiempo en segundos en que se desea ajustar la serie temporal.
        
    :returns:
        el DataFrame con los timestamps corregidos.
    """
    
    db = database.copy()
    
    date_format = '%d-%m-%Y %H:%M'
    
    # ajustar timestamps
    for i in db.index:
        # obtener timestamp
        timestamp = db.at[i, u'Timestamp']
        
        # sumar delta_time
        timestamp = datetime.strptime(timestamp, date_format)
        timestamp = timestamp + timedelta(seconds=delta_time)
        
        # reasignar nuevo timestamp
        db.at[i, u'Timestamp'] = datetime.strftime(timestamp, date_format)
        
    return db

# -----------------------------------------------------------------------------
# ordenar base de datos radiacion por día
def reshape_by_day(database, colname, initial_date, final_date):
    """
    -> DataFrame
    
    Reordena el registro temporal de la base de datos separando cada día en una
    columna, siguiendo la base de tiempo indicada.
    
    :param DataFrame database:
        base de datos que contiene el registro temporal a procesar.
    :param str colname:
        nombre de la columna que contiene los datos de interés.
    :param str initial_date:
        fecha y hora desde la cual considerar los datos.
    :param str final_date:
        fecha y hora hasta la cual considerar los datos (sin incluir).
        
          
    :returns: DataFrame de la base de datos procesada
    """
    print('\n' + '-'*80)
    print('reshaping data time series')
    print('-'*80 + '\n')
    sleep(0.25)
    
    # inicializar nueva base de datos -----------------------------------------
    date_format = '%d-%m-%Y %H:%M'
    
    initial_date = datetime.strptime(validate_date(initial_date), date_format)
    final_date = datetime.strptime(validate_date(final_date), date_format)
    
    assert final_date > initial_date
    
    # columnas
    delta_days = final_date - initial_date
    num_days = delta_days.days
    dates = [initial_date + timedelta(days=x) for x in range(num_days)]
    cols = [datetime.strftime(d, '%d-%m-%Y') for d in dates]
    
    
    # filas
    timestep = get_timestep(database['Timestamp'], date_format)
    num_hours = int( 24*(3600/timestep) )
    
    initial_hour = datetime.strptime('00:00','%H:%M')
    hours = [initial_hour + timedelta(seconds=timestep*x) for x in range(num_hours)]
    rows = [datetime.strftime(h, '%H:%M') for h in hours]
    
    # obtener diferencia en segundos
    first_hour = datetime.strptime(database.at[0,'Timestamp'], date_format)
    first_hour = datetime.strftime(first_hour, '%H:%M')
    first_hour = datetime.strptime(first_hour, '%H:%M')
    
    closest_hour = min(hours, key=lambda h: abs( first_hour - h ))
    delta_sec = (closest_hour - first_hour).seconds
    delta_sec = (delta_sec - 24*3600) if delta_sec > timestep else delta_sec
    
    # DataFrame
    df = pd.DataFrame(0.0, index = rows, columns= cols)
    
    # colocar datos de radiación en el nuevo DataFrame
    bar = ProgressBar()
    for i in bar( database.index ):
        try:
            timestamp = datetime.strptime( database.at[i, 'Timestamp'], date_format)
            
            # aplicar corrección de desfase
            timestamp += timedelta(seconds=delta_sec)
            
            if (timestamp >= initial_date) and (timestamp < final_date):
                date = datetime.strftime(timestamp, '%d-%m-%Y')
                hour = datetime.strftime(timestamp, '%H:%M')

                df.at[hour, date] = database.at[i, colname]
            
        except (ValueError, TypeError):
            continue
    
    # retornar database    
    return df

#------------------------------------------------------------------------------
# comprimir database / cambiar base time
def compact_database(database, factor, use_average=False):
    """
    -> DataFrame
    
    Comprime la base de tiempo del set de datos sumando o ponderando los datos
    correspondientes al nuevo intervalo definido.
    
    :param DataFrame database:
        base de datos que contiene el registro temporal a procesar.
    :param int factor:
        factor por el cual se comprimirá la base de datos.
    :param bool use_average:
        permite decidir si sumar o ponderar los datos al comprimir el dataframe.
    """
    
    print('\n' + '-'*80)
    print('compacting data time series')
    print('-'*80 + '\n')
    sleep(0.25)
    
    # -------------------------------------------------------------------------
    # obtener base de tiempo de dataset
    date_format = '%d-%m-%Y %H:%M'
    timestep = get_timestep(database['Timestamp'], date_format)
    
    # -------------------------------------------------------------------------
    # compactar dataframe
    colnames = database.columns.values
    df = pd.DataFrame(0.0, index=range( ceil(len(database.index)/factor) ),
                      columns=colnames)
    df = df.astype({u'Timestamp': str})

    # obtener fecha inicial
    date_start = datetime.strptime(database.at[0, u'Timestamp'], date_format)
    next_date = date_start + timedelta(seconds = factor*timestep)
    
    df.at[0, u'Timestamp'] = datetime.strftime(date_start, date_format)
    
    # peso ponderador
    weight = 1.0/factor if use_average else 1.0
    
    bar = ProgressBar()
    for i in bar( database.index ):
        # obtener fecha
        date = datetime.strptime(database.at[i, u'Timestamp'], date_format)
        
        # si se sale del intervalo actual
        if date >= next_date:
            # añadir timestamp al nuevo dataframe
            df.at[i//factor, 'Timestamp'] = datetime.strftime(next_date, date_format)
            next_date = next_date + timedelta(seconds = factor*timestep)
        
        # sumar valores a fila correspondiente
        for c in colnames[1:]:
                df.at[i//factor, c] += database.at[i, c]*weight
    
    return df
        
#------------------------------------------------------------------------------
# alinear dataset con radiación extraterrestre
def align_radiation(database, clear_sky_days, **kargs ):
    """
    -> DataFrame
    
    Ajusta los timestamps en la base de datos tal que el perfil de radiación
    global se alinie con el de la radiación extraterrestre.
    
    :param DataFrame database:
        base de datos que contiene el registro temporal a procesar.
    :param list(str) clear_sky_days:
        lista de días en el dataset con una alto índice de claridad, en formato
        %d-%m-%Y.
    
    :returns:
        DataFrame con los timestamps modificados. 
    """
    
    date_format = '%d-%m-%Y %H:%M'
    timestamps = database['Timestamp'].values
    
    diff = []
    
    for day in clear_sky_days:
        
        # obtener ventana de datos correspondiente ----------------------------
        start_index = get_date_index(timestamps, day)
        
        next_day = datetime.strptime( validate_date(day), date_format )
        next_day = datetime.strftime( next_day + timedelta(days=1), date_format)
        stop_index = get_date_index( timestamps, next_day )
        
        day_times = timestamps[start_index:stop_index]
        
        # alinear máximos -----------------------------------------------------
        ext_rad = [ext_irradiance(t, **kargs) for t in day_times]
        ext_index = ext_rad.index( max(ext_rad) )
        try:
            rad_data = list(database['Global'].values[start_index:stop_index])
        except KeyError:
            rad_data = list(database['Potencia'].values[start_index:stop_index])
            
        rad_index = rad_data.index( max(rad_data) )
        
        # calcular diferencia entre máximos
        diff.append( ext_index - rad_index )
    
    # calular ajuste en segundos ----------------------------------------------
    timestep = get_timestep(database['Timestamp'], date_format)
    delay = round(sum(diff)/len(diff))*timestep
    print(delay)
    # modificar timestamps
    new_timestamps = [datetime.strftime( datetime.strptime(t, date_format)
                     + timedelta(seconds=delay), date_format ) for t in timestamps]
    
    # retornar base de datos modificada
    db = database
    db['Timestamp'] = new_timestamps
    return db

#------------------------------------------------------------------------------
# corregir daylight saving time
def correct_daylight_saving(database, start_date, end_date, positive=True):
    """
    -> DataFrame
    
    Ajusta los timestamps en la base de datos corrigiendo el periodo de cambio
    de hora por Daylight Saving Time entre date_start y date_end.
    
    :param DataFrame database:
        base de datos que contiene el registro temporal a procesar.
    :param str date_start:
        fecha y hora en la que comienza el DST, %d-%m-%Y %H:%M.
    :param str date_start:
        fecha y hora en la que termina el DST, %d-%m-%Y %H:%M. 
    
    :returns:
        DataFrame con los timestamps modificados. 
    """
    
    print('\n' + '-'*80)
    print('correcting daylight saving time')
    print('-'*80 + '\n')
    sleep(0.25)
    
    # parsing de fechas -------------------------------------------------------
    date_format = '%d-%m-%Y %H:%M'
    start_date = datetime.strptime( validate_date(start_date), date_format )
    end_date = datetime.strptime( validate_date(end_date), date_format )
    
    # procesamiento -----------------------------------------------------------
    db = database.copy()
    
    # obtner timestep
    timestep = get_timestep(database['Timestamp'], date_format)
    
    delta = (3600.0/timestep)
    delta = delta if positive else -delta
    
    # procesar
    bar = ProgressBar()
    for i in bar( db.index ):
        # obtener timestamp
        timestamp = datetime.strptime(db.at[i, 'Timestamp'], date_format)
        
        # si está en el periodo del DST
        if (timestamp >= start_date) and (timestamp <= end_date):
            try:
                db.iloc[i, 1:] = database.iloc[int(i - delta), 1:]
            except IndexError:
                continue
            
            
    # retornar base de datos
    return db
        
#------------------------------------------------------------------------------  