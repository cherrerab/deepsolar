# -*- coding: utf-8 -*-
"""
Proyecto Beauchef PV Forecasting

Cristian Herrera

v1.0 - update, Aug 01 2019
"""

import pandas as pd
from pandas.api.types import is_numeric_dtype
pd.options.mode.chained_assignment = None

from datetime import datetime, timedelta

from solarpv import (validate_date, get_date_index, ext_irradiance)

from math import ceil
from progressbar import ProgressBar
import os
import numpy as np

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
# base de datos estación solarimétrica
def generate_solarimetric_database(path, sheet_name, header, usecols,
                                   keep_negatives=True, adjust_time=0,
                                   fix_timestamps=True):
    """
    -> pandas.DataFrame
    
    Filtra y prepara la base de datos para posteriormente procesar los
    regristros de irradiancia.
    Crea un archivo csv con las columnas:
       { 'Timestamp', 'Temperature', 'Global', 'Diffuse', 'Direct' }
    
    :param str path:
        ubicacion del csv o excel que contiene la base de datos de la estación.
    :param str sheet_name:
        nombre de la hoja donde se encuentran los datos en el archivo.
    :param int header:
        fila en el libro de excel o en el csv  que contiene las cabeceras de
        las columnas.
    :param str usecols:
        contiene las columnas que han de utilizarse para procesar la base de
        datos.
    :param bool keep_negatives: (default, True)
        si mantener valores negativos de irradiacion.
    :param str adjust_time:
        permite corregir en segundos el timestamp asociado a cada dato.
    :param bool fix_timestamps:
        si corregir la serie de timestamps en el dataset.
        
          
    :returns: DataFrame de la base de datos
    """
    print('\n' + '='*80)
    print('Parsing Solarimetric Station Data')
    print('='*80 + '\n')
    
    # abrir archivo de base de datos ------------------------------------------
    colnames = [u'Timestamp', u'Temperature', u'Global', u'Diffuse',
                u'Direct']
    
    db = parse_database(path, sheet_name, header, usecols, colnames)
    assert db is not None
    
    # filtrar y formatear -----------------------------------------------------
    
    remove_index = []
    bar = ProgressBar()
    for i in bar( db.index ):
        # formatear columnas criticas
        try:
            # comprobar si fue emitida post iyear
            fecha = validate_date(db.at[i, u'Timestamp'])
            date_format = '%d-%m-%Y %H:%M'
            fecha = datetime.strptime(fecha, date_format)
            
            # formatear timestamp
            timestamp = fecha + timedelta(seconds=adjust_time)
            db.at[i, u'Timestamp'] = datetime.strftime(timestamp, date_format)
            
            # formatear temperatura
            db.at[i, u'Temperature'] = float(db.at[i, u'Temperature'])
            
            # formatear irradiancia global
            value = float(db.at[i, u'Global'])
            db.at[i, u'Global'] = value if keep_negatives else max(0.0, value)
            
            # formatear irradiancia difusa
            value = float(db.at[i, u'Diffuse'])
            db.at[i, u'Diffuse'] = value if keep_negatives else max(0.0, value)

            # formatear irradiancia directa
            value = float(db.at[i, u'Direct'])
            db.at[i, u'Direct'] = value if keep_negatives else max(0.0, value)
                        
        except (ValueError, TypeError):
            remove_index.append(i)
            print('MissingData: Wrong value format in '+db.at[i, u'Timestamp'])
            continue
        
    # remover filas filtradas            
    db.drop(index = remove_index, inplace=True)
    
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
# base de datos generación fotovoltaica
def generate_photovoltaic_database(dir_path, usecols, keep_negatives=True,
                                   adjust_time=0, fix_timestamps=True):
    """
    Lee los archivos que contienen los datos de potencia (kW) del arreglo
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
    :param bool fix_timestamps:
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
    colnames = [u'Timestamp', u'Potencia']
    
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
        
        # formatear potencia
        potencia = db.at[i, u'Potencia']
        potencia = '0,000' if potencia=='' else potencia
        potencia = float( potencia.replace(',', '.') )
        db.at[i, u'Potencia'] = potencia
        
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
        
    
# -----------------------------------------------------------------------------
# transformar base de datos de radiancia a radiación
def radiance_to_radiation(database):
    """
    Transforma la base de datos de radiancia (W/m2) a de radiación (Wh/m2).
    
    :param DataFrame database:
        base de datos que contiene el registro de irradiancia a procesar.
        
    :returns:
        DataFrame con los datos procesados.
    """
    
    db = database.copy()
    # obtener timestep en el dataset
    timestamps = db['Timestamp'].values
    
    date_format = '%d-%m-%Y %H:%M'
    timestep = (  datetime.strptime(timestamps[1], date_format)
                - datetime.strptime(timestamps[0], date_format))
    secs = timestep.seconds/3600
    
    db['Global'] = [ rad*secs for rad in db['Global'].values]
    db['Diffuse'] = [ rad*secs for rad in db['Diffuse'].values]
    db['Direct'] = [ rad*secs for rad in db['Direct'].values]
    
    return db

# -----------------------------------------------------------------------------
# ordenar base de datos radiacion por día
def reshape_radiation(database, colname, initial_date, final_date):
    """
    -> DataFrame
    
    Prepara la base de datos ordenando el registro de datos de radiación
    separando cada día en una columna, siguiendo la base de tiempo indicada.
    
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
    print('\n' + '='*80)
    print('Reshaping Data')
    print('='*80 + '\n')
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
    timestep = ( datetime.strptime(database.at[1,'Timestamp'], date_format) 
                -datetime.strptime(database.at[0,'Timestamp'], date_format) )
    
    timestep = timestep.seconds
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
            
            if (timestamp >= initial_date) and (timestamp <= final_date):
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
    
    print('\n' + '='*80)
    print('Compacting Data Time Series')
    print('='*80 + '\n')
    sleep(0.25)
    
    # -------------------------------------------------------------------------
    # obtener base de tiempo de dataset
    date_format = '%d-%m-%Y %H:%M'
    timestep = (  datetime.strptime(database.at[1, u'Timestamp'], date_format)
                - datetime.strptime(database.at[0, u'Timestamp'], date_format))
    timestep = timestep.seconds
    
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
    timestep = (datetime.strptime(timestamps[1], date_format)
                - datetime.strptime(timestamps[0], date_format))
    delay = round(sum(diff)/len(diff))*timestep.seconds
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
def correct_daylight_saving(database, start_date, end_date):
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
    
    print('\n' + '='*80)
    print('Correcting Daylight Saving Time')
    print('='*80 + '\n')
    sleep(0.25)
    
    # parsing de fechas -------------------------------------------------------
    date_format = '%d-%m-%Y %H:%M'
    start_date = datetime.strptime( validate_date(start_date), date_format )
    end_date = datetime.strptime( validate_date(end_date), date_format )
    
    # procesamiento -----------------------------------------------------------
    db = database.copy()
    
    # obtner timestep
    timestep = (  datetime.strptime(database.at[1, u'Timestamp'], date_format)
                - datetime.strptime(database.at[0, u'Timestamp'], date_format))
    timestep = timestep.seconds
    
    delta = (3600.0/timestep)
    
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
        
    
    
    
       
       
    




