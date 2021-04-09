# -*- coding: utf-8 -*-
"""
Proyecto Beauchef PV Forecasting

Cristian Herrera

v1.0 - update, Aug 01 2019
"""

import pandas as pd
pd.options.mode.chained_assignment = None

from datetime import datetime, timedelta

from deepsolar._tools import validate_date, get_timestep
from deepsolar.database._tools import parse_database

from progressbar import ProgressBar


# -----------------------------------------------------------------------------
# base de datos estación solarimétrica
def solarimetric_dataset(path, sheet_name, header, usecols, keep_negatives=True,
                         adjust_time=0, fix_timestamps=True):
    """
    -> pandas.DataFrame
    
    Filtra y prepara la base de datos para posteriormente procesar los
    regristros de irradiancia.
    
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
    print('\n' + '-'*80)
    print('parsing solarimetric station data')
    print('-'*80 + '\n')
    
    # abrir archivo de base de datos ------------------------------------------
    colnames = [u'Timestamp', u'Temperature', u'Global', u'Diffuse',
                u'Direct']
    
    db = parse_database(path, sheet_name, header, usecols, colnames)
    assert db is not None
    
    # filtrar y formatear -----------------------------------------------------
    date_format = '%d-%m-%Y %H:%M'
    
    remove_index = []
    bar = ProgressBar()
    for i in bar( db.index ):
        # formatear columnas criticas
        try:
            # comprobar si fue emitida post iyear
            fecha = validate_date(db.at[i, u'Timestamp'])
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
    data_count = db.shape[0]
    
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
    
    # print missing data
    print('missing data: ' + str(db.shape[0] - data_count))
  
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
    date_format = '%d-%m-%Y %H:%M'
    secs = get_timestep(db['Timestamp'], date_format)/3600
    
    db['Global'] = [ rad*secs for rad in db['Global'].values]
    db['Diffuse'] = [ rad*secs for rad in db['Diffuse'].values]
    db['Direct'] = [ rad*secs for rad in db['Direct'].values]
    
    return db

#------------------------------------------------------------------------------