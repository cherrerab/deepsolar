# -*- coding: utf-8 -*-
"""
Proyecto Beauchef PV Forecasting

Cristian Herrera

v1.0 - update, Aug 01 2019
"""
import numpy as np

from datetime import datetime, timedelta
from math import (pi, cos, sin, tan, acos)
#------------------------------------------------------------------------------
# validar formato de fecha
def validate_date(date_text):
    """
    -> str
    
    Retorna la fecha entregada de acuerdo al formato dd-mm-yyyy HH:MM.
    
    :param str date_text:
        string que contiene la fecha a validar
    
    :returns:
        str de la fecha entregada en el formato dd-mm-yyyy HH:MM.
    """
    # caso borde
    if date_text == '':
        return ''
    
    date_text = str(date_text)
        
    # formatos de fecha a probar
    date_formats = iter(['%d-%m-%Y %H:%M','%d-%m-%Y %H:%M:%S','%Y-%m-%d %H:%M',
                         '%Y-%m-%d %H:%M:%S', '%d-%m-%Y', '%Y-%m-%d'])
    
    correct_read = False
    while not correct_read:
        try:
            # cambiar el formato
            date_format = next(date_formats)
            # probar si fecha tiene formato YYYY-MM-DD
            date = datetime.strptime(date_text, date_format)
            correct_read = True
        
        # si el formato no funciona
        except ValueError:
            continue
            
        except StopIteration:
            return ''
    
    return datetime.strftime(date, '%d-%m-%Y %H:%M')

#------------------------------------------------------------------------------
# obtener fecha 
def get_date_index(timestamps, search_date, nearest=None):
    """
    -> int
    
    Retorna el index en que se encuentra la fecha 'search_date' en el DataFrame
    de fechas 'timestamps'.
    
    :param Data Series timestamps:
        serie que contiene la lista de fechas (se asume ordenada).
    :param str search_date:
        fecha que se desea buscar en timestamps.
    :param bool nearest ('lower' or 'upper'):
        si entregar el indice de la fecha anterior más cercana en caso de que
        no se encuentre 'search_date' dentro de 'timestamp'
        
        
    :returns:
        index en que 'search_date' se encuentra en 'timestamps'.
    """
    
    try:
        timestamps = timestamps.values
    except AttributeError:
        timestamps
        
    # busqueda exacta
    date_format = '%d-%m-%Y %H:%M'
    search_date = validate_date(search_date)
    
    time_index = np.where( timestamps==search_date)[0]
    if time_index.any():
        return time_index[0]
    
    # busqueda inexacta
    if nearest != None:
        search_date = datetime.strptime( search_date, date_format )
    
        first_date = datetime.strptime(timestamps[0], date_format)
        last_date = datetime.strptime(timestamps[-1], date_format)
        
        # revisar limites
        if search_date < first_date:
            return 0
        elif search_date > last_date:
            return len(timestamps)-1
        
        # revisar lista
        for i in range( len(timestamps) ):
            idate = datetime.strptime(timestamps[i], date_format)
            
            # más cercano
            if (idate >= search_date):
                time_index = i if nearest=='upper' else i-1
                return time_index
    
    # en caso de no encontrar el timestamp
    return None

# obtener timestamp de un registro temporal
def get_timestep(timestamps, date_format):
    """
    -> int
    
    Retorna la cantidad de segundos que hay en cada intervalo de la serie
    temporal.
    
    :param array-like timestamps:
        serie temporal de timestamps.
    :param str date_format:
        formato de los timestamps en la serie.
        
    :returns:
        timestep en segundos.
    """
    
    # checkear tipo de serie
    try:
        timestamps = timestamps.values
    except AttributeError:
        timestamps
     
    # obtener timestep
    timestep = (  datetime.strptime(timestamps[1], date_format)
                - datetime.strptime(timestamps[0], date_format))
    timestep = timestep.seconds
    
    return float(timestep)
    
#------------------------------------------------------------------------------
# obtener irradiación extraterrestre
def ext_irradiance(timestamp, lat=-33.45775, lon=70.66466111, Bs=0.0, Zs=0.0):
    """
    -> float
    
    Calcula a partir del timestamp y la posición geográfica entregada, la
    irradiancia extraterrestre incidente en el plano especificado en W/m2.
    
    :param str timestamp:
        string que contiene la hora y fecha a utilizar (UTC).
    :param float lat:
        latitud del punto geográfico.
    :param float lon:
        longitud del punto geográfico en grados oeste [0,360).
    :param float Bs: (default, 0.0)
        inclinación de la superficie.
        ángulo de inclinación respecto al plano horizontal de la superficie 
        sobre la cual calcular la irradiancia (grados).
    :param float Zs: (default, 0.0)
        azimut de la superficie.
        ángulo entre la proyección de la normal de la superficie en el plano
        horizontal y la dirección hacia el ecuador (grados).
        
    :returns:
        float del valor de irradiancia extraterrestre en W/m2.
    """
    
    # corregir latitud y longitud
    lat = lat*(pi/180.0)
    lon = lon*(pi/180.0)
    
    # tiempo
    date_format = '%d-%m-%Y %H:%M'
    date = datetime.strptime(timestamp, date_format)
    date_tt = date.timetuple()
    
    # dia del año
    min_day = date_tt.tm_hour*60.0 + date_tt.tm_min
    n_day = date_tt.tm_yday + min_day/(24*60.0)
    
    # extraterrestrial irradiance W/m2 on the plane normal to the radiation
    # Spencer (1971)
    G_sc = 1367.0
    B = (n_day - 1)*(2*pi/365.0)
    G_on = G_sc*(1.000110 + 0.034221*cos(B) + 0.001280*sin(B) +
                 0.000719*cos(2*B) + 0.000077*sin(2*B))
    
    # solar time - Spencer (1971)
    E = 229.2*(0.000075 + 0.001868*cos(B) - 0.032077*sin(B) - 0.014615*cos(2*B)
        - 0.04089*sin(2*B))
    
    solar_min = min_day + 4*(0 - lon*(180.0/pi)) + E
    solar_hour = solar_min/60.0
    
    # ángulo solar (w)
    w = 15*(solar_hour - 12.0)*(pi/180.0)
    
    # declinacion - Spencer (1971)
    delta = (0.006918 - 0.399912*cos(B) + 0.070257*sin(B) - 0.006758*cos(2*B)
            + 0.000907*sin(2*B) - 0.002697*cos(3*B) + 0.00148*sin(3*B))
    
#    # ángulo de alba (w_ss)
#    w_ss = -abs( acos(-tan(lat)*tan(delta)) )
#    
#    # si el sol no ha salido la irradiancia es cero
#    if abs(w) >= abs(w_ss):
#        return 0.0
    
    # ángulo de incidencia (theta) sobre la superficie
    # ángulo entre la radiación del sol y la normal a la superficie
    
    cos_theta = (sin(delta)*sin(lat)*cos(Bs)
                 - sin(delta)*cos(lat)*sin(Bs)*cos(Zs)
                 + cos(delta)*cos(lat)*cos(Bs)*cos(w)
                 + cos(delta)*sin(lat)*sin(Bs)*cos(Zs)*cos(w)
                 + cos(delta)*sin(Bs)*sin(Zs)*sin(w))
    
    # retornar irradiancia solar, W/m2
    if cos_theta < 0.0:
        return 0.0
    
    return G_on*cos_theta

#------------------------------------------------------------------------------
# obtener irradiación extraterrestre
def ext_irradiation(timestamp, secs, step='backward',
                    lat=-33.45775, lon=70.66466111, Bs=0.0, Zs=0.0):
    """
    -> float
    
    Calcula a partir del timestamp y la posición geográfica entregada, la
    irradiación extraterrestre incidente en el plano especificado en Wh/m2 
    durante el tiempo especificado.
    
    :param str timestamp:
        string que contiene la hora y fecha del comienzo del periodo (UTC).
    :param int secs:
        cantidad de segundos que dura el periodo a calcular.
    :param str step:
        si el intervalo comienza (forward) o termina (backward) en el timestamp.
    :param float lat:
        latitud del punto geográfico.
    :param float lon:
        longitud del punto geográfico en grados oeste [0,360).
    :param float Bs: (default, 0.0)
        inclinación de la superficie.
        ángulo de inclinación respecto al plano horizontal de la superficie 
        sobre la cual calcular la irradiancia (grados).
    :param float Zs: (default, 0.0)
        azimut de la superficie.
        ángulo entre la proyección de la normal de la superficie en el plano
        horizontal y la dirección hacia el ecuador (grados).
        
    :returns:
        float del valor de irradiación extraterrestre en Wh/m2.
    """
    
    # corregir latitud y longitud
    lat = lat*(pi/180.0)
    lon = lon*(pi/180.0)
    
    # tiempo
    date_format = '%d-%m-%Y %H:%M'
    date = datetime.strptime(timestamp, date_format)
    
    date_tt = date.timetuple()
    
    # dia del año
    min_day = date_tt.tm_hour*60.0 + date_tt.tm_min
    n_day = date_tt.tm_yday + min_day/(24*60.0)
    
    # extraterrestrial irradiance W/m2 on the plane normal to the radiation
    # Spencer (1971)
    G_sc = 1367.0
    B = (n_day - 1)*(2*pi/365.0)
    G_on = G_sc*(1.000110 + 0.034221*cos(B) + 0.001280*sin(B) +
                 0.000719*cos(2*B) + 0.000077*sin(2*B))
    
    # solar time - Spencer (1971)
    E = 229.2*(0.000075 + 0.001868*cos(B) - 0.032077*sin(B) - 0.014615*cos(2*B)
        - 0.04089*sin(2*B))
    
    solar_min = min_day + 4*(0 - lon*(180.0/pi)) + E
    
    if step=='forward':
        solar_hour_1 = solar_min/60.0
        solar_hour_2 = (solar_min + secs/60.0)/60.0
    elif step=='backward':
        solar_hour_1 = (solar_min - secs/60.0)/60.0
        solar_hour_2 = solar_min/60.0
        
    
    # ángulo solar (w)
    w1 = 15*(solar_hour_1 - 12.0)*(pi/180.0)
    w2 = 15*(solar_hour_2 - 12.0)*(pi/180.0)
    
    # declinacion - Spencer (1971)
    delta = (0.006918 - 0.399912*cos(B) + 0.070257*sin(B) - 0.006758*cos(2*B)
            + 0.000907*sin(2*B) - 0.002697*cos(3*B) + 0.00148*sin(3*B))
    
#    # ángulo de alba (w_ss)
#    w_ss = -abs( acos(-tan(lat)*tan(delta)) )
#    
#    # ajustar periodo a limites donde existe sol
#    if abs(w1) >= abs(w_ss):
#        w1 = abs(w_ss) if w1 >= abs(w_ss) else -abs(w_ss)
#        
#    if abs(w2) > abs(w_ss):
#        w2 = abs(w_ss) if w2 >= abs(w_ss) else -abs(w_ss)
    
    
    factor = 43200.0/pi
    
    # ángulo de incidencia (theta) sobre la superficie
    # ángulo entre la radiación del sol y la normal a la superficie
    
    cos_theta = (sin(delta)*sin(lat)*cos(Bs)*secs
                 - sin(delta)*cos(lat)*sin(Bs)*cos(Zs)*secs
                 + cos(delta)*cos(lat)*cos(Bs)*(sin(w2) - sin(w1))*factor
                 + cos(delta)*sin(lat)*sin(Bs)*cos(Zs)*(sin(w2) - sin(w1))*factor
                 - cos(delta)*sin(Bs)*sin(Zs)*(cos(w2) - cos(w1))*factor)
    
    # retornar irradicion solar, Wh/m2
    if (cos_theta < 0.0):
        return 0.0
    
    if abs(w1) == abs(w2):
        return 0.0
    
    return G_on*cos_theta/3600.0

#------------------------------------------------------------------------------
def solar_incidence(timestamp, lat=-33.45775, lon=70.66466111, Bs=0.0, Zs=0.0):
    """
    -> float
    
    Calcula el cos(theta) correspondiente al ángulo de incidencia del sol sobre
    la superficie.
    
    :param str timestamp:
        string que contiene la hora y fecha a utilizar (UTC).
    :param float lat:
        latitud del punto geográfico.
    :param float lon:
        longitud del punto geográfico en grados oeste [0,360).
    :param float Bs: (default, 0.0)
        inclinación de la superficie.
        ángulo de inclinación respecto al plano horizontal de la superficie 
        sobre la cual calcular la irradiancia (grados).
    :param float Zs: (default, 0.0)
        azimut de la superficie.
        ángulo entre la proyección de la normal de la superficie en el plano
        horizontal y la dirección hacia el ecuador (grados).
        
    :returns:
        float del valor del coseno del ángule de incidencia.
        * si el valor es negativo, se retorna 0.
    """
    
    # corregir latitud y longitud
    lat = lat*(pi/180.0)
    lon = lon*(pi/180.0)
    
    # tiempo
    date_format = '%d-%m-%Y %H:%M'
    date = datetime.strptime(timestamp, date_format)
    date_tt = date.timetuple()
    
    # dia del año
    min_day = date_tt.tm_hour*60.0 + date_tt.tm_min
    n_day = date_tt.tm_yday + min_day/(24*60.0)
    
    # solar time - Spencer (1971)
    B = (n_day - 1)*(2*pi/365.0)
    E = 229.2*(0.000075 + 0.001868*cos(B) - 0.032077*sin(B) - 0.014615*cos(2*B)
        - 0.04089*sin(2*B))
    
    solar_min = min_day + 4*(0 - lon*(180.0/pi)) + E
    solar_hour = solar_min/60.0
    
    # ángulo solar (w)
    w = 15*(solar_hour - 12.0)*(pi/180.0)
    
    # declinacion - Spencer (1971)
    delta = (0.006918 - 0.399912*cos(B) + 0.070257*sin(B) - 0.006758*cos(2*B)
            + 0.000907*sin(2*B) - 0.002697*cos(3*B) + 0.00148*sin(3*B))
    
#    # ángulo de alba (w_ss)
#    w_ss = -abs( acos(-tan(lat)*tan(delta)) )
#    
#    # si el sol no ha salido la irradiancia es cero
#    if abs(w) >= abs(w_ss):
#        return 0.0
    
    # corregir angulos de la superfice
    Bs = Bs*pi/180
    Zs = Zs*pi/180
    
    # ángulo de incidencia (theta) sobre la superficie
    # ángulo entre la radiación del sol y la normal a la superficie
    
    cos_theta = (sin(delta)*sin(lat)*cos(Bs)
                 - sin(delta)*cos(lat)*sin(Bs)*cos(Zs)
                 + cos(delta)*cos(lat)*cos(Bs)*cos(w)
                 + cos(delta)*sin(lat)*sin(Bs)*cos(Zs)*cos(w)
                 + cos(delta)*sin(Bs)*sin(Zs)*sin(w))
    
    # retornar irradiancia solar, W/m2
    if cos_theta < 0.0:
        return 0.0
    
    return cos_theta