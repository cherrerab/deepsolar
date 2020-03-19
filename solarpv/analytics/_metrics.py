# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 11:44:43 2020

@author: Cristian
"""

# -*- coding: utf-8 -*-
"""
Proyecto Beauchef PV Forecasting

Cristian Herrera

v1.0 - update, Aug 01 2019
"""


import matplotlib.pyplot as plt
import random
import os
import inspect

from datetime import datetime, timedelta

import pandas as pd

from solarpv._tools import validate_date, get_date_index, ext_irradiation, get_timestep, solar_incidence
from solarpv.database import read_csv_file

import numpy as np
from math import (pi, cos, sin, tan, acos)
from scipy.stats import skew, kurtosis


#------------------------------------------------------------------------------
# mean absolute error
def mean_squared_error(Y_true, Y_pred):
    """
    -> float
    
    calcula el error cuadrado medio de la diferencia entre Y_true e Y_pred.
    
    :param np.array Y_true:
        serie de valores reales.
    :param np.array Y_pred:
        serie de valores estimados.
    
    :returns:
        error cuadrado medio de la diferencia.
    """
    
    # calcular error
    error = Y_pred.flatten() - Y_true.flatten()
    
    # calcular error cuadrado medio
    mse = np.sqrt(np.mean(np.power(error, 2)))
    
    return mse

#------------------------------------------------------------------------------
# mean absolute error
def mean_absolute_error(Y_true, Y_pred):
    """
    -> float
    
    calcula el error absoluto medio de la diferencia entre Y_true e Y_pred.
    
    :param np.array Y_true:
        serie de valores reales.
    :param np.array Y_pred:
        serie de valores estimados.
    
    :returns:
        error absoluto medio de la diferencia.
    """
    
    # calcular error
    error = Y_pred.flatten() - Y_true.flatten()
    
    # calcular error absoluto medio
    mae = np.mean( np.abs(error) )
    
    return mae

#------------------------------------------------------------------------------
# mean bias error
def mean_bias_error(Y_true, Y_pred):
    """
    -> float
    
    calcula el error medio de la diferencia entre Y_true e Y_pred.
    
    :param np.array Y_true:
        serie de valores reales.
    :param np.array Y_pred:
        serie de valores estimados.
    
    :returns:
        error medio de la diferencia.
    """
    
    # calcular error
    error = Y_pred.flatten() - Y_true.flatten()
    
    # calcular error absoluto medio
    mae = np.mean( error )
    
    return mae

#------------------------------------------------------------------------------
# skew error
def skew_error(Y_true, Y_pred, **kargs):
    """
    -> float
    
    calcula el skewness de la diferencia entre Y_true e Y_pred.
    
    :param np.array Y_true:
        serie de valores reales.
    :param np.array Y_pred:
        serie de valores estimados.
    
    :returns:
        skewness de la diferencia.
    """
    
    # calcular error
    error = Y_pred.flatten() - Y_true.flatten()
    
    # calcular kurtosis
    skw = skew(error, **kargs)
    
    return skw

#------------------------------------------------------------------------------
# kurtosis
def kurtosis_error(Y_true, Y_pred, **kargs):
    """
    -> float
    
    calcula el kurtosis de la diferencia entre Y_true e Y_pred.
    
    :param np.array Y_true:
        serie de valores reales.
    :param np.array Y_pred:
        serie de valores estimados.
    
    :returns:
        kurtosis de la diferencia.
    """
    
    # calcular error
    error = Y_pred.flatten() - Y_true.flatten()
    
    # calcular kurtosis
    kts = kurtosis(error, **kargs)
    
    return kts
        
#------------------------------------------------------------------------------
# forecast skill
def forecast_skill(Y_true, Y_pred, Y_base, **kargs):
    """
    -> float
    
    calcula el forecast skill (%) entre los valores Y_pred e Y_base.
    
    :param np.array Y_true:
        serie de valores reales.
    :param np.array Y_pred:
        serie de valores estimados.
    :param np.array Y_pred:
        serie de valores estimados mediante un modelo base.
    
    :returns:
        forecast skill de los valores estimados.
    """
    
    # calcular rmse Y_pred
    rmse = np.sqrt( mean_squared_error(Y_true, Y_pred) )
    
    # calcular rmse Y_base
    rmse_b = np.sqrt( mean_squared_error(Y_true, Y_base) )
    
    # calcular forecast skill
    fs = 1.0 - rmse/rmse_b
    
    return fs