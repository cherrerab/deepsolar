# -*- coding: utf-8 -*-
"""
Proyecto Beauchef PV Forecasting

Cristian Herrera

v1.0 - update, Aug 01 2019
"""

import numpy as np

import keras
from keras.models import Sequential

from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten

from keras.utils import np_utils
from keras.utils import plot_model

from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import GridSearchCV

# -----------------------------------------------------------------------------
# crear arquitectura LSTM-Dense
def init_lstm_model(input_shape, output_shape, lstm_layers, lstm_units,
                    dense_layers_1, dense_units_1, dense_activation_1,
                    dense_layers_2, dense_units_2, dense_activation_2,
                    dense_layers_3, dense_units_3, dense_activation_3,
                    optimizer='adam', dropout_rate=0.2):
    """
    -> keras.model
    
    método para la inicialización de un modelo LSTM.
    
    :param tuple input_shape:
        shape del set de datos de entrada.
    :param int lstm_layers:
        cantidad de capas LSTM a agregar al modelo.
    :param int lstm_units:
        cantidad de nodos/unidades en las capas LSTM.
        
    :param int dense_layers_1:
        cantidad de capas Dense a agregar al modelo previo al Flatten.
    :param int dense_units_1:
        cantidad de nodos/unidades en las capas Dense previas al Flatten.
    :param str dense_activation_1:
        tipo de función de activación a usar en las capas.
        
    :param int dense_layers_2:
        cantidad de capas Dense a agregar al modelo previo al Flatten.
    :param int dense_units_2:
        cantidad de nodos/unidades en las capas Dense previas al Flatten.
    :param str dense_activation_2:
        tipo de función de activación a usar en las capas.
        
    :param int dense_layers_3:
        cantidad de capas Dense a agregar al modelo posterior al Flatten.
    :param int dense_units_3:
        cantidad de nodos/unidades en las capas Dense posteriores al Flatten.
    :param str dense_activation_3:
        tipo de función de activación a usar en las capas Dense posteriores al Flatten.
    
    :param int batch_size:
        tamaños del batch de training en cada epoch.
    :param int epochs:
        cantidad de epochs que durará el training del modelo.
    :param str optimizer:
        optimizador a utilizar en el training.
    :param float dropout:
        porcentaje de unidades a anular posterior a cada etapa .
    
    :returns:
        keras.model con la arquitectura especificada.
    """
    
    # inicialización ----------------------------------------------------------
    model = Sequential()
    
    # capas LSTM --------------------------------------------------------------
    # agregar la capa de ínput LSTM
    model.add( LSTM(units=lstm_units, return_sequences=True, input_shape=input_shape) )

    # agregar el resto de capas
    for i in np.arange(lstm_layers - 1):
        model.add( LSTM(units=lstm_units, return_sequences=True) )
        
    # agregar DropOut
    model.add( Dropout(rate=dropout_rate) )
        
    # capas Dense_1 -----------------------------------------------------------
    # agregar capas Dense
    for i in np.arange(dense_layers_1):
        model.add( Dense(units=dense_units_1, activation=dense_activation_1) )
        
    # agregar DropOut
    model.add( Dropout(rate=dropout_rate) )
    
    # capas Dense_2 -----------------------------------------------------------
    # agregar capas Dense
    for i in np.arange(dense_layers_2):
        model.add( Dense(units=dense_units_2, activation=dense_activation_2) )
        
    # agregar DropOut
    model.add( Dropout(rate=dropout_rate) )
    
    # agregar Flatten ---------------------------------------------------------
    model.add( Flatten() )
    
    # agregar Dense 3 ---------------------------------------------------------
    # agregar capas Dense
    for i in np.arange(dense_layers_3):
        model.add( Dense(units=dense_units_3, activation=dense_activation_3) )
        
    # agregar DropOut
    model.add( Dropout(rate=dropout_rate) )
    
    # agregar output ----------------------------------------------------------
    model.add( Dense(units=output_shape, activation = 'linear') )
    
    # configuramos el modelo de optimizacion a utilizar
    model.compile(optimizer=optimizer, loss = 'mse', metrics = ['mae'])
        
    return model

# -----------------------------------------------------------------------------
# optimizar modelo medainte grid search
def optimize_model_structure(build_fn, param_grid, X_train, X_test, Y_train, Y_test):
    
    # setear método de inicialización
    model = KerasRegressor(build_fn=build_fn)
    
    # inicializar grid search
    grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=3)
    grid_result = grid.fit(X_train, Y_train)
    
    print("Best score: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    
    # retornar mejor modelo
    return grid_result.best_estimator_ 