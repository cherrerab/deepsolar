# -*- coding: utf-8 -*-

###############################################################################
############################# PROCESS DATA ####################################
###############################################################################
import os

import pandas as pd

from solarpv.analytics import cluster_daily_radiation
from solarpv.database import compact_database
from solarpv.database import adjust_timestamps

from datetime import datetime

# importar datasets - 30 min
sma_15min_path = 'C:\\Cristian\\003. SMA DATASET\\005. 15 MINUTES SYSTEM DATA2\\sma-15min-dataset.pkl'
sma_15min_dataset = pd.read_pickle(sma_15min_path)
sma_15min_dataset = compact_database(sma_15min_dataset, 2, use_average=True)
sma_15min_dataset = adjust_timestamps(sma_15min_dataset, -15*60)

#%%############################################################################
################################ ANALYSIS #####################################
###############################################################################
from datetime import timedelta
from solarpv.analytics import plot_1D_radiation_data
from solarpv.analytics import plot_2D_radiation_data

# checkear resultado con día soleado
plot_1D_radiation_data(sma_15min_dataset, 'Sistema', '04-11-2018', '05-11-2018', multiply_factor=80)

# plotear dataset
plot_2D_radiation_data(sma_15min_dataset, unit='kW', colname='Sistema', initial_date='28-08-2018',final_date='25-09-2019')

#%%############################################################################
############################# SETUP DATASET ###################################
###############################################################################
import numpy as np
from solarpv.database import setup_lstm_dataset
from solarpv.database import lstm_standard_scaling

dataset = sma_15min_dataset[['Timestamp','Sistema']]

# time window size
n_input = 12
n_output = 6
overlap = 18

# realizar dataset split
X_train, X_test, Y_train, Y_test = setup_lstm_dataset(dataset, 'Sistema', 8, 2, n_input, n_output, overlap, ['Timestamp'])

# -----------------------------------------------------------------------------
# normalización
X_train, X_test, std_scaler = lstm_standard_scaling(X_train, X_test, Y_train, Y_test)

# reshape dataset
n_feature = X_train.shape[2]

X_train = np.reshape(X_train, (-1, n_input*n_feature))
X_test = np.reshape(X_test, (-1, n_input*n_feature))

# normalizar output
feature_min = std_scaler[0, 0]
feature_max = std_scaler[0, 1]

Y_train = (Y_train - feature_min)/(feature_max-feature_min)
Y_test = (Y_test - feature_min)/(feature_max-feature_min)

#%%############################################################################
############################### ANN MODEL #####################################
###############################################################################
import keras
from keras.layers import LSTM
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils import np_utils
from keras.utils import plot_model
from keras.layers import Flatten

import matplotlib.pyplot as plt

# inicializamos la ANN
model = Sequential()

# creamos la primera capa de input
model.add(Dense(units = 256, activation = 'relu', input_shape = (n_input*n_feature,)))
keras.layers.Dropout(rate = 0.2)

# creamos la segunda capa LSTM (con DropOut)
model.add(Dense(units = 256, activation = 'relu'))
keras.layers.Dropout(rate = 0.2)

# creamos la tercera y cuarta capa (con DropOut)
model.add(Dense(units = 128, activation = 'relu'))
keras.layers.Dropout(rate = 0.2)

model.add(Dense(units = 128, activation = 'relu'))
keras.layers.Dropout(rate = 0.2)

model.add(Dense(units = 64, activation = 'relu'))
keras.layers.Dropout(rate = 0.2)

# creamos la capa de salida
model.add(Dense(units = n_output, activation = 'linear'))

# configuramos el modelo de optimizacion a utilizar
model.compile(optimizer = 'adam', loss = 'mse', metrics = ['mae'])

# entrenamos el modelo
model_history = model.fit(X_train, Y_train, batch_size = 128, epochs = 256, validation_data = (X_test, Y_test))

#%%############################################################################
####################### TRAINING EVALUATION ###################################
###############################################################################

# visualizamos la evolucion de la funcion de perdida
test_mse, test_mae = model.evaluate(X_test, Y_test, batch_size = 30)

plt.figure()
plt.plot(model_history.history['loss'])
plt.plot(model_history.history['val_loss'])
plt.title('ANN model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['train', 'validation'], loc='upper right')
plt.show()

#%%############################################################################
########################### LOAD SOLAR DATA ###################################
###############################################################################
from solarpv.database import compact_database
from solarpv.database import select_date_range
from solarpv.database import radiance_to_radiation

# cargar datos solarimetricos
solar_1min_path = 'C:\\Cristian\\001. SOLARIMETRIC DATA\\solarimetric-1min-dataset.pkl'
solar_1min_dataset = pd.read_pickle(solar_1min_path)
solar_1min_dataset = select_date_range(solar_1min_dataset, '27-08-2018 00:00', '25-09-2019 00:00')
solar_1min_dataset = radiance_to_radiation(solar_1min_dataset)

#%%############################################################################
########################## MODEL EVALUATION ###################################
###############################################################################

from solarpv.database import compact_database
from solarpv.database import select_date_range
from solarpv.database import radiance_to_radiation

from solarpv.analytics import cluster_evaluation

eval_data = dataset.copy()

for i in np.arange(std_scaler.shape[0]):
    feature_min = std_scaler[i, 0]
    feature_max = std_scaler[i, 1]
    eval_data.iloc[:,i+1] = (eval_data.iloc[:,i+1] - feature_min)/(feature_max-feature_min)


# evaluar modelo de forecasting
cluster_metrics = cluster_evaluation(solar_1min_dataset, eval_data, model)