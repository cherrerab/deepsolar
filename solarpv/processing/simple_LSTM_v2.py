# -*- coding: utf-8 -*-
#%%############################################################################
############################### LOAD DATA #####################################
###############################################################################
import pandas as pd

from solarpv.database import compact_database
from solarpv.database import select_date_range
from solarpv.database import adjust_timestamps
from solarpv.database import radiance_to_radiation

# -----------------------------------------------------------------------------
# cargar datos de potencia-SMA
sma_15min_path = 'C:\\Cristian\\003. SMA DATASET\\005. 15 MINUTES SYSTEM DATA2\\sma-15min-dataset.pkl'
power_dataset = pd.read_pickle(sma_15min_path)

# compactar a base de 30min
power_dataset = compact_database(power_dataset, 2, use_average=True)
power_dataset = adjust_timestamps(power_dataset, -15*60)

#%%############################################################################
############################# SETUP DATASET ###################################
###############################################################################
from solarpv import ext_irradiation
from solarpv.database import reshape_by_day
from solarpv.database import setup_lstm_dataset

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import pandas as pd
import numpy as np
from math import floor

from datetime import datetime, timedelta

# inicilizar dataset
colnames = ['timestamp', 'power', 'n_day', 'min_day']
dataset = pd.DataFrame(0.0, index=power_dataset.index, columns=colnames)

# asignar columna de timestamp y potencia
dataset['timestamp'] = power_dataset['Timestamp']
dataset['power'] = power_dataset['Sistema']

# -----------------------------------------------------------------------------
# rellenar resto columnas
date_format = '%d-%m-%Y %H:%M'
for i in dataset.index:
    # obtener timestamp del dato
    timestamp = dataset.at[i, 'timestamp']
    timestamp = datetime.strptime(timestamp, date_format)
    
    date_tt = timestamp.timetuple()
    
    # calcular minuto del día y día en el año
    min_day = date_tt.tm_hour*60.0 + date_tt.tm_min
    n_day = date_tt.tm_yday
    
    # asignar al dataset
    dataset.at[i, 'n_day'] = n_day
    dataset.at[i, 'min_day'] = min_day

# ----------------------------------------------------------------------------- 
# setup time windows
n_input = 12
n_output = 6
overlap = 18

# preparar trining set
scaler = MinMaxScaler()
scaler.fit(data)

X, Y = setup_lstm_dataset(dataset, 'power', n_input, n_output, overlap, avoid_cols=['timestamp'])

# obtener training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 217)

#%%############################################################################
############################## LSTM MODEL #####################################
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
model.add(LSTM(units = 64, return_sequences = True, input_shape = (n_input,1)))
#keras.layers.Dropout(rate = 0.1)

# creamos la segunda capa LSTM (con DropOut)
model.add(LSTM(units = 64, return_sequences = True))
#keras.layers.Dropout(rate = 0.1)

# creamos la tercera y cuarta capa (con DropOut)
model.add(Dense(units = 128, activation = 'relu'))
#keras.layers.Dropout(rate = 0.1)
model.add(Dense(units = 128, activation = 'relu'))
#keras.layers.Dropout(rate = 0.1)

# creamos la capa de salida
model.add(Flatten())
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




