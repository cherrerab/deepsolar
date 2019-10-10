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

db = sma_15min_dataset

# arreglar timestamps
db = db.reset_index()
db.drop('index',axis=1, inplace=True)

date_format = '%d-%m-%Y %H:%M'

time_freq = str(3600/2.0) + 'S'
first_date = datetime.strptime( db['Timestamp'].iloc[0], date_format)
last_date = datetime.strptime( db['Timestamp'].iloc[-1], date_format)

idx = pd.date_range(first_date, last_date, freq=time_freq)

db.index = pd.DatetimeIndex( db[u'Timestamp'].values, dayfirst=True )
db = db.reindex( idx, fill_value=0.0 )
db[u'Timestamp'] = idx.strftime(date_format)

# resetear index a enteros
db = db.reset_index()
db.drop('index',axis=1, inplace=True)

# guardar dataset
#dir_path = 'C:\\Users\\Cristian\\Desktop\\BEAUCHEF PV FORECASTING\\001. DATASETS\\005. MODEL DATASETS'
#save_path = os.path.join(dir_path, 'sma_30min_dataset_20171010_20190923.pkl')

#%%############################################################################
################################ ANALYSIS #####################################
###############################################################################
from datetime import timedelta
from solarpv.analytics import plot_1D_radiation_data
from solarpv.analytics import plot_2D_radiation_data

date_first = datetime.strftime(first_date, '%d-%m-%Y')
date_last = datetime.strftime(last_date + timedelta(days=1), '%d-%m-%Y')
# checkear resultado con día soleado
plot_1D_radiation_data(db, 'Sistema', '04-11-2018', '05-11-2018', multiply_factor=80)

# plotear dataset
plot_2D_radiation_data(db, unit='kW', colname='Sistema', initial_date=date_first,final_date=date_last)

#%%############################################################################
############################# SETUP DATASET ###################################
###############################################################################
from sklearn.model_selection import train_test_split
import numpy as np
from math import floor

from datetime import datetime

from solarpv import ext_irradiation
from solarpv.database import reshape_by_day
from solarpv.database import time_window_dataset

from sklearn.preprocessing import MinMaxScaler

system_data = db.copy()

# obtenemos la cantidad de datos
n_data = system_data.shape[0]

# obtenemos el máximo valor de potencia registrado para normalizar
max_pot = np.max(system_data['Sistema'].values, axis=None)
system_data['Sistema'] = system_data['Sistema']/max_pot

# previews point
n_input = 12
n_output = 6
overlap = 18

X, Y = time_window_dataset(system_data['Sistema'], n_input, n_output, overlap)

X = np.expand_dims(X,-1)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 217)

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

# evaluar modelo de forecasting
cluster_metrics = cluster_evaluation(solar_1min_dataset, system_data, model)

