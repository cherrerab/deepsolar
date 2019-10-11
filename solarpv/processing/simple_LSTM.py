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
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from solarpv.database import setup_lstm_dataset

system_data = db[['Timestamp','Sistema']]

# time window size
n_input = 12
n_output = 6
overlap = 18

# realizar dataset split
X_train, X_test, Y_train, Y_test = setup_lstm_dataset(system_data, 'Sistema', 8, 2, 12, 6, 18, ['Timestamp'])

# -----------------------------------------------------------------------------
# normalización
std_scaler = np.zeros((X_train.shape[2], 2))

# por cada feature del dataset de training
for i in range(X_train.shape[2]):
    # obtener el mínimo
    feature_min = np.min(X_train[:,:,i], axis=None)
    # obtener el máximo
    feature_max = np.max(X_train[:,:,i], axis=None)
    
    # normalizar
    X_train[:,:,i] = (X_train[:,:,i] - feature_min)/(feature_max - feature_min)
    
    # agregar al scaler
    std_scaler[i, 0] = feature_min
    std_scaler[i, 1] = feature_max

# por cada feature del dataset de testing
for i in range(X_train.shape[2]):
    # obtener parámetros del scaler
    feature_min = std_scaler[i, 0]
    feature_max = std_scaler[i, 1]
    
    # normalizar
    X_test[:,:,i] = (X_test[:,:,i] - feature_min)/(feature_max - feature_min)
    
    Y_train[:,:] = (Y_train[:,:] - feature_min)/(feature_max - feature_min)
    Y_test[:,:] = (Y_test[:,:] - feature_min)/(feature_max - feature_min)

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

eval_data = system_data.copy()
eval_data['Sistema'] = (eval_data['Sistema'] - feature_min)/(feature_max-feature_min)


# evaluar modelo de forecasting
cluster_metrics = cluster_evaluation(solar_1min_dataset, eval_data, model)

