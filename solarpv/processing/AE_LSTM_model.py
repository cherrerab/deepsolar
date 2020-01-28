# -*- coding: utf-8 -*-
#%% replication setup ---------------------------------------------------------
import numpy as np
import random
import os
import tensorflow as tf
from keras import backend as K

seed_value = 2427

# set `PYTHONHASHSEED` environment variable at a fixed value
os.environ['PYTHONHASHSEED']=str(seed_value)

# set `python` built-in pseudo-random generator at a fixed value
random.seed(seed_value)

# set `numpy` pseudo-random generator at a fixed value
np.random.seed(seed_value)

# set `tensorflow` pseudo-random generator at a fixed value
tf.set_random_seed(seed_value)

# configure a new global `tensorflow` session
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)

#%% load data -----------------------------------------------------------------
import pandas as pd

from solarpv.database import compact_database
from solarpv.database import select_date_range
from solarpv.database import adjust_timestamps
from solarpv.database import radiance_to_radiation

# -----------------------------------------------------------------------------
# cargar datos de potencia-SMA
sma_15min_path = '/media/hecate/Seagate Backup Plus Drive/datasets/system-power-15min-dataset.pkl'
power_dataset = pd.read_pickle(sma_15min_path)
power_dataset = select_date_range(power_dataset, '27-08-2018 04:15', '07-09-2019 00:00')

# compactar a base de 30min
power_dataset = compact_database(power_dataset, 2, use_average=True)
power_dataset = adjust_timestamps(power_dataset, -15*60)

# -----------------------------------------------------------------------------
# cargar datos de temperatura-SMA
temp_15min_path = '/media/hecate/Seagate Backup Plus Drive/datasets/temperature-15min-dataset.pkl'
temperature_dataset = pd.read_pickle(temp_15min_path)
temperature_dataset = select_date_range(temperature_dataset, '27-08-2018 04:15', '07-09-2019 00:00')

# compactar a base de 30min
temperature_dataset = compact_database(temperature_dataset, 2, use_average=True)
temperature_dataset = adjust_timestamps(temperature_dataset, -15*60)

# -----------------------------------------------------------------------------
# cargar datos solarimétricos
solar_1min_path = '/media/hecate/Seagate Backup Plus Drive/datasets/solarimetric-1min-dataset.pkl'
solarimetric_dataset = pd.read_pickle(solar_1min_path)
solarimetric_dataset = select_date_range(solarimetric_dataset, '27-08-2018 04:00', '07-09-2019 00:00')

# compactar a base de 30min
solarimetric_dataset = compact_database(solarimetric_dataset, 30, use_average=True)

# -----------------------------------------------------------------------------
# cargar enocoded satellital data base de 30min
encoded_goes16_path = '/media/hecate/Seagate Backup Plus Drive/datasets/goes16-96encoded-dataset.pkl'
encoded_goes16_dataset = pd.read_pickle(encoded_goes16_path)
encoded_goes16_dataset = select_date_range(encoded_goes16_dataset, '27-08-2018 04:00', '07-09-2019 00:00')


#%% setup dataset -------------------------------------------------------------
from solarpv.database import setup_lstm_dataset
from solarpv.database import lstm_standard_scaling

from datetime import datetime

import pandas as pd

# inicilizar dataset
colnames = ['Timestamp', 'Power', 'Module Temperature', 'External Temperature', 'Global', 'Diffuse', 'Direct', 'n_day', 'min_day']
dataset = pd.DataFrame(0.0, index=power_dataset.index, columns=colnames)

# asignar columna de timestamp y potencia
dataset['Timestamp'] = power_dataset['Timestamp']
dataset['Power'] = power_dataset['Sistema']

dataset['Module Temperature'] = temperature_dataset['Module']
dataset['External Temperature'] = solarimetric_dataset['Temperature']

dataset['Global'] = solarimetric_dataset['Global']
dataset['Diffuse'] = solarimetric_dataset['Diffuse']
dataset['Direct'] = solarimetric_dataset['Direct']

# -----------------------------------------------------------------------------
# rellenar resto columnas
date_format = '%d-%m-%Y %H:%M'
for i in dataset.index:
    # obtener timestamp del dato
    timestamp = dataset.at[i, 'Timestamp']
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
overlap = 3

# setup data splitting
train_days = 8
test_days = 2

# realizar dataset split sobre datos operativos
X_train, X_test, Y_train, Y_test = setup_lstm_dataset(dataset, 'Power',
                                                      train_days, test_days,
                                                      n_input, n_output,
                                                      overlap, ['Timestamp'])

# realizar dataset split sobre datos de las imágenes satelitales
X_goes16_train, X_goes16_test, _, _ = setup_lstm_dataset(encoded_goes16_dataset, 0,
                                                         train_days, test_days,
                                                         n_input, n_output,
                                                         overlap, ['Timestamp'])

# -----------------------------------------------------------------------------
# normalización
X_train, X_test, std_scaler = lstm_standard_scaling(X_train, X_test)
X_goes16_train, X_goes16_test, std_scaler_goes16 = lstm_standard_scaling(X_goes16_train, X_goes16_test)

# reshape dataset
n_feature = X_train.shape[2]
encoding_dim = X_goes16_train.shape[2]

feature_min = std_scaler[0, 0]
feature_max = std_scaler[0, 1]

Y_train = (Y_train - feature_min)/(feature_max-feature_min)
Y_test = (Y_test - feature_min)/(feature_max-feature_min)

#%% train lstm model ----------------------------------------------------------
import keras
from keras.models import Model
from keras.layers import Input
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Concatenate
from keras.optimizers import Adam


from keras.utils import np_utils
from keras.utils import plot_model

import matplotlib.pyplot as plt

# inicializamos la LSTM que trabaja con los datos -----------------------------
input_data = Input( shape=(n_input, n_feature) )

# añadimos las capas de procesamiento
data_model = LSTM(units = 256, return_sequences = True)(input_data)
data_model = LSTM(units = 256, return_sequences = True)(data_model)
data_model = Dropout(rate = 0.2)(data_model)

data_model = Dense(units = 512, activation = 'relu')(data_model)
data_model = Dropout(rate = 0.2)(data_model)

data_model = Dense(units = 256, activation = 'relu')(data_model)
data_model = Dense(units = 256, activation = 'relu')(data_model)
data_model = Dropout(rate = 0.2)(data_model)

# añadimos las capas de salida
data_model = Flatten()(data_model)
output_data = Dense(units = 256, activation = 'relu')(data_model)
data_model = Dropout(rate = 0.2)(data_model)

# inicializamos la LSTM que trabaja con las imágenes satelitales codificadas --
input_goes16 = Input( shape=(n_input, encoding_dim) )

# añadimos las capas de procesamiento
goes16_model = LSTM(units = 512, return_sequences = True)(input_data)
goes16_model = LSTM(units = 512, return_sequences = True)(goes16_model)
goes16_model = Dropout(rate = 0.2)(goes16_model)

goes16_model = Dense(units = 512, activation = 'relu')(goes16_model)
goes16_model = Dropout(rate = 0.2)(goes16_model)

goes16_model = Dense(units = 512, activation = 'relu')(goes16_model)
goes16_model = Dense(units = 256, activation = 'relu')(goes16_model)
goes16_model = Dropout(rate = 0.2)(goes16_model)

# añadimos las capas de salida
goes16_model = Flatten()(goes16_model)
output_goes16 = Dense(units = 256, activation = 'relu')(goes16_model)
output_goes16 = Dropout(rate = 0.2)(output_goes16)

# concatenamos los modelos para el modelo final
concat_layer = Concatenate()([output_data, output_goes16])
output_layer = Dense(units = n_output, activation = 'linear')(concat_layer)

forecasting_model = Model(inputs = [input_data, input_goes16], outputs = output_layer)

# configuramos el modelo de optimizacion a utilizar
optimizer_adam = Adam(lr=0.0001)
forecasting_model.compile(optimizer = optimizer_adam, loss = 'mse', metrics = ['mae'])

# entrenamos el modelo
model_history = forecasting_model.fit([X_train, X_goes16_train], Y_train, batch_size = 256, epochs = 256, validation_data = ([X_test, X_goes16_test], Y_test))

#%% load solar data -----------------------------------------------------------
from solarpv.database import compact_database

# cargar datos solarimetricos
solar_1min_path = '/media/hecate/Seagate Backup Plus Drive/datasets/solarimetric-1min-dataset.pkl'
solar_1min_dataset = pd.read_pickle(solar_1min_path)
solar_1min_dataset = select_date_range(solar_1min_dataset, '28-08-2018 00:00', '07-09-2019 00:00')
solar_1min_dataset = radiance_to_radiation(solar_1min_dataset)
solar_5min_dataset = compact_database(solar_1min_dataset, 5, use_average=True)

#%% model evaluation ----------------------------------------------------------
import numpy as np
from solarpv.database import compact_database
from solarpv.database import select_date_range
from solarpv.database import radiance_to_radiation

from solarpv.analytics import cluster_evaluation

eval_data = dataset.copy()
eval_goes16 = encoded_goes16_dataset.copy()

for i in np.arange(std_scaler.shape[0]):
    feature_min = std_scaler[i, 0]
    feature_max = std_scaler[i, 1]
    eval_data.iloc[:,i+1] = (eval_data.iloc[:,i+1] - feature_min)/(feature_max-feature_min)
    
for i in np.arange(std_scaler_goes16.shape[0]):
    feature_min = std_scaler_goes16[i, 0]
    feature_max = std_scaler_goes16[i, 1]
    eval_goes16.iloc[:,i+1] = (eval_goes16.iloc[:,i+1] - feature_min)/(feature_max-feature_min)

# evaluar modelo de forecasting
cluster_metrics = cluster_evaluation(solar_5min_dataset, [eval_data, eval_goes16], 'Power', forecasting_model, plot_clusters=True, random_state=33)

