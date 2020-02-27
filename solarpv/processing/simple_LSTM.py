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
power_dataset = adjust_timestamps(power_dataset, 15*60)

# -----------------------------------------------------------------------------
# cargar datos de temperatura-SMA
temp_15min_path = '/media/hecate/Seagate Backup Plus Drive/datasets/temperature-15min-dataset.pkl'
temperature_dataset = pd.read_pickle(temp_15min_path)
temperature_dataset = select_date_range(temperature_dataset, '27-08-2018 04:15', '07-09-2019 00:00')

# compactar a base de 30min
temperature_dataset = compact_database(temperature_dataset, 2, use_average=True)
temperature_dataset = adjust_timestamps(temperature_dataset, 15*60)

# -----------------------------------------------------------------------------
# cargar datos solarimétricos
solar_1min_path = '/media/hecate/Seagate Backup Plus Drive/datasets/solarimetric-1min-dataset.pkl'
solarimetric_dataset = pd.read_pickle(solar_1min_path)
solarimetric_dataset = select_date_range(solarimetric_dataset, '27-08-2018 04:01', '07-09-2019 00:00')

# compactar a base de 30min
solarimetric_dataset = compact_database(solarimetric_dataset, 30, use_average=True)
solarimetric_dataset = adjust_timestamps(solarimetric_dataset, 29*60)

#%% analysis ------------------------------------------------------------------
from datetime import datetime, timedelta
from solarpv.analytics import plot_2D_radiation_data
from solarpv.analytics import plot_1D_radiation_data
from solarpv.analytics import plot_performance_ratio

# plotear dataset potencia
plot_2D_radiation_data(power_dataset, unit='kW', colname='Sistema', initial_date='27-08-2018',final_date='07-09-2019')

# checkear resultado con día soleado
plot_1D_radiation_data(power_dataset, 'Sistema', '04-11-2018', '05-11-2018', multiply_factor=40)
plot_1D_radiation_data(radiance_to_radiation(solarimetric_dataset), 'Global', '04-11-2018', '05-11-2018', multiply_factor=1)

# plotear dataset temperatura
plot_2D_radiation_data(temperature_dataset, unit='°C', colname='Module', initial_date='27-08-2018',final_date='07-09-2019')

# plotear dataset solarimetrico
plot_2D_radiation_data(radiance_to_radiation(solarimetric_dataset), unit='kWh/m2', colname='Global', initial_date='27-08-2018',final_date='07-09-2019')

plot_2D_radiation_data(solarimetric_dataset, unit='kW/m2', colname='Global', initial_date='27-08-2018',final_date='07-09-2019')
plot_2D_radiation_data(solarimetric_dataset, unit='kW/m2', colname='Diffuse', initial_date='27-08-2018',final_date='07-09-2019')
plot_2D_radiation_data(solarimetric_dataset, unit='kW/m2', colname='Direct', initial_date='27-08-2018',final_date='07-09-2019')

plot_2D_radiation_data(solarimetric_dataset, unit='°C', colname='Temperature', initial_date='27-08-2018',final_date='07-09-2019')

# plotear performance ratio
plot_performance_ratio(power_dataset, solarimetric_dataset, '27-08-2018', '07-09-2019')

#%% setup dataset -------------------------------------------------------------
from datetime import datetime, timedelta
from solarpv.database import setup_lstm_dataset
from solarpv.database import lstm_standard_scaling


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

# realizar dataset split
X_train, X_test, Y_train, Y_test = setup_lstm_dataset(dataset, 'Power', 8, 2, n_input, n_output, overlap, ['Timestamp'])

# -----------------------------------------------------------------------------
# normalización
X_train, X_test, std_scaler = lstm_standard_scaling(X_train, X_test)

# reshape dataset
n_feature = X_train.shape[2]

feature_min = std_scaler[0, 0]
feature_max = std_scaler[0, 1]

Y_train = (Y_train - feature_min)/(feature_max-feature_min)
Y_test = (Y_test - feature_min)/(feature_max-feature_min)

#%% train lstm model ----------------------------------------------------------
import keras
from keras.models import Model
from keras.layers import Input
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.optimizers import Adam

import matplotlib.pyplot as plt

# inicializamos la LSTM que trabaja con los datos -----------------------------
input_data = Input( shape=(n_input, n_feature) )

# añadimos las capas de procesamiento
data_model = LSTM(units = 1024, return_sequences = True)(input_data)
data_model = LSTM(units = 1024, return_sequences = True)(data_model)
data_model = Dropout(rate = 0.2)(data_model)

data_model = Dense(units = 1024, activation = 'relu')(data_model)
data_model = Dropout(rate = 0.2)(data_model)

data_model = Dense(units = 512, activation = 'relu')(data_model)
data_model = Dense(units = 512, activation = 'relu')(data_model)
data_model = Dropout(rate = 0.2)(data_model)

# añadimos las capas de salida
data_model = Flatten()(data_model)
output_data = Dense(units = 512, activation = 'relu')(data_model)
data_model = Dropout(rate = 0.2)(data_model)

output_layer = Dense(units = n_output, activation = 'linear')(output_data)

# compilamos el modelo
forecasting_model = Model(inputs = input_data, outputs = output_layer)

optimizer_adam = Adam(lr=0.0001)
forecasting_model.compile(optimizer = optimizer_adam, loss = 'mse', metrics = ['mae'])

# entrenamos el modelo
model_history = forecasting_model.fit(X_train, Y_train, batch_size = 256, epochs = 256, validation_data = (X_test, Y_test))

#%% training evaluation -------------------------------------------------------

# visualizamos la evolucion de la funcion de perdida
test_mse, test_mae = forecasting_model.evaluate(X_test, Y_test, batch_size = 30)

plt.figure()
plt.plot(model_history.history['loss'], c=(0.050383, 0.029803, 0.527975, 1.0))
plt.plot(model_history.history['val_loss'], c=(0.798216, 0.280197, 0.469538, 1.0))
plt.title('ANN model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['train', 'validation'], loc='upper right')
plt.show()

#%% load solar data -----------------------------------------------------------
from solarpv.database import compact_database

# cargar datos solarimetricos
solar_1min_path = 'C:\\Cristian\\datasets\\solarimetric_1min_dataset.pkl'
solar_1min_dataset = pd.read_pickle(solar_1min_path)
solar_1min_dataset = select_date_range(solar_1min_dataset, '28-08-2018 00:01', '07-09-2019 00:00')
solar_1min_dataset = radiance_to_radiation(solar_1min_dataset)

solar_5min_dataset = compact_database(solar_1min_dataset, 5, use_average=True)
solar_5min_dataset = adjust_timestamps(solar_5min_dataset, 4*60)

#%% model evaluation ----------------------------------------------------------
import numpy as np
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
cluster_metrics = cluster_evaluation(solar_5min_dataset, [eval_data], 'Power', model, plot_clusters=True, random_state=33)