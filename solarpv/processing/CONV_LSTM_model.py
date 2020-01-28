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
from solarpv.database import resize_goes16_dataset

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

# cargar imágenes satelitales
goes16_ds_path = '/media/hecate/Seagate Backup Plus Drive/datasets/goes16-30min-48px-dataset.pkl'
goes16_dataset = pd.read_pickle(goes16_ds_path)
goes16_dataset = select_date_range(goes16_dataset, '27-08-2018 04:00', '07-09-2019 00:00')

#%% setup dataset -------------------------------------------------------------
from solarpv.database import setup_lstm_dataset
from solarpv.database import setup_convlstm_dataset

from solarpv.database import lstm_standard_scaling
from solarpv.database import img_standard_scaling

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
n_input = 6
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
X_goes16_train, X_goes16_test = setup_convlstm_dataset(goes16_dataset,
                                                       train_days, test_days,
                                                       n_input, n_output,
                                                       overlap)

# -----------------------------------------------------------------------------
# normalización
X_train, X_test, std_scaler = lstm_standard_scaling(X_train, X_test)
X_goes16_train, X_goes16_test, _ = img_standard_scaling(X_goes16_train, X_goes16_test)

# reshape dataset
n_feature = X_train.shape[2]
img_size = X_goes16_train.shape[2]

feature_min = std_scaler[0, 0]
feature_max = std_scaler[0, 1]

Y_train = (Y_train - feature_min)/(feature_max-feature_min)
Y_test = (Y_test - feature_min)/(feature_max-feature_min)

#%% load processed datasets
import numpy as np

dataset_file = np.load('/media/hecate/Seagate Backup Plus Drive/datasets/convlstm-dataset.npz')

X_train = dataset_file['X_train']
X_test = dataset_file['X_test']

X_goes16_train = dataset_file['X_goes16_train']
X_goes16_test = dataset_file['X_goes16_test']

Y_train = dataset_file['Y_train']
Y_test = dataset_file['Y_test']

# setup time windows
n_input = 6
n_output = 6
overlap = 3

# reshape dataset
n_feature = X_train.shape[2]
img_size = X_goes16_train.shape[2]

#%% train lstm setup ----------------------------------------------------------
import keras
from keras.models import Model
from keras.layers import Input
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import ConvLSTM2D
from keras.layers import BatchNormalization
from keras.layers import Concatenate
from keras.optimizers import Adam


from keras.utils import np_utils
from keras.utils import plot_model

import matplotlib.pyplot as plt

# inicializamos la LSTM que trabaja con los datos -----------------------------
input_data = Input( shape=(n_input, n_feature) )

# añadimos las capas de procesamiento
data_model = LSTM(units = 128, return_sequences = True)(input_data)
data_model = LSTM(units = 128, return_sequences = True)(data_model)
data_model = Dropout(rate = 0.2)(data_model)

data_model = Dense(units = 128, activation = 'relu')(data_model)
data_model = Dropout(rate = 0.2)(data_model)

data_model = Dense(units = 128, activation = 'relu')(data_model)
data_model = Dense(units = 64, activation = 'relu')(data_model)
data_model = Dropout(rate = 0.2)(data_model)

# añadimos las capas de salida
data_model = Flatten()(data_model)
output_data = Dense(units = 64, activation = 'relu')(data_model)

# inicializamos la LSTM que trabaja con las imágenes --------------------------
input_goes16 = Input( shape=(n_input, img_size, img_size, 1) )

# añadimos las capas de procesamiento
goes16_model = ConvLSTM2D(filters=8, kernel_size=(3, 3),
                          padding='same', return_sequences=True)(input_goes16)
goes16_model = BatchNormalization()(goes16_model)

#goes16_model = ConvLSTM2D(filters=96, kernel_size=(3, 3),
#                          padding='same', return_sequences=True)(goes16_model)
#goes16_model = BatchNormalization()(goes16_model)

goes16_model = ConvLSTM2D(filters=8, kernel_size=(3, 3),
                          padding='same', return_sequences=True)(goes16_model)
goes16_model = BatchNormalization()(goes16_model)

# añadimos las capas de salida
goes16_model = Flatten()(goes16_model)
output_goes16 = Dense(units = 512, activation = 'relu')(goes16_model)

# concatenamos los modelos para el modelo final
concat_layer = Concatenate()([output_data, output_goes16])
output_layer = Dense(units = n_output, activation = 'linear')(concat_layer)

forecasting_model = Model(inputs = [input_data, input_goes16], outputs = output_layer)

# configuramos el modelo de optimizacion a utilizar
optimizer_adam = Adam(lr=0.0001)
forecasting_model.compile(optimizer = optimizer_adam, loss = 'mse', metrics = ['mae'])

#%% train model ---------------------------------------------------------------

# entrenamos el modelo
model_history = forecasting_model.fit([X_train, X_goes16_train], Y_train, batch_size = 256, epochs = 512, validation_data = ([X_test, X_goes16_test], Y_test))