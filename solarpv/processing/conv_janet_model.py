# -*- coding: utf-8 -*-
#%% replication setup ---------------------------------------------------------
import numpy as np
import random
import os
import tensorflow as tf

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

#%% load data -----------------------------------------------------------------
import pandas as pd

# -----------------------------------------------------------------------------
# cargar datos del sistema
sys_dataset_path = 'C:\\Users\\Cristian\\Desktop\\BEAUCHEF PV FORECASTING\\datasets\\datasets_pkl\\processed\\dat_syst_30min_s20180827_e20190907.pkl'
sys_dataset = pd.read_pickle(sys_dataset_path)

# -----------------------------------------------------------------------------
# cargar datos del goes16
g16_dataset_path = 'C:\\Users\\Cristian\\Desktop\\BEAUCHEF PV FORECASTING\\datasets\\datasets_pkl\\processed\\dat_syst_30min_s20180827_e20190907.pkl'
g16_dataset = pd.read_pickle(g16_dataset_path)

#%% setup dataset -------------------------------------------------------------
from solarpv.database import setup_lstm_dataset
from solarpv.database import setup_convlstm_dataset

from solarpv.database import lstm_standard_scaling
from solarpv.database import img_standard_scaling

# ----------------------------------------------------------------------------- 
# setup time windows
n_input = 12
n_output = 6
overlap = 3

# setup data splitting
train_days = 8
test_days = 2

# realizar dataset split sobre datos operativos
X_train, X_test, Y_train, Y_test = setup_lstm_dataset(sys_dataset, 'Power',
                                                      train_days, test_days,
                                                      n_input, n_output,
                                                      overlap, ['Timestamp'])

# realizar dataset split sobre datos de las imágenes satelitales
X_g16_train, X_g16_test = setup_convlstm_dataset(g16_dataset,
                                                       train_days, test_days,
                                                       n_input, n_output,
                                                       overlap)

# -----------------------------------------------------------------------------
# normalización
X_train, X_test, std_scaler = lstm_standard_scaling(X_train, X_test)
X_g16_train, X_g16_test, _ = img_standard_scaling(X_g16_train, X_g16_test)

# reshape dataset
n_feature = X_train.shape[2]
img_size = X_g16_train.shape[2]

feature_min = std_scaler[0, 0]
feature_max = std_scaler[0, 1]

Y_train = (Y_train - feature_min)/(feature_max-feature_min)
Y_test = (Y_test - feature_min)/(feature_max-feature_min)

#%% train model
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

from solarpv.layers import ConvJANet

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
goes16_model = ConvJANet(filters=8, kernel_size=(3, 3),
                         padding='same', return_sequences=True)(input_goes16)
goes16_model = BatchNormalization()(goes16_model)

goes16_model = ConvJANet(filters=8, kernel_size=(3, 3),
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