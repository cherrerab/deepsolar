# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 16:38:18 2020

@author: Cristian
"""

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

# -----------------------------------------------------------------------------
# cargar datos
dataset_path = 'C:\\Users\\Cristian\\Desktop\\BEAUCHEF PV FORECASTING\\datasets\\datasets_pkl\\processed\\dat_syst_30min_s20180827_e20190907.pkl'
dataset = pd.read_pickle(dataset_path)

#%% setup dataset -------------------------------------------------------------
from deepsolar.database import setup_lstm_dataset
from deepsolar.database import lstm_standard_scaling

# ----------------------------------------------------------------------------- 
# setup time windows
n_input = 12
n_output = 6
overlap = 1

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

#%% setup model
import keras
from keras.layers import LSTM
from deepsolar.layers import JANet

from deepsolar.model import build_lstm_model

from keras.optimizers import Adam

from keras.utils import np_utils
from keras.utils import plot_model

lstm_type = JANet
lstm_layers_1 = 2; lstm_units_1 = 128; lstm_activation_1 = 'relu'
dense_layers_1 = 2; dense_units_1 = 256; dense_activation_1 = 'relu'
dense_layers_2 = 2; dense_units_2 = 128; dense_activation_2 = 'relu'
dense_layers_3 = 2; dense_units_3 = 128; dense_activation_3 = 'relu'

optimizer = Adam
learning_rate = 1e-4

dropout_rate = 0.2

forecasting_model = build_lstm_model(n_input, n_output, n_feature, lstm_type,
                                          lstm_layers_1, lstm_units_1, lstm_activation_1,
                                          dense_layers_1, dense_units_1, dense_activation_1,
                                          dense_layers_2, dense_units_2, dense_activation_2,
                                          dense_layers_3, dense_units_3, dense_activation_3,
                                          dropout_rate=dropout_rate)

forecasting_model.compile(optimizer=optimizer(lr=learning_rate), loss = 'mse', metrics = ['mae'])
forecasting_model.summary()

#%% train model
from keras.callbacks import ModelCheckpoint
from keras.callbacks import ReduceLROnPlateau
from datetime import datetime
import os

batch_size = 128
epochs = 200

# model name
current_time = datetime.now()
current_str = current_time.strftime('%Y%m%d%H%M%S')
lstm_name = 'JANet' if lstm_type==JANet else 'LSTM'
model_name = lstm_name + '_c' + current_str

# callback
dir_path = ''
file_name = model_name + '.h5'
save_path = os.path.join(dir_path, file_name)

model_checkpoint = ModelCheckpoint(save_path, save_best_only=True, monitor='val_mean_absolute_error')

reduce_lr = ReduceLROnPlateau(monitor='val_mean_absolute_error', factor=0.25, patience=5, min_lr=1e-6)

# entrenamos el modelo
model_history = forecasting_model.fit(X_train, Y_train,
                                      batch_size=batch_size, epochs=epochs,
                                      validation_data = (X_test, Y_test),
                                      callbacks=[reduce_lr])