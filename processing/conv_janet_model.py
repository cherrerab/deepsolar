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
sys_dataset_path = '/media/hecate/Seagate Backup Plus Drive/datasets/datasets_pkl/processed/dat_syst_30min_s20180827_e20190907.pkl'
sys_dataset = pd.read_pickle(sys_dataset_path)

# -----------------------------------------------------------------------------
# cargar datos del goes16
g16_dataset_path = '/media/hecate/Seagate Backup Plus Drive/datasets/datasets_pkl/processed/dat_noaa_30min_s20180827_e20190907.pkl'
g16_dataset = pd.read_pickle(g16_dataset_path)

#%% setup dataset -------------------------------------------------------------
from deepsolar.database import setup_lstm_dataset
from deepsolar.database import setup_convlstm_dataset

from deepsolar.database import lstm_standard_scaling
from deepsolar.database import img_standard_scaling

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

#%% setup model
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
from keras.layers import MaxPooling3D

from deepsolar.layers import ConvJANet
from deepsolar.layers import JANet

from deepsolar.model import build_lstm_conv_model

from keras.optimizers import Adam

from keras.utils import np_utils
from keras.utils import plot_model

lstm_type = LSTM
lstm_layers_1 = 2; lstm_units_1 = 128; lstm_activation_1 = 'relu'
dense_layers_1 = 2; dense_units_1 = 123; dense_activation_1 = 'relu'
dense_layers_2 = 1; dense_units_2 = 64; dense_activation_2 = 'relu'
dense_layers_3 = 2; dense_units_3 = 64; dense_activation_3 = 'relu'

lstm_type = ConvJANet
conv_layers_1 = 1, conv_filters_1 = 32, conv_activation_1 = 'linear',
conv_layers_2 = 2, conv_filters_2 = 24, conv_activation_2 = 'linear',
conv_layers_3 = 1, conv_filters_3 = 18, conv_activation_3 = 'tanh',
dense_layers_4 = 2, dense_units_4 = 128, dense_activation_4 = 'tanh',

optimizer = Adam
learning_rate = 1e-4

dropout_rate = 0.1

forecasting_model = build_lstm_conv_model(n_input, n_output, n_feature, img_size,
                                          lstm_type, conv_type,
                                          lstm_layers_1, lstm_units_1, lstm_activation_1,
                                          dense_layers_1, dense_units_1, dense_activation_1,
                                          dense_layers_2, dense_units_2, dense_activation_2,
                                          dense_layers_3, dense_units_3, dense_activation_3,
                                          conv_layers_1, conv_filters_1, conv_activation_1,
                                          conv_layers_2, conv_filters_2, conv_activation_2,
                                          conv_layers_3, conv_filters_3, conv_activation_3,
                                          dense_layers_4, dense_units_4, dense_activation_4,
                                          optimizer, learning_rate, dropout_rate=0.2)

forecasting_model.compile(optimizer=optimizer(lr=learning_rate), loss = 'mse', metrics = ['mae'])
forecasting_model.summary()

#%% train model
from keras.callbacks import ModelCheckpoint
from keras.callbacks import ReduceLROnPlateau
from datetime import datetime
import os

batch_size = 128
epochs = 512

# model name
current_time = datetime.now()
current_str = current_time.strftime('%Y%m%d%H%M%S')
model_name = 'SysG16Model_c' + current_str

# callback
dir_path = ''
file_name = model_name + '_e{epoch:3d}_s{val_mean_absolute_error:.4f}.h5'
save_path = os.path.join(dir_path, file_name)

model_checkpoint = ModelCheckpoint(save_path, save_best_only=True, monitor='val_mean_absolute_error')

reduce_lr = ReduceLROnPlateau(monitor='val_mean_absolute_error', factor=0.25, patience=5, min_lr=1e-6)

# entrenamos el modelo
model_history = forecasting_model.fit([X_train, X_g16_train], Y_train,
                                      batch_size=batch_size, epochs=epochs,
                                      validation_data = ([X_test, X_g16_test], Y_test),
                                      callbacks=[model_checkpoint, reduce_lr])
