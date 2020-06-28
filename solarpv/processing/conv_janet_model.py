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
from solarpv.database import setup_lstm_dataset
from solarpv.database import setup_convlstm_dataset

from solarpv.database import lstm_standard_scaling
from solarpv.database import img_standard_scaling

# ----------------------------------------------------------------------------- 
# setup time windows
n_input = 12
n_output = 6
overlap = 1

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
X_g16_train, X_g16_test, g16_scaler = img_standard_scaling(X_g16_train, X_g16_test, clip=0.99)

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

from solarpv.layers import ConvJANet
from solarpv.layers import JANet

from solarpv.model import build_lstm_conv_model

from keras.optimizers import Adam
from keras.optimizers import RMSprop

from keras.utils import np_utils
from keras.utils import plot_model

lstm_type = JANet
lstm_layers_1 = 2; lstm_units_1 = 128; lstm_activation_1 = 'relu'
dense_layers_1 = 2; dense_units_1 = 128; dense_activation_1 = 'relu'
dense_layers_2 = 1; dense_units_2 = 128; dense_activation_2 = 'relu'
dense_layers_3 = 1; dense_units_3 = 128; dense_activation_3 = 'relu'

conv_type = ConvJANet
conv_layers_1 = 1; conv_filters_1 = 8; conv_activation_1 = 'relu'
conv_layers_2 = 2; conv_filters_2 = 16; conv_activation_2 = 'relu'
conv_layers_3 = 2; conv_filters_3 = 16; conv_activation_3 = 'relu'
dense_layers_4 = 1; dense_units_4 = 96; dense_activation_4 = 'sigmoid'

dense_layers_5 = 5; dense_units_5 = 64; dense_activation_5 = 'relu'

optimizer = Adam
learning_rate = 0.001

dropout_rate = 0.2

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
                                          dense_layers_5, dense_units_5, dense_activation_5,
                                          dropout_rate)

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
conv_name = 'ConvJANet' if conv_type==ConvJANet else 'ConvLSTM'
model_name = lstm_name + '_' + conv_name + '_c' + current_str

# callback
dir_path = '/home/hecate/Desktop/models/sys_g16_models'
file_name = model_name + '.h5'
save_path = os.path.join(dir_path, file_name)

model_checkpoint = ModelCheckpoint(save_path, save_best_only=True, monitor='val_loss')

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)

# entrenamos el modelo
model_history = forecasting_model.fit([X_train, X_g16_train], Y_train,
                                      batch_size=batch_size, epochs=epochs,
                                      validation_data = ([X_test, X_g16_test], Y_test),
                                      callbacks=[model_checkpoint, reduce_lr])
