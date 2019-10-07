# -*- coding: utf-8 -*-

###############################################################################
############################# PROCESS DATA ####################################
###############################################################################
import os

import pandas as pd

from solarpv.analytics import cluster_daily_radiation
from solarpv.database import compact_database
from solarpv.database import select_date_range
from solarpv.database import adjust_timestamps

from datetime import datetime

# importar datasets
sma_15min_path = 'C:\\Users\\Cristian\\Desktop\\BEAUCHEF PV FORECASTING\\001. DATASETS\\003. SMA DATA\\005. 15 MINUTES SYSTEM DATA2\\sma-15min-dataset.pkl'
sma_15min_dataset = pd.read_pickle(sma_15min_path)
sma_15min_dataset = compact_database(sma_15min_dataset, 4, use_average=True)
sma_15min_dataset = adjust_timestamps(sma_15min_dataset, -15*60)

sma_hour_path = 'C:\\Users\\Cristian\\Desktop\\BEAUCHEF PV FORECASTING\\001. DATASETS\\003. SMA DATA\\001. HOURLY SYSTEM DATA\\sma-hourly-dataset.pkl'
sma_hour_dataset = pd.read_pickle(sma_hour_path)
sma_hour_dataset = select_date_range(sma_hour_dataset, '10-10-2017', '27-08-2018')

# combinar datasets
db = pd.concat([sma_hour_dataset, sma_15min_dataset])

# arreglar timestamps
db = db.reset_index()
db.drop('index',axis=1, inplace=True)

date_format = '%d-%m-%Y %H:%M'

time_freq = str(3600) + 'S'
first_date = datetime.strptime( '10-10-2017 00:00', date_format)
last_date = datetime.strptime( '23-09-2019 23:00', date_format)

idx = pd.date_range(first_date, last_date, freq=time_freq)

db.index = pd.DatetimeIndex( db[u'Timestamp'].values, dayfirst=True )
db = db.reindex( idx, fill_value=0.0 )
db[u'Timestamp'] = idx.strftime(date_format)

# resetear index a enteros
db = db.reset_index()
db.drop('index',axis=1, inplace=True)

# guardar dataset
dir_path = 'C:\\Users\\Cristian\\Desktop\\BEAUCHEF PV FORECASTING\\001. DATASETS\\005. MODEL DATASETS'
save_path = os.path.join(dir_path, 'sma_hourly_dataset_20171010_20190923.pkl')

#%%############################################################################
################################ ANALYSIS #####################################
###############################################################################

from solarpv.analytics import plot_1D_radiation_data
from solarpv.analytics import plot_2D_radiation_data

# checkear resultado con día soleado
plot_1D_radiation_data(db, 'Sistema', '04-11-2018', '05-11-2018', multiply_factor=80)

# plotear dataset
plot_2D_radiation_data(db, unit='kW', colname='Sistema', initial_date='10-10-2017',final_date='23-09-2019')

#%%############################################################################
############################# SETUP DATASET ###################################
###############################################################################

from solarpv.database import reshape_by_day
from sklearn.model_selection import train_test_split
import numpy as np

from datetime import datetime
from solarpv import ext_irradiation

from sklearn.preprocessing import MinMaxScaler

data = reshape_by_day(db, 'Sistema', '10-10-2017', '24-09-2019')

# obtenemos el máximo valor de potencia registrado para normalizar
max_pot = np.max(data.values, axis=None)

X_raw = reshape_by_day(db, 'Sistema', '10-10-2017', '23-09-2019').transpose()
Y = reshape_by_day(db, 'Sistema', '11-10-2017', '24-09-2019').transpose()

# obtener dataset de input
X = pd.DataFrame(0.0, index=np.arange(Y.shape[0]), columns=['G_avg', 'TOD_max', 'NDD', 'n_day'])

for i in X.index:
    x = X_raw.iloc[i, :]
    
    # obtener promedio de generación
    X.at[i, 'G_avg'] = np.mean( x )
    
    x_norm = x/max_pot
    
    # obtener radiación extraterrestre durante el día
    date_format = '%d-%m-%Y'
    date = X_raw.index[i]
    timestamps = [date + ' ' + h for h in X_raw.columns]
    
    ext_rad = [ext_irradiation(t, 3600) for t in timestamps]
    ext_rad = np.array(ext_rad)/(1367.0)
    
    # obtener el máximo de la tercera derivada
    X.at[i, 'TOD_max'] = np.max( np.abs( np.diff( np.diff( np.diff(x_norm - ext_rad) ) ) ) )
    
    # obtener el error cuadrático medio entre la generacion y la irradiacion extraterrestre
    X.at[i, 'NDD'] = np.sqrt( np.mean(np.power(x_norm - ext_rad, 2)) )
    
    # obtener el numero del dia
    date = datetime.strptime(date, date_format)
    date_tt = date.timetuple()
    X.at[i, 'n_day'] = date_tt.tm_yday
    
# normalizamos el dataset
sc =  MinMaxScaler(feature_range=(-1,1))
    
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.15, random_state = 0)

X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#%%############################################################################
############################### ANN MODEL #####################################
###############################################################################
import keras 
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils import np_utils
from keras.utils import plot_model

import matplotlib.pyplot as plt

# inicializamos la ANN
model = Sequential()

# creamos la primera capa de input
model.add(Dense(units = 64, activation = 'relu', input_dim = 4))
keras.layers.Dropout(rate = 0.1)

# creamos la segunda y terceras capas (con DropOut)
model.add(Dense(units = 128, activation = 'relu'))
keras.layers.Dropout(rate = 0.1)

model.add(Dense(units = 128, activation = 'relu'))
keras.layers.Dropout(rate = 0.1)

# creamos la capa de salida
model.add(Dense(units = 24))

# configuramos el modelo de optimizacion a utilizar
model.compile(optimizer = 'rmsprop', loss = 'mse', metrics = ['mae'])

# entrenamos el modelo
model_history = model.fit(X_train, Y_train, batch_size = 50, epochs = 100, validation_data = (X_test, Y_test))

#%%############################################################################
########################## MODEL EVALUATION ###################################
###############################################################################

# obtenemos el error respecto al test set
Y_pred = model.predict(X_test)
mse = np.mean(np.power(Y_test - Y_pred, 2), axis = 1)

# visualizamos la evolucion de la funcion de perdida
test_mse, test_mae = model.evaluate(X_test, Y_test, batch_size = 30)

plt.plot(model_history.history['loss'])
plt.plot(model_history.history['val_loss'])
plt.title('ANN model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['train', 'validation'], loc='upper right')
plt.show()


