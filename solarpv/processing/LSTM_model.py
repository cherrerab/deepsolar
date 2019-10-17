# -*- coding: utf-8 -*-
#%%############################################################################
############################### LOAD DATA #####################################
###############################################################################
import pandas as pd

from solarpv.database import compact_database
from solarpv.database import select_date_range
from solarpv.database import adjust_timestamps
from solarpv.database import radiance_to_radiation

# -----------------------------------------------------------------------------
# cargar datos de potencia-SMA
sma_15min_path = 'C:\\Cristian\\003. SMA DATASET\\005. 15 MINUTES SYSTEM DATA2\\sma-15min-dataset.pkl'
power_dataset = pd.read_pickle(sma_15min_path)
power_dataset = select_date_range(power_dataset, '27-08-2018 04:15', '07-09-2019 00:00')

# compactar a base de 30min
power_dataset = compact_database(power_dataset, 2, use_average=True)
power_dataset = adjust_timestamps(power_dataset, -15*60)

# -----------------------------------------------------------------------------
# cargar datos de temperatura-SMA
temp_15min_path = 'C:\\Cristian\\003. SMA DATASET\\004. TEMPERATURE DATA\\temperature-15min-dataset.pkl'
temperature_dataset = pd.read_pickle(temp_15min_path)
temperature_dataset = select_date_range(temperature_dataset, '27-08-2018 04:15', '07-09-2019 00:00')

# compactar a base de 30min
temperature_dataset = compact_database(temperature_dataset, 2, use_average=True)
temperature_dataset = adjust_timestamps(temperature_dataset, -15*60)

# -----------------------------------------------------------------------------
# cargar datos solarimétricos
solar_1min_path = 'C:\\Cristian\\001. SOLARIMETRIC DATA\\solarimetric-1min-dataset.pkl'
solarimetric_dataset = pd.read_pickle(solar_1min_path)
solarimetric_dataset = select_date_range(solarimetric_dataset, '27-08-2018 04:00', '07-09-2019 00:00')

# compactar a base de 30min
solarimetric_dataset = compact_database(solarimetric_dataset, 30, use_average=True)
solarimetric_dataset = adjust_timestamps(solarimetric_dataset, -30*60)

#%%############################################################################
############################# SETUP DATASET ###################################
###############################################################################
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

# realizar dataset split
X_train, X_test, Y_train, Y_test = setup_lstm_dataset(dataset, 'Power', 8, 2, n_input, n_output, overlap, ['Timestamp'])

# -----------------------------------------------------------------------------
# normalización
X_train, X_test, std_scaler = lstm_standard_scaling(X_train, X_test, Y_train, Y_test)

# reshape dataset
n_feature = X_train.shape[2]

feature_min = std_scaler[0, 0]
feature_max = std_scaler[0, 1]

Y_train = (Y_train - feature_min)/(feature_max-feature_min)
Y_test = (Y_test - feature_min)/(feature_max-feature_min)

#%%############################################################################
############################## LSTM MODEL #####################################
###############################################################################

from solarpv.model import optimize_model_structure
from solarpv.model import init_lstm_model

lstm_layers = [2,3]
lstm_units = [32,64]

dense_layers_1 = [2,3]
dense_units_1 = [128, 256]
dense_activation_1 = ['relu']

dense_layers_2 = [1]
dense_units_2 = [64]
dense_activation_2 = ['relu']

dense_layers_3 = [2]
dense_units_3 = [64]
dense_activation_3 = ['relu']

optimizers = ['adam']
batches = [128]
epochs = [200]

param_grid  = dict(input_shape=[(n_input, n_feature)], output_shape=[n_output],
                   lstm_layers=lstm_layers, lstm_units=lstm_units,
                   dense_layers_1=dense_layers_1, dense_units_1=dense_units_1, dense_activation_1=dense_activation_1,
                   dense_layers_2=dense_layers_2, dense_units_2=dense_units_2, dense_activation_2=dense_activation_2,
                   dense_layers_3=dense_layers_3, dense_units_3=dense_units_3, dense_activation_3=dense_activation_3,
                   optimizer=optimizers, epochs=epochs, batch_size=batches)

best_model = optimize_model_structure(init_lstm_model, param_grid, X_train, X_test, Y_train, Y_test)