# -*- coding: utf-8 -*-
#%% replication setup
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

#%% load model
from keras.models import load_model
from deepsolar.layers import JANet
from deepsolar.layers import ConvJANet

file_path = '/home/hecate/Desktop/models/sys_g16_models/best_model/JANet_ConvJANet_c20200322111711.h5'
forecasting_model = load_model(file_path, custom_objects={'JANet':JANet, 'ConvJANet':ConvJANet})

#%% load dataset
import pandas as pd

# cargar datos del sistema
sys_dataset_path = '/media/hecate/Seagate Backup Plus Drive/datasets/datasets_pkl/processed/dat_syst_30min_s20180827_e20190907.pkl'
sys_dataset = pd.read_pickle(sys_dataset_path)

# cargar datos del goes16
g16_dataset_path = '/media/hecate/Seagate Backup Plus Drive/datasets/datasets_pkl/processed/dat_noaa_30min_s20180827_e20190907.pkl'
g16_dataset = pd.read_pickle(g16_dataset_path)

#%% process dataset
from deepsolar.database import setup_lstm_dataset
from deepsolar.database import setup_convlstm_dataset

from deepsolar.database import lstm_standard_scaling
from deepsolar.database import img_standard_scaling

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

## realizar dataset split sobre datos de las imágenes satelitales
#X_g16_train, X_g16_test = setup_convlstm_dataset(g16_dataset,
#                                                       train_days, test_days,
#                                                       n_input, n_output,
#                                                       overlap)

# -----------------------------------------------------------------------------
# normalización
X_train, X_test, sys_scaler = lstm_standard_scaling(X_train, X_test)
#X_g16_train, X_g16_test, g16_scaler = img_standard_scaling(X_g16_train, X_g16_test)

# obtener input shape
n_feature = X_train.shape[2]
#img_size = X_g16_train.shape[2]

# normalizar sys_dataset
scl_dataset = sys_dataset.copy()

for i in np.arange(n_feature):
    feature_min = sys_scaler[i, 0]
    feature_max = sys_scaler[i, 1]
    scl_dataset.iloc[:,i+1] = (sys_dataset.iloc[:,i+1] - feature_min)/(feature_max - feature_min)
    
sys_min = sys_scaler[0, 0]
sys_max = sys_scaler[0, 1]
    
## normalizar g16_dataset
#g16_min = g16_scaler[0, 0]
#g16_max = g16_scaler[0, 1]
#g16_dataset.iloc[:, 3:] = (sys_dataset.iloc[:,3:] - g16_min)/(g16_max - g16_min)

#%% get list of days
from deepsolar import get_timestep
from datetime import datetime

date_format = '%d-%m-%Y %H:%M'
timestamps = sys_dataset['Timestamp'].values

timestep = get_timestep(sys_dataset['Timestamp'], date_format)

initial_date = datetime.strptime(timestamps[0], date_format)
final_date = datetime.strptime(timestamps[-1], date_format)

dates = pd.date_range(initial_date, final_date, freq='1D')
dates = dates.strftime('%d-%m-%Y')

initial_date = initial_date.strftime('%d-%m-%Y')
final_date = final_date.strftime('%d-%m-%Y')

#%% get cluster labels
from deepsolar.database import compact_database
from deepsolar.database import select_date_range
from deepsolar.database import adjust_timestamps

from deepsolar.analytics import cluster_daily_radiation

# cargar datos solarimetricos
#solar_1min_path = '/media/hecate/Seagate Backup Plus Drive/datasets/datasets_pkl/solarimetric/sol_radI_01min_s20170703_e20190907.pkl'
#solar_1min_dataset = pd.read_pickle(solar_1min_path)
#solar_1min_dataset = select_date_range(solar_1min_dataset, initial_date, final_date)

solar_5min_dataset = compact_database(sol_radG_01min_dataset, 5, use_average=True)
solar_5min_dataset = adjust_timestamps(solar_5min_dataset, 4*60)

# realizar clustering sobre los datos
cluster_labels = np.array( cluster_daily_radiation(solar_5min_dataset, plot_clusters=True) )
cluster_labels[np.where(cluster_labels==3)] = 2
num_labels = np.max(cluster_labels) + 1

#%% get sets information
assert len(dates)==len(cluster_labels)
i = 0
train_i = train_days
test_i = train_days + test_days

train_labels = []
test_labels = []

for i, label in enumerate(cluster_labels):
    if i < train_i:
        train_labels.append(label)
        
    elif (i >= train_i) and (i < test_i):
        test_labels.append(label)
    
    elif (i >= test_i):
        train_labels.append(label)
        train_i += train_days
        test_i += train_days + test_days
    
print('train set proportions:')
print('soleados: %f' %(train_labels.count(0)/len(train_labels)))
print('nublados: %f' %(train_labels.count(1)/len(train_labels)))
print('parcial: %f\n' %(train_labels.count(2)/len(train_labels)))

print('test set proportions:')
print('soleados: %f' %(test_labels.count(0)/len(test_labels)))
print('nublados: %f' %(test_labels.count(1)/len(test_labels)))
print('parcial: %f\n' %(test_labels.count(2)/len(test_labels)))  

#%% train AR model
from deepsolar.analytics import ar_fit

ar_param, _, _ = ar_fit(X_train, Y_train)

#%% evaluate model
from deepsolar import get_date_index

from deepsolar.analytics import clearsky_variability, beauchef_pers_predict
from deepsolar.analytics import ar_predict_v2, knn_predict
from deepsolar.analytics import forecast_pred

from deepsolar.analytics import mean_squared_error, mean_absolute_error, mean_bias_error
from deepsolar.analytics import skew_error, kurtosis_error, forecast_skill

from deepsolar.analytics import plot_error_dist, plot_error_variability
from deepsolar.analytics import plot_forecast_pred, plot_forecast_accuracy

from datetime import datetime, timedelta

# definir parámetros de evaluación
data_leap = 1

# inicializar cluster_metrics
metrics = ['mbe', 'mae', 'rmse', 'fs', 'std', 'skw', 'kts']
eval_df = pd.DataFrame(index=np.arange(num_labels), columns=metrics)

# para cada etiqueta resultante del clustering
for label in np.arange(num_labels):
    # obtener fechas correspondientes a la etiqueta
    cluster_dates = dates[cluster_labels==label]
    
    # escoger una fecha de muestra al azar
    cluster_sample = random.choice( list(cluster_dates) )
    
    # inicializar datos cluster
    cluster_true = []; cluster_pred = []
    cluster_pers = []; cluster_knn = []; cluster_ar = []
    cluster_var = []; cluster_time = []
    
    # por cada una de las fechas del cluster
    for date in cluster_dates:
        # obtener forecasting_times de testing
        timeleap = timestep*data_leap
        
        initial_hour = datetime.strptime(date, '%d-%m-%Y')
        hours = [initial_hour + timedelta(seconds=timeleap*i) for i in range(int(24*3600/timeleap))]
        pred_times = [datetime.strftime(h, date_format) for h in hours]
        
        # calcular predicción en cada una de las horas
        for pred_time in pred_times:
            try:
                # prediccion del modelo
                Y_true, Y_pred = forecast_pred(datasets=[scl_dataset, g16_dataset],
                                               output_name='Power',
                                               model=forecasting_model,
                                               pred_time=pred_time,
                                               img_sequence=1)
                
                # prediccion del persistence model
                Y_pers = beauchef_pers_predict(sys_dataset, pred_time, n_output)
                Y_pers = (Y_pers - sys_min)/(sys_max - sys_min)
                
                # prediccion del modelo knn
                date_index = get_date_index(sys_dataset['Timestamp'], get_date_index)
                start_index = date_index - n_input
                X_pred = scl_dataset.iloc[start_index:date_index]
                
                Y_knn = knn_predict(X=X_pred, k=12,
                                    X_train=X_train, Y_train=Y_train)
                
                # prediccion del modelo AR
                Y_ar = ar_predict_v2(X_pred, ar_param, n_output)
                
                # clearsky variability
                cs_var = clearsky_variability(sys_dataset, pred_time, n_input)
                
                # print progress
                print('\revaluating cluster ' + str(int(label)) + ': ' + pred_time, end='')
                
            except TypeError:
                continue
            
            # agregar datos a cluster data
            cluster_true.append(Y_true); cluster_pred.append(Y_pred)
            cluster_pers.append(Y_pers)
            cluster_knn.append(Y_knn); cluster_ar.append(Y_ar);
            cluster_time.append([pred_time]*n_output)
            cluster_var.append(cs_var)
    
    # calcular metricas del cluster
    Y_true = np.concatenate(cluster_true, axis=1)
    Y_pred = np.concatenate(cluster_pred, axis=1)
    
    Y_pers = np.concatenate(cluster_pers, axis=1)
    Y_knn = np.concatenate(cluster_knn, axis=1)
    Y_ar = np.concatenate(cluster_ar, axis=1)
    
    Y_time = np.array(cluster_time)
    cs_var = np.array(cluster_var)
    
    eval_df.at[label, 'mbe'] = mean_bias_error(Y_true, Y_pred)
    eval_df.at[label, 'mae'] = mean_absolute_error(Y_true, Y_pred)
    eval_df.at[label, 'rmse'] = np.sqrt(mean_squared_error(Y_true, Y_pred))
    eval_df.at[label, 'fs'] = forecast_skill(Y_true, Y_pred, Y_pers)
    eval_df.at[label, 'skw'] = skew_error(Y_true, Y_pred)
    eval_df.at[label, 'kts'] = kurtosis_error(Y_true, Y_pred)
    
    # plot cluster_sample
    plot_title = 'cluster '+ str(int(label))
    plot_forecast_pred(datasets=[scl_dataset, g16_dataset],
                       output_name='Power',
                       model=forecasting_model,
                       forecast_date=cluster_sample,
                       img_sequence=1,
                       title=plot_title)
    
    # plot gráfico estimación
    plot_forecast_accuracy(Y_true, Y_pred, timestamps=Y_time, title=plot_title, s=0.1)
    
    # plot gráficos de error
    horizons = [str( (h + 1)*timestep/60.0 ) + ' min' for h in range(n_output)]
    plot_error_dist(Y_true, [Y_pred, Y_pers, Y_ar, Y_knn],
                    ['ConvJANet', 'Pers', 'AR', 'KNN'],
                    horizons, bins=40, log=True, range=(-0.5,0.5))
    
    plot_error_variability(Y_true, [Y_pred, Y_pers, Y_ar, Y_knn], 
                           ['ConvJANet', 'Pers', 'AR', 'KNN'],
                           cs_var, horizons)


# guardar datos
save_path, _ = os.path.splitext(file_path)
save_path = save_path + '.csv'
eval_df.to_csv(save_path)

#%% forecast persistence
import pandas as pd
from deepsolar import get_date_index
from deepsolar.analytics import beauchef_pers_predict
import matplotlib.pyplot as plt
import numpy as np

# cargar datos del sistema
sys_dataset_path = 'C:\\Users\\Cristian\\Desktop\\BEAUCHEF PV FORECASTING\\datasets\\datasets_pkl\\processed\\dat_syst_30min_s20180827_e20190907.pkl'
sys_dataset = pd.read_pickle(sys_dataset_path)

forecast_date = '04-11-2018'
hours=['10:00','13:00','16:00','21:00']
#hours = ['13:00']
n_output = 6

# obtener timestamps de forecasting
pred_times = [forecast_date + ' ' + h for h in hours]
plot_axs = [(0, 0), (0, 1), (1, 0), (1, 1)]

# incializar plot
fig, axs = plt.subplots(2, 2)
fig.suptitle(forecast_date)

for test_time, ix in zip(pred_times, plot_axs):
    # obtener index correspondiente a forecasting_date
    date_index = get_date_index(sys_dataset['Timestamp'], test_time)
    
    # obtener ventana de tiempo
    start_index = date_index - 1
    end_index = date_index + n_output
    
    # checkear ventana de tiempo
    assert start_index >= 0
    assert end_index < sys_dataset.shape[0]
    
    # definir datos de input
    X = scl_dataset.loc[start_index, 'Power']
    T_X = sys_dataset.index[start_index:date_index]
    T_Y = sys_dataset.index[date_index:end_index]
    
    # aplicar modelo
    Y_pred = beauchef_pers_predict(sys_dataset, test_time, n_output)
    Y_pred = np.reshape(Y_pred, (n_output,-1))
    Y_pred = (Y_pred - sys_min)/(sys_max - sys_min)
    
    # datos base
    time_margin = 12
    start_index = max([0, start_index - time_margin])
    end_index = min([sys_dataset.shape[0], end_index + time_margin])
    T_data = sys_dataset.index[start_index:end_index]
    Y_true = scl_dataset['Power'].iloc[start_index:end_index]

    # plotear
    axs[ix[0], ix[1]].plot(T_data, Y_true, ls='--', c='k')
    
    axs[ix[0], ix[1]].plot(T_X, X, 'o-', c=[0.050383, 0.029803, 0.527975])
    axs[ix[0], ix[1]].plot(T_Y, Y_pred,'o-', c=[0.798216, 0.280197, 0.469538])
    
    axs[ix[0], ix[1]].set_ylabel('Potencia promedio kW')
    axs[ix[0], ix[1]].set_xlabel('data points')
    
    axs[ix[0], ix[1]].set_ylim([0.0, 1.0])
    axs[ix[0], ix[1]].set_ylim([0.0, 1.0])
    
#%% autoregresive
import pandas as pd
from deepsolar import get_date_index
from deepsolar.analytics import ar_predict_v2
import matplotlib.pyplot as plt
import numpy as np

# cargar datos del sistema
sys_dataset_path = 'C:\\Users\\Cristian\\Desktop\\BEAUCHEF PV FORECASTING\\datasets\\datasets_pkl\\processed\\dat_syst_30min_s20180827_e20190907.pkl'
sys_dataset = pd.read_pickle(sys_dataset_path)

forecast_date = '04-06-2019'
hours=['10:00','14:00','16:00','21:00']
#hours = ['13:00']
n_output = 6
n_input = 12

ar_param = np.array([[-0.21854125,  0.65082081, -0.63612383,  0.2976977 ,  0.11272937, -0.49361561,  0.41630267, -0.22443388, -0.13259425, -0.04567141, 0.01251299,  1.23463916]])

# obtener timestamps de forecasting
pred_times = [forecast_date + ' ' + h for h in hours]
plot_axs = [(0, 0), (0, 1), (1, 0), (1, 1)]

# incializar plot
fig, axs = plt.subplots(2, 2)
fig.suptitle(forecast_date)

for test_time, ix in zip(pred_times, plot_axs):
    # obtener index correspondiente a forecasting_date
    date_index = get_date_index(sys_dataset['Timestamp'], test_time)
    
    # obtener ventana de tiempo
    start_index = date_index - n_input
    end_index = date_index + n_output
    
    # checkear ventana de tiempo
    assert start_index >= 0
    assert end_index < sys_dataset.shape[0]
    
    # definir datos de input
    X = scl_dataset.iloc[start_index:date_index, 1]
    T_X = sys_dataset.index[start_index:date_index]
    T_Y = sys_dataset.index[date_index:end_index]
    
    # aplicar modelo
    Y_pred = ar_predict_v2(X, ar_param, n_output)
    Y_pred = np.reshape(Y_pred, (n_output,-1))
    
    # datos base
    time_margin = 12
    start_index = max([0, start_index - time_margin])
    end_index = min([sys_dataset.shape[0], end_index + time_margin])
    T_data = sys_dataset.index[start_index:end_index]
    Y_true = scl_dataset['Power'].iloc[start_index:end_index]

    # plotear
    axs[ix[0], ix[1]].plot(T_data, Y_true, ls='--', c='k')
    
    axs[ix[0], ix[1]].plot(T_X, X, 'o-', c=[0.050383, 0.029803, 0.527975])
    axs[ix[0], ix[1]].plot(T_Y, Y_pred,'o-', c=[0.798216, 0.280197, 0.469538])
    
    axs[ix[0], ix[1]].set_ylabel('Potencia promedio normalizada')
    axs[ix[0], ix[1]].set_xlabel('data points')
    
    axs[ix[0], ix[1]].set_ylim([0.0, 1.0])
    axs[ix[0], ix[1]].set_ylim([0.0, 1.0])
    
#%% knn
import pandas as pd
from deepsolar import get_date_index
from deepsolar.analytics import knn_predict
import matplotlib.pyplot as plt
import numpy as np

# cargar datos del sistema
sys_dataset_path = 'C:\\Users\\Cristian\\Desktop\\BEAUCHEF PV FORECASTING\\datasets\\datasets_pkl\\processed\\dat_syst_30min_s20180827_e20190907.pkl'
sys_dataset = pd.read_pickle(sys_dataset_path)

X_train_pwr = np.reshape(X_train[:,:,0], (X_train.shape[0], X_train.shape[1]))

forecast_date = '04-11-2018'
hours=['10:00','14:00','16:00','21:00']
#hours = ['13:00']
n_output = 6
n_input = 12

# obtener timestamps de forecasting
pred_times = [forecast_date + ' ' + h for h in hours]
plot_axs = [(0, 0), (0, 1), (1, 0), (1, 1)]

# incializar plot
fig, axs = plt.subplots(2, 2)
fig.suptitle(forecast_date)

for test_time, ix in zip(pred_times, plot_axs):
    # obtener index correspondiente a forecasting_date
    date_index = get_date_index(sys_dataset['Timestamp'], test_time)
    
    # obtener ventana de tiempo
    start_index = date_index - n_input
    end_index = date_index + n_output
    
    # checkear ventana de tiempo
    assert start_index >= 0
    assert end_index < sys_dataset.shape[0]
    
    # definir datos de input
    X = scl_dataset.iloc[start_index:date_index, 1]
    T_X = sys_dataset.index[start_index:date_index]
    T_Y = sys_dataset.index[date_index:end_index]
    
    # aplicar modelo
    Y_pred = knn_predict(X, 12, X_train_pwr, Y_train)
    Y_pred = np.reshape(Y_pred, (n_output,-1))
    
    # datos base
    time_margin = 12
    start_index = max([0, start_index - time_margin])
    end_index = min([sys_dataset.shape[0], end_index + time_margin])
    T_data = sys_dataset.index[start_index:end_index]
    Y_true = scl_dataset['Power'].iloc[start_index:end_index]

    # plotear
    axs[ix[0], ix[1]].plot(T_data, Y_true, ls='--', c='k')
    
    axs[ix[0], ix[1]].plot(T_X, X, 'o-', c=[0.050383, 0.029803, 0.527975])
    axs[ix[0], ix[1]].plot(T_Y, Y_pred,'o-', c=[0.798216, 0.280197, 0.469538])
    
    axs[ix[0], ix[1]].set_ylabel('Potencia promedio normalizada')
    axs[ix[0], ix[1]].set_xlabel('data points')
    
    axs[ix[0], ix[1]].set_ylim([0.0, 1.0])
    axs[ix[0], ix[1]].set_ylim([0.0, 1.0])