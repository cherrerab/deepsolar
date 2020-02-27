# -*- coding: utf-8 -*-
from solarpv.analytics import cluster_daily_radiation
import numpy as np
import random

#%% load solar data -----------------------------------------------------------
import pandas as pd
from datetime import datetime, timedelta

from solarpv.database import compact_database
from solarpv.database import select_date_range
from solarpv.database import adjust_timestamps
from solarpv.database import radiance_to_radiation

from solarpv.analytics import plot_1D_radiation_data
from solarpv.analytics import plot_goes16_secuence

# cargar datos solarimetricos
solar_1min_path = 'C:\\Cristian\\datasets\\solarimetric_1min_dataset.pkl'
solar_1min_dataset = pd.read_pickle(solar_1min_path)
solar_1min_dataset = select_date_range(solar_1min_dataset, '28-08-2018 00:00', '07-09-2019 00:00')
solar_1min_dataset = radiance_to_radiation(solar_1min_dataset)
solar_5min_dataset = compact_database(solar_1min_dataset, 5, use_average=True)

# cargar imágenes satelitales
goes16_ds_path = '/media/hecate/Seagate Backup Plus Drive/datasets/goes16-30min-48px-dataset.pkl'
goes16_dataset = pd.read_pickle(goes16_ds_path)
goes16_dataset = select_date_range(goes16_dataset, '27-08-2018 04:00', '07-09-2019 00:00')

#%% obtener lista de días -------------------------------------------------------
date_format = '%d-%m-%Y %H:%M'
timestamps = solar_5min_dataset['Timestamp'].values

initial_date = datetime.strptime(timestamps[0], date_format)
final_date = datetime.strptime(timestamps[-1], date_format)

dates = pd.date_range(initial_date, final_date, freq='1D')
dates = dates.strftime('%d-%m-%Y')

# realizar clustering sobre los datos -----------------------------------------
cluster_labels = np.array( cluster_daily_radiation(solar_5min_dataset, plot_clusters=False) )
num_labels = np.max(cluster_labels) + 1

# para cada etiqueta resultante del clustering
for label in np.arange(num_labels):
        
    # obtener fechas correspondientes a la etiqueta
    cluster_dates = dates[cluster_labels==label]
    # escoger una fecha de muestra al azar
    cluster_sample = random.choice( list(cluster_dates) )
    
    # plotear
    start_date = cluster_sample + ' 9:00'
    end_date = cluster_sample + ' 21:30'
    
    plot_1D_radiation_data(solar_5min_dataset, 'Global', start_date, end_date)
    plot_goes16_secuence(goes16_dataset, start_date, end_date)
    
    
    