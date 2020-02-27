# -*- coding: utf-8 -*-

#%% load data -----------------------------------------------------------------
import pandas as pd

from solarpv.database import compact_database
from solarpv.database import select_date_range
from solarpv.database import adjust_timestamps
from solarpv.database import radiance_to_radiation

# -----------------------------------------------------------------------------
# cargar datos de potencia-SMA
sma_15min_path = 'C:\\Cristian\\datasets\\sma_system_15min_dataset.pkl'
power_dataset = pd.read_pickle(sma_15min_path)
power_dataset = select_date_range(power_dataset, '27-08-2018 04:15', '07-09-2019 00:00')

# compactar a base de 30min
power_dataset = compact_database(power_dataset, 2, use_average=True)
power_dataset = adjust_timestamps(power_dataset, 15*60)

# -----------------------------------------------------------------------------
# cargar datos de temperatura-SMA
temp_15min_path = 'C:\\Cristian\\datasets\\sma_temperature_15min_dataset.pkl'
temperature_dataset = pd.read_pickle(temp_15min_path)
temperature_dataset = select_date_range(temperature_dataset, '27-08-2018 04:15', '07-09-2019 00:00')

# compactar a base de 30min
temperature_dataset = compact_database(temperature_dataset, 2, use_average=True)
temperature_dataset = adjust_timestamps(temperature_dataset, 15*60)

# -----------------------------------------------------------------------------
# cargar datos solarimétricos
solar_1min_path = 'C:\\Cristian\\datasets\\solarimetric_1min_dataset.pkl'
solarimetric_dataset = pd.read_pickle(solar_1min_path)
solarimetric_dataset = select_date_range(solarimetric_dataset, '27-08-2018 04:01', '07-09-2019 00:00')

# compactar a base de 30min
solarimetric_dataset = compact_database(solarimetric_dataset, 30, use_average=True)
solarimetric_dataset = adjust_timestamps(solarimetric_dataset, 29*60)

# convertir a radiacion
solarimetric_dataset = radiance_to_radiation(solarimetric_dataset)

#%% analysis ------------------------------------------------------------------
from datetime import datetime, timedelta
from solarpv.analytics import plot_2D_radiation_data
from solarpv.analytics import plot_1D_radiation_data
from solarpv.analytics import plot_performance_ratio

plot_1D_radiation_data(power_dataset, 'Sistema', '04-11-2018', '05-11-2018', scale_factor=40)
plot_1D_radiation_data(solarimetric_dataset, 'Global', '04-11-2018', '05-11-2018', scale_factor=1)

#%% setup dataset -------------------------------------------------------------
import pandas as pd

# inicilizar dataset
colnames = ['Timestamp', 'Power', 'Inversor_1', 'Inversor_2', 'Inversor_3', 'Inversor_4', 'Module Temperature', 'External Temperature', 'Global', 'Diffuse', 'Direct']
dataset = pd.DataFrame(0.0, index=power_dataset.index, columns=colnames)

# asignar columna de timestamp y potencia
dataset['Timestamp'] = power_dataset['Timestamp']

dataset['Power'] = power_dataset['Sistema']
dataset['Inversor_1'] = power_dataset['SB 2500HF-30 997']
dataset['Inversor_2'] = power_dataset['SMC 5000A 493']
dataset['Inversor_3'] = power_dataset['SB 2500HF-30 273']
dataset['Inversor_4'] = power_dataset['SMC 5000A 434']

dataset['Module Temperature'] = temperature_dataset['Module']
dataset['External Temperature'] = solarimetric_dataset['Temperature']

dataset['Global'] = solarimetric_dataset['Global']
dataset['Diffuse'] = solarimetric_dataset['Diffuse']
dataset['Direct'] = solarimetric_dataset['Direct']

#%% persistence model
from solarpv.analytics import persistence_forecast

timestamp = '27-10-2018 15:30'
forecast = persistence_forecast(dataset, timestamp, 4, 30*60, ['Inversor_1', 'Inversor_2'])