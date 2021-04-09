# -*- coding: utf-8 -*-

#%% load datasets
import pandas as pd

from deepsolar.database import compact_database
from deepsolar.database import select_date_range
from deepsolar.database import adjust_timestamps
from deepsolar.database import radiance_to_radiation

# -----------------------------------------------------------------------------
# sma_system 15min data
sys_powr_15min_path = 'C:\\Users\\Cristian\\Desktop\\deepsolar\\workspace\\datasets\\photovoltaic\\sys_powr_15min_s20180827_e20191018.pkl'
sys_powr_15min_dataset = pd.read_pickle(sys_powr_15min_path)
sys_powr_15min_dataset = select_date_range(sys_powr_15min_dataset, '27-08-2018 04:15', '19-10-2019 00:00')

# compactar a base de 30min
sys_powr_30min_dataset = compact_database(sys_powr_15min_dataset, 2, use_average=True)
sys_powr_30min_dataset = adjust_timestamps(sys_powr_30min_dataset, 15*60)

# -----------------------------------------------------------------------------
# sma_sensor 15min data
sys_temp_15min_path = 'C:\\Users\\Cristian\\Desktop\\deepsolar\\workspace\\datasets\\photovoltaic\\sys_temp_15min_s20180827_e20191018.pkl'
sys_temp_15min_dataset = pd.read_pickle(sys_temp_15min_path)
sys_temp_15min_dataset = select_date_range(sys_temp_15min_dataset, '27-08-2018 04:15', '19-10-2019 00:00')

# compactar a base de 30min
sys_temp_30min_dataset = compact_database(sys_temp_15min_dataset, 2, use_average=True)
sys_temp_30min_dataset = adjust_timestamps(sys_temp_30min_dataset, 15*60)
#%%
# -----------------------------------------------------------------------------
# cargar datos solarimétricos
sol_radG_01min_path = 'C:\\Users\\Cristian\\Desktop\\deepsolar\\workspace\\datasets\\solarimetric\\sol_radG_01min_s20180801_e20191018.pkl'
sol_radG_01min_dataset = pd.read_pickle(sol_radG_01min_path)
sol_radG_01min_dataset = select_date_range(sol_radG_01min_dataset, '27-08-2018 04:01', '19-10-2019 00:00')

sol_radI_01min_path = 'C:\\Users\\Cristian\\Desktop\\deepsolar\\workspace\\datasets\\solarimetric\\sol_radI_01min_s20180801_e20191018.pkl'
sol_radI_01min_dataset = pd.read_pickle(sol_radI_01min_path)
sol_radI_01min_dataset = select_date_range(sol_radI_01min_dataset, '27-08-2018 04:01', '19-10-2019 00:00')

# compactar a base de 15min
sol_radG_15min_dataset = compact_database(sol_radG_01min_dataset, 15, use_average=True)
sol_radG_15min_dataset = adjust_timestamps(sol_radG_15min_dataset, 14*60)

sol_radI_15min_dataset = compact_database(sol_radI_01min_dataset, 15, use_average=False)
sol_radI_15min_dataset = adjust_timestamps(sol_radI_15min_dataset, 14*60)

# compactar a base de 30min
sol_radG_30min_dataset = compact_database(sol_radG_01min_dataset, 30, use_average=True)
sol_radG_30min_dataset = adjust_timestamps(sol_radG_30min_dataset, 29*60)

sol_radI_30min_dataset = compact_database(sol_radI_01min_dataset, 30, use_average=False)
sol_radI_30min_dataset = adjust_timestamps(sol_radI_30min_dataset, 29*30)

#%% setup 15min dataset
from datetime import datetime
import pandas as pd
import numpy as np

# inicilizar dataset
colnames = ['Timestamp', 'Power', 'Inversor 1', 'Inversor 2', 'Inversor 3', 'Inversor 4',
            'Module Temperature', 'External Temperature',
            'Global', 'Diffuse', 'Direct',
            'n_day', 'min_day', 'clean_day']

dataset_15min = pd.DataFrame(0.0, index=sys_powr_15min_dataset.index, columns=colnames)

# asignar columna de timestamp y potencia
dataset_15min['Timestamp'] = sys_powr_15min_dataset['Timestamp']

dataset_15min['Power'] = sys_powr_15min_dataset['Sistema']
dataset_15min['Inversor 1'] = sys_powr_15min_dataset['SB 2500HF-30 997']
dataset_15min['Inversor 2'] = sys_powr_15min_dataset['SMC 5000A 493']
dataset_15min['Inversor 3'] = sys_powr_15min_dataset['SB 2500HF-30 273']
dataset_15min['Inversor 4'] = sys_powr_15min_dataset['SMC 5000A 434']

dataset_15min['Module Temperature'] = sys_temp_15min_dataset['Module']
dataset_15min['External Temperature'] = sol_radG_15min_dataset['Temperature']

dataset_15min['Global'] = sol_radI_15min_dataset['Global']
dataset_15min['Diffuse'] = sol_radI_15min_dataset['Diffuse']
dataset_15min['Direct'] = sol_radI_15min_dataset['Direct']

# -----------------------------------------------------------------------------
# generar resto columnas
date_format = '%d-%m-%Y %H:%M'

# dias en que se realiza limpieza
clean_dates = ['27-08-2018', '05-03-2019']
clean_dates = [datetime.strptime(c, '%d-%m-%Y') for c in clean_dates]

for i in dataset_15min.index:
    # obtener timestamp del dato
    timestamp = dataset_15min.at[i, 'Timestamp']
    timestamp = datetime.strptime(timestamp, date_format)
    
    date_tt = timestamp.timetuple()
    
    # calcular minuto del día y día en el año
    min_day = date_tt.tm_hour*60.0 + date_tt.tm_min
    n_day = date_tt.tm_yday
    
    # obtener dia de limpieza anterior
    clean_idx = np.searchsorted(clean_dates, timestamp)
    clean_day = timestamp - clean_dates[clean_idx-1]
    
    # asignar al dataset
    dataset_15min.at[i, 'n_day'] = n_day
    dataset_15min.at[i, 'min_day'] = min_day
    dataset_15min.at[i, 'clean_day'] = clean_day.days
    
#%% save 15min dataset
import os

save_dir = 'C:\\Users\\Cristian\\Desktop\\deepsolar\\workspace\\datasets\\training'

save_path = os.path.join(save_dir, 'dat_syst_15min_s20180827_e20191018.pkl')
dataset_15min.to_pickle(save_path)


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
#%% setup 15min dataset
from datetime import datetime
import pandas as pd
import numpy as np

# inicilizar dataset
colnames = ['Timestamp', 'Power', 'Inversor 1', 'Inversor 2', 'Inversor 3', 'Inversor 4',
            'Module Temperature', 'External Temperature',
            'Global', 'Diffuse', 'Direct',
            'n_day', 'min_day', 'clean_day']

dataset_30min = pd.DataFrame(0.0, index=sys_powr_30min_dataset.index, columns=colnames)

# asignar columna de timestamp y potencia
dataset_30min['Timestamp'] = sys_powr_30min_dataset['Timestamp']

dataset_30min['Power'] = sys_powr_30min_dataset['Sistema']
dataset_30min['Inversor 1'] = sys_powr_30min_dataset['SB 2500HF-30 997']
dataset_30min['Inversor 2'] = sys_powr_30min_dataset['SMC 5000A 493']
dataset_30min['Inversor 3'] = sys_powr_30min_dataset['SB 2500HF-30 273']
dataset_30min['Inversor 4'] = sys_powr_30min_dataset['SMC 5000A 434']

dataset_30min['Module Temperature'] = sys_temp_30min_dataset['Module']
dataset_30min['External Temperature'] = sol_radG_30min_dataset['Temperature']

dataset_30min['Global'] = sol_radI_30min_dataset['Global']
dataset_30min['Diffuse'] = sol_radI_30min_dataset['Diffuse']
dataset_30min['Direct'] = sol_radI_30min_dataset['Direct']

# -----------------------------------------------------------------------------
# generar resto columnas
date_format = '%d-%m-%Y %H:%M'

# dias en que se realiza limpieza
clean_dates = ['27-08-2018', '05-03-2019']
clean_dates = [datetime.strptime(c, '%d-%m-%Y') for c in clean_dates]

for i in dataset_30min.index:
    # obtener timestamp del dato
    timestamp = dataset_30min.at[i, 'Timestamp']
    timestamp = datetime.strptime(timestamp, date_format)
    
    date_tt = timestamp.timetuple()
    
    # calcular minuto del día y día en el año
    min_day = date_tt.tm_hour*60.0 + date_tt.tm_min
    n_day = date_tt.tm_yday
    
    # obtener dia de limpieza anterior
    clean_idx = np.searchsorted(clean_dates, timestamp)
    clean_day = timestamp - clean_dates[clean_idx-1]
    
    # asignar al dataset
    dataset_30min.at[i, 'n_day'] = n_day
    dataset_30min.at[i, 'min_day'] = min_day
    dataset_30min.at[i, 'clean_day'] = clean_day.days
    
#%% save 15min dataset
import os

save_dir = 'C:\\Users\\Cristian\\Desktop\\deepsolar\\workspace\\datasets\\training'

save_path = os.path.join(save_dir, 'dat_syst_30min_s20180827_e20191018.pkl')
dataset_30min.to_pickle(save_path)

