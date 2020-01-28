# -*- coding: utf-8 -*-

#%% load data timestamps ------------------------------------------------------
import pandas as pd

from solarpv.database import compact_database
from solarpv.database import select_date_range
from solarpv.database import adjust_timestamps

# -----------------------------------------------------------------------------
# cargar datos de potencia-SMA
sma_15min_path = '/media/hecate/Seagate Backup Plus Drive/datasets/system-power-15min-dataset.pkl'
power_dataset = pd.read_pickle(sma_15min_path)
power_dataset = select_date_range(power_dataset, '27-08-2018 04:15', '07-09-2019 00:00')

# compactar a base de 30min
power_dataset = compact_database(power_dataset, 2, use_average=True)
power_dataset = adjust_timestamps(power_dataset, -15*60)

# -----------------------------------------------------------------------------
# obtener timestamps
data_timestamps = power_dataset['Timestamp']

#%% generar goes-16 database --------------------------------------------------
from solarpv.database import goes16_dataset
from solarpv.database import show_goes16_dataset
import os

# dirección de carpetas con imágenes satelilates
dir_paths = ['/media/hecate/Seagate Backup Plus Drive/goes16_ABI-L1b-RadF_M3_C04', '/media/hecate/Seagate Backup Plus Drive/goes16_ABI-L1b-RadF_M6_C04']

# construir base de datos
goes_16_ds = goes16_dataset(dir_paths, data_timestamps, 48)

# guardar dataset
save_dir = '/media/hecate/Seagate Backup Plus Drive/datasets'
save_path = os.path.join(save_dir, 'goes16-30min-48px-dataset.pkl')
goes_16_ds.to_pickle(save_path)

