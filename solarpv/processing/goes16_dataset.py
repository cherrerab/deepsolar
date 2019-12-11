# -*- coding: utf-8 -*-

#%% load data timestamps ------------------------------------------------------
import pandas as pd

from solarpv.database import compact_database
from solarpv.database import select_date_range
from solarpv.database import adjust_timestamps

# -----------------------------------------------------------------------------
# cargar datos de potencia-SMA
sma_15min_path = 'C:\\Cristian\\003. SMA DATASET\\005. 15 MINUTES SYSTEM DATA2\\sma-15min-dataset.pkl'
power_dataset = pd.read_pickle(sma_15min_path)
power_dataset = select_date_range(power_dataset, '27-08-2018 04:15', '07-09-2019 00:00')

# compactar a base de 30min
power_dataset = compact_database(power_dataset, 2, use_average=True)
power_dataset = adjust_timestamps(power_dataset, -15*60)

# -----------------------------------------------------------------------------
# obtener timestamps
data_timestamps = power_dataset['Timestamp'].values[10510:10610]

#%% generar goes-16 database --------------------------------------------------
from solarpv.database import goes16_dataset

# dirección de carpetas con imágenes satelilates
dir_paths = ['D:\\goes16_ABI-L1b-RadF_M3_C04', 'D:\\goes16_ABI-L1b-RadF_M6_C04']

# construir base de datos
goes_16_ds = goes16_dataset(dir_paths, data_timestamps, 100)


