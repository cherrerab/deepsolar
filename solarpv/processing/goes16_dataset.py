# -*- coding: utf-8 -*-

#%% load data timestamps ------------------------------------------------------
import pandas as pd

from solarpv.database import compact_database
from solarpv.database import select_date_range
from solarpv.database import adjust_timestamps

# cargar system_dataset
dat_syst_15min_path = 'C:\\Cristian\\datasets_pkl\\processed\\dat_syst_15min_s20180827_e20190907.pkl'
dat_syst_15min_dataset = pd.read_pickle(dat_syst_15min_path)

# obtener timestamps
data_timestamps = dat_syst_15min_dataset['Timestamp']

#%% generar goes-16 database --------------------------------------------------
from solarpv.database import goes16_dataset
from solarpv.database import show_goes16_dataset
import os

# dirección de carpetas con imágenes satelilates
dir_paths = ['/media/hecate/Seagate Backup Plus Drive/goes16_ABI-L1b-RadF_M3_C04', '/media/hecate/Seagate Backup Plus Drive/goes16_ABI-L1b-RadF_M6_C04']

# construir base de datos
dat_noaa_15min_dataset = goes16_dataset(dir_paths, data_timestamps, 48)

# guardar dataset
save_dir = '/media/hecate/Seagate Backup Plus Drive/datasets_pkl/processed'
save_path = os.path.join(save_dir, 'dat_noaa_15min_s20180827_e20190907.pkl')
dat_noaa_15min_dataset.to_pickle(save_path)

