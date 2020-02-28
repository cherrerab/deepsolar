# -*- coding: utf-8 -*-
#%% load data -----------------------------------------------------------------
import pandas as pd

from solarpv.database import compact_database
from solarpv.database import select_date_range
from solarpv.database import adjust_timestamps
from solarpv.database import radiance_to_radiation

# -----------------------------------------------------------------------------
# cargar datos del sistema
dat_syst_30min_path = 'C:\\Cristian\\datasets_pkl\\processed\\dat_syst_15min_s20180827_e20190907.pkl'
dat_syst_30min_dataset =  pd.read_pickle(dat_syst_30min_path)

#%% analysis ------------------------------------------------------------------
from datetime import datetime, timedelta
from solarpv.analytics import plot_2D_radiation_data
from solarpv.analytics import plot_1D_radiation_data
from solarpv.analytics import plot_performance_ratio

plot_1D_radiation_data(dat_syst_30min_dataset, 'Power', '04-11-2018', '05-11-2018', scale_factor=40,  Bs=30.0, Zs=-90.0)
plot_1D_radiation_data(dat_syst_30min_dataset, 'Global', '04-11-2018', '05-11-2018', scale_factor=1, Bs=30.0, Zs=-90.0)

#%% persistence model
from solarpv.analytics import persistence_forecast

timestamp = '04-11-2018 13:45'
forecast = persistence_forecast(dat_syst_30min_dataset, timestamp, 4, ['Inversor 1', 'Inversor 2'], Bs=20.0, Zs=-90.0, verbose=True)
