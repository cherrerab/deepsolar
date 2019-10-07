# -*- coding: utf-8 -*-

#%%
###############################################################################
########################## SOLARIMETRIC 1 MIN DATA ############################
###############################################################################
import os

from solarpv.database import solarimetric_dataset
from solarpv.database import radiance_to_radiation

from solarpv.analytics import plot_1D_radiation_data
from solarpv.analytics import plot_2D_radiation_data

# directorio donde se encuentran los archivos
dir_path = 'C:\\Users\\Cristian\\Desktop\\BEAUCHEF PV FORECASTING\\001. DATASETS\\001. SOLARIMETRIC DATA'
data_file = 'CR1000_Beauchef_min_Corregido.csv'

data_path = os.path.join(dir_path, data_file)

# generar dataset
solar_1min = solarimetric_dataset(data_path, 'CR1000_Beauchef_min_Corregido', 4, ['A','C','D','E','F'], keep_negatives=False)

solar_rad_1min = radiance_to_radiation(solar_1min)

# checkear resultado con día soleado
plot_1D_radiation_data(solar_rad_1min, 'Global', '04-11-2018', '05-11-2018')

# plotear dataset
plot_2D_radiation_data(solar_rad_1min, unit='Wh/m2', colname='Global', initial_date='29-08-2018',final_date='07-09-2019')

# guardar dataset
save_path = os.path.join(dir_path, 'solarimetric-1min-dataset.pkl')
solar_1min.to_pickle(save_path)

#%%
###############################################################################
########################### SMA 5 MIN DATA ####################################
###############################################################################
import os

from solarpv.database import photovoltaic_dataset
from solarpv.database import correct_daylight_saving
from solarpv.database import adjust_timestamps

from solarpv.analytics import plot_1D_radiation_data
from solarpv.analytics import plot_2D_radiation_data

# directorio donde se encuentran los archivos
dir_path = 'C:\\Users\\Cristian\\Desktop\\BEAUCHEF PV FORECASTING\\001. DATASETS\\003. SMA DATA\\002. 5 MINUTES SYSTEM DATA'

# generar dataset
sma_5min = photovoltaic_dataset(dir_path, ['A','B'])

# corregir daylight saving time
sma_5min = correct_daylight_saving(sma_5min, '01-01-2018', '14-10-2018')
sma_5min = correct_daylight_saving(sma_5min, '10-03-2019', '31-12-2019')

# ajustar timezone del dataset (GMT -3)
sma_5min = adjust_timestamps(sma_5min, 3*3600)

# checkear resultado con día soleado
plot_1D_radiation_data(sma_5min, 'Sistema', '04-11-2018', '05-11-2018')

# plotear dataset
plot_2D_radiation_data(sma_5min, unit='kW', colname='Sistema', initial_date='29-08-2018',final_date='07-09-2019')

# guardar dataset
save_path = os.path.join(dir_path, 'sma-5min-dataset.pkl')
sma_5min.to_pickle(save_path)

#%%
###############################################################################
########################### SMA 15 MIN DATA ####################################
###############################################################################
import os

from solarpv.database import photovoltaic_dataset
from solarpv.database import correct_daylight_saving
from solarpv.database import adjust_timestamps

from solarpv.analytics import plot_1D_radiation_data
from solarpv.analytics import plot_2D_radiation_data

# directorio donde se encuentran los archivos
dir_path = 'C:\\Users\\Cristian\\Desktop\\BEAUCHEF PV FORECASTING\\001. DATASETS\\003. SMA DATA\\005. 15 MINUTES SYSTEM DATA2'

# generar dataset
sma_15min = photovoltaic_dataset(dir_path, ['A','B','C','D','E','F'], all_equipments=True)

# corregir daylight saving time
sma_15min = correct_daylight_saving(sma_15min, '14-10-2018', '10-03-2019', positive=False)

# ajustar timezone del dataset (GMT -3)
sma_15min = adjust_timestamps(sma_15min, 4*3600)

# checkear resultado con día soleado
plot_1D_radiation_data(sma_15min, 'Sistema', '04-11-2018', '05-11-2018')

# plotear dataset
plot_2D_radiation_data(sma_15min, unit='kW', colname='Sistema', initial_date='27-08-2018',final_date='23-09-2019')

# guardar dataset
save_path = os.path.join(dir_path, 'sma-15min-dataset.pkl')
sma_15min.to_pickle(save_path)

#%%
###############################################################################
########################### SMA HOURLY DATA ###################################
###############################################################################
import os

from solarpv.database import photovoltaic_dataset
from solarpv.database import correct_daylight_saving
from solarpv.database import adjust_timestamps

from solarpv.analytics import plot_1D_radiation_data
from solarpv.analytics import plot_2D_radiation_data

# directorio donde se encuentran los archivos
dir_path = 'C:\\Users\\Cristian\\Desktop\\BEAUCHEF PV FORECASTING\\001. DATASETS\\003. SMA DATA\\001. HOURLY SYSTEM DATA'

# generar dataset
sma_hour = photovoltaic_dataset(dir_path, ['A','B','C','D','E','F'], all_equipments=True)

# corregir daylight saving time
sma_hour = correct_daylight_saving(sma_hour, '27-10-2017', '10-03-2018', positive=False)

# ajustar timezone del dataset (GMT -3)
sma_hour = adjust_timestamps(sma_hour, 3*3600)

# plotear dataset
plot_2D_radiation_data(sma_hour, unit='kW', colname='Sistema', initial_date='2016-04-12',final_date='2018-08-26')

# guardar dataset
save_path = os.path.join(dir_path, 'sma-hourly-dataset.pkl')
sma_hour.to_pickle(save_path)

