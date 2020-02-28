# -*- coding: utf-8 -*-

#%% solarimetric 01min data
import os

from solarpv.database import solarimetric_dataset
from solarpv.database import radiance_to_radiation

# directorio donde se encuentran los archivos
dir_path = 'C:\\Cristian\\datasets_raw\\solarimetric\\solarimetric_01min_s20170703_e20190907'
data_file = 'CR1000_Beauchef_min_Corregido.csv'

data_path = os.path.join(dir_path, data_file)

# generar dataset
sol_radG_01min_dataset = solarimetric_dataset(data_path, 'CR1000_Beauchef_min_Corregido', 4, ['A','C','D','E','F'], keep_negatives=False)

sol_radI_01min_dataset = radiance_to_radiation(sol_radG_01min_dataset)

# guardar dataset
save_dir = 'C:\\Cristian\\datasets_pkl\\solarimetric'

save_path = os.path.join(save_dir, 'sol_radG_01min_s20170703_e20190907.pkl')
sol_radG_01min_dataset.to_pickle(save_path)

save_path = os.path.join(save_dir, 'sol_radI_01min_s20170703_e20190907.pkl')
sol_radI_01min_dataset.to_pickle(save_path)

#%% sma_system 15min data
import os

from solarpv.database import photovoltaic_dataset
from solarpv.database import correct_daylight_saving
from solarpv.database import adjust_timestamps

# directorio donde se encuentran los archivos
dir_path = 'C:\\Cristian\\datasets_raw\\sma\\sma_system_15min_s20180827_e20190923'

# generar dataset
sma_powr_15min_dataset = photovoltaic_dataset(dir_path, ['A','B','C','D','E','F'], all_equipments=True)

# corregir daylight saving time
sma_powr_15min_dataset = correct_daylight_saving(sma_powr_15min_dataset, '14-10-2018', '10-03-2019', positive=False)

# ajustar timezone del dataset a UTC (GMT -4)
sma_powr_15min_dataset = adjust_timestamps(sma_powr_15min_dataset, 4*3600)

# guardar dataset
save_dir = 'C:\\Cristian\\datasets_pkl\\sma'
save_path = os.path.join(save_dir, 'sma_powr_15min_s20180827_e20190923.pkl')
sma_powr_15min_dataset.to_pickle(save_path)

#%% sma_sensor 15min data
import os

from solarpv.database import temperature_dataset
from solarpv.database import correct_daylight_saving
from solarpv.database import adjust_timestamps

# directorio donde se encuentran los archivos
dir_path = 'C:\\Cristian\\datasets_raw\\sma\\sma_sensor_15min_s20180827_e20190923'

# generar dataset
sma_temp_15min_dataset = temperature_dataset(dir_path, ['A','B','C'])

# corregir daylight saving time
sma_temp_15min_dataset = correct_daylight_saving(sma_temp_15min_dataset, '14-10-2018', '10-03-2019', positive=False)

# ajustar timezone del dataset a UTC (GMT -4)
sma_temp_15min_dataset = adjust_timestamps(sma_temp_15min_dataset, 4*3600)

# guardar dataset
save_dir = 'C:\\Cristian\\datasets_pkl\\sma'
save_path = os.path.join(save_dir, 'sma_temp_15min_s20180827_e20190923.pkl')
sma_temp_15min_dataset.to_pickle(save_path)
