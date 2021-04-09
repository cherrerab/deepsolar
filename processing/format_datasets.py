# -*- coding: utf-8 -*-

#%% solarimetric 01min data
import os

from deepsolar.database import solarimetric_dataset
from deepsolar.database import radiance_to_radiation

# directorio donde se encuentran los archivos
dir_path = 'C:\\Users\\Cristian\\Desktop\\deepsolar\\workspace\\datasets\\raw\\solarimetric'
data_file = 'CR1000_20180801_20191018.csv'

data_path = os.path.join(dir_path, data_file)

# generar dataset
sol_radG_01min_dataset = solarimetric_dataset(data_path, 'Hoja1', 4, ['A','C','D','E','F'], keep_negatives=False)

sol_radI_01min_dataset = radiance_to_radiation(sol_radG_01min_dataset)

# guardar dataset
save_dir = 'C:\\Users\\Cristian\\Desktop\\deepsolar\\workspace\\datasets\\solarimetric'

save_path = os.path.join(save_dir, 'sol_radG_01min_s20180801_e20191018.pkl')
sol_radG_01min_dataset.to_pickle(save_path)

save_path = os.path.join(save_dir, 'sol_radI_01min_s20180801_e20191018.pkl')
sol_radI_01min_dataset.to_pickle(save_path)

#%% sma_system 15min data
import os

from deepsolar.database import photovoltaic_dataset
from deepsolar.database import correct_daylight_saving
from deepsolar.database import adjust_timestamps

# directorio donde se encuentran los archivos
dir_path = 'C:\\Users\\Cristian\\Desktop\\deepsolar\\workspace\\datasets\\raw\\photovoltaic\\system'

# generar dataset
sys_powr_15min_dataset = photovoltaic_dataset(dir_path, ['A','B','C','D','E','F'], all_equipments=True)

# corregir daylight saving time
sys_powr_15min_dataset = correct_daylight_saving(sys_powr_15min_dataset, '14-10-2018', '10-03-2019', positive=False)
sys_powr_15min_dataset = correct_daylight_saving(sys_powr_15min_dataset, '13-10-2019', '01-11-2019', positive=False)
# ajustar timezone del dataset a UTC (GMT -4)
sys_powr_15min_dataset = adjust_timestamps(sys_powr_15min_dataset, 4*3600)

# guardar dataset
save_dir = 'C:\\Users\\Cristian\\Desktop\\deepsolar\\workspace\\datasets\\photovoltaic'
save_path = os.path.join(save_dir, 'sys_powr_15min_s20180827_e20191018.pkl')
sys_powr_15min_dataset.to_pickle(save_path)

#%% sma_sensor 15min data
import os

from deepsolar.database import temperature_dataset
from deepsolar.database import correct_daylight_saving
from deepsolar.database import adjust_timestamps

# directorio donde se encuentran los archivos
dir_path = 'C:\\Users\\Cristian\\Desktop\\deepsolar\\workspace\\datasets\\raw\\photovoltaic\\sensor'

# generar dataset
sys_temp_15min_dataset = temperature_dataset(dir_path, ['A','B','C'])

# corregir daylight saving time
sys_temp_15min_dataset = correct_daylight_saving(sys_temp_15min_dataset, '14-10-2018', '10-03-2019', positive=False)
sys_temp_15min_dataset = correct_daylight_saving(sys_temp_15min_dataset, '13-10-2019', '01-11-2019', positive=False)

# ajustar timezone del dataset a UTC (GMT -4)
sys_temp_15min_dataset = adjust_timestamps(sys_temp_15min_dataset, 4*3600)

# guardar dataset
save_dir = 'C:\\Users\\Cristian\\Desktop\\deepsolar\\workspace\\datasets\\photovoltaic'
save_path = os.path.join(save_dir, 'sys_temp_15min_s20180827_e20191018.pkl')
sys_temp_15min_dataset.to_pickle(save_path)
