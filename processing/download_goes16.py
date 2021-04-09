# -*- coding: utf-8 -*-

"""
Descargar imágenes satelitales de la noaa.

1. Revisar en la carpeta en donde se van a descargar el timestamp del archivo 
   más reciente (se pueden ordenar por nombre).
   por ejemplo, en la carpeta goes16_ABI-L1b-RadF_M6_C04, el último archivo es:
       OR_ABI-L1b-RadF-M6C04_G16_s20192582350177_e20192582359485_c20192582359540
   donde el tag de creación es c20192582359540. Esto significa que el archivo 
   fue creado el 2019 258 23:59:54.0.
2. En la página https://www.calendario-365.es/numeros-de-dias/2019.html
   se puede revisar a que fecha corresponde el número de día del tag.
   En el caso anterior, 258 corresponde al 15-09-2019.
   
3. Modificar los parámetros start_time y end_time para la descarga.
   Recomiendo que start_time sea siempre dos horas antes del archivo más reciente.
   (los archivos simplemente se sobreescribirán).
   
   Por ejemplo en este caso, recomendaría (formato %d-%m-%Y %H:%M):
       start_time = '15-09-2019 21:59'
       
   end_time puede ser cualquier fecha posterior, pero recomiendo correr cada un
   par de semanas o quizás un mes.'
       

"""

from deepsolar.noaa import download_goes16_data_v2

# ubicación de la carpeta donde se guardarán los datasets descargados
save_path = 'C:\\Users\\Cristian\\Downloads'

# fecha en formato %d-%m-%Y %H:%M desde la que comienza el periodo.
start_time = '14-12-2020 16:00'

# fecha en formato %d-%m-%Y %H:%M en la que termina el periodo.
end_time = '14-12-2020 16:20'

# download
dwnl = download_goes16_data_v2(save_path, start_time, end_time, channel='C04')



