#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#%% load data -----------------------------------------------------------------
import pandas as pd

# cargar imágenes satelitales
goes16_ds_path = '/media/hecate/Seagate Backup Plus Drive/datasets/goes16-30min-dataset.pkl'
goes16_dataset = pd.read_pickle(goes16_ds_path)

#%% encode dataset with model
import keras
from keras.models import load_model
from solarpv.database import encode_goes16_dataset

autoencoder_path = '/home/hecate/Desktop/models/test-01/AE_96.h5'
autoencoder = load_model(autoencoder_path)

encoded_dataset = encode_goes16_dataset(goes16_dataset, autoencoder, 'dense_9')

#%% save dataset
save_path = '/media/hecate/Seagate Backup Plus Drive/datasets/goes16-96encoded-dataset.pkl'
encoded_dataset.to_pickle(save_path)