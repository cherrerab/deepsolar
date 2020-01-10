# -*- coding: utf-8 -*-
#%% load data -----------------------------------------------------------------
import pandas as pd

# cargar imágenes satelitales
goes16_ds_path = '/media/hecate/Seagate Backup Plus Drive/datasets/goes16-30min-dataset.pkl'
goes16_dataset = pd.read_pickle(goes16_ds_path)

#%% setup dataset -------------------------------------------------------------
import numpy as np
from sklearn.model_selection import train_test_split

# obtener únicamente los datos de imágenes
X_flat = goes16_dataset.iloc[:, 1:].values
del goes16_dataset

# generar sets de training y testing
X_train_flat, X_test_flat ,_,_ = train_test_split(X_flat, X_flat, test_size = 0.25, random_state = 24)

# reshape de los sets
X_train = np.nan_to_num( np.reshape(X_train_flat,[-1, 128, 128, 1]) )
X_test = np.nan_to_num( np.reshape(X_test_flat,[-1, 128, 128, 1]) )

# normalizar
max_value = np.max(X_train, axis=None)
min_value = np.min(X_train, axis=None)

X_train = (X_train - min_value)/(max_value - min_value)
X_test = (X_test - min_value)/(max_value - min_value)

# eliminar datos innecesarios
del X_flat
del X_train_flat
del X_test_flat

#%% set autoencoder model -----------------------------------------------------
from keras.models import Model
from keras.callbacks import EarlyStopping

from keras.layers import Input
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Reshape
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import MaxPooling2D
from keras.layers import UpSampling2D
from keras.layers import Dropout
from keras.layers import BatchNormalization

import matplotlib.pyplot as plt


# inicializar serie de encoding dimensions a probar
encoding_dimensions = [512, 384, 256, 192, 128, 96, 64]
encoding_dimensions = [512]

#encoding_dimensions = [512]
num_tests = len(encoding_dimensions)

autoencoder_results = pd.DataFrame(index=np.arange(num_tests),
                                   columns=['encoding_dim', 'MSE'])

for i, encoding_dim in enumerate(encoding_dimensions):
    
    # definimos el tamaño de input de nuestro modelo y la del encoder
    input_dim = (128, 128)
    
    # inicializamos la red AE añadiendo la capa de input
    input_layer = Input( shape=(input_dim[0], input_dim[1], 1) )
    
    # añadimos las capas de encoding ------------------------------------------
    encoder = Conv2D( 96, (3, 3), activation='relu', padding='same' )(input_layer)
    encoder = MaxPooling2D( (2, 2) )(encoder)
    
    encoder = Dropout(0.1)(encoder)
    encoder = Conv2D( 64, (3, 3), activation='relu', padding='same' )(encoder)
    encoder = MaxPooling2D( (2, 2) )(encoder)
    
    encoder = Dropout(0.1)(encoder)
    encoder = Conv2D( 64, (5, 5), activation='relu', padding='same' )(encoder)
    
    # añadimos las capas del espacio latente ----------------------------------
    encoder = Flatten()(encoder)
    
    latent_space = Dense(encoding_dim, activation='softmax')(encoder)
    
    decoder = Dense(32*32*64, activation='relu')(latent_space)
    decoder = Reshape( (32, 32, 64) )(decoder)
    
    # añadimos las capas de decoding correspondientes -------------------------
    decoder = Conv2DTranspose( 64, (5, 5), activation='relu', padding='same')(decoder)
    decoder = UpSampling2D( (2, 2) )(decoder)
    decoder = BatchNormalization()(decoder)
    
    decoder = Conv2DTranspose( 64, (3, 3), activation='relu', padding='same')(decoder)
    decoder = UpSampling2D( (2, 2) )(decoder)
    decoder = BatchNormalization()(decoder)
    
    decoder = Conv2DTranspose( 96, (3, 3), activation='relu', padding='same')(decoder)
    
    decoded_layer = Conv2D(1 , (3, 3), activation='sigmoid', padding='same')(decoder)
    
    # generamos el autoencoder
    autoencoder = Model(inputs = input_layer, outputs = decoded_layer)
    
    # compilamos el modelo incializando sus optimizadores
    autoencoder.compile(optimizer = 'adam', loss = 'mse')
    early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)
    
    # training
    model_history = autoencoder.fit(X_train, X_train, batch_size=128, epochs=25, 
                                    verbose=1, validation_data=(X_test, X_test))

    # training evaluation -----------------------------------------------------
    test_mse = autoencoder.evaluate(X_test, X_test, batch_size = 512)
    
    autoencoder_results.at[i, 'encoding_dim'] = encoding_dim
    autoencoder_results.at[i, 'MSE'] = test_mse
    
print(autoencoder_results)

#%% evaluation
import matplotlib.pyplot as plt

X_eval = np.reshape(X_test[100,:,:], (1, 128, 128, 1))
X_pred = autoencoder.predict(X_eval)

X_pred = np.reshape(X_pred, (128, 128))
X_eval = np.reshape(X_eval, (128, 128))

plt.figure()
plt.imshow(X_eval)

plt.figure()
plt.imshow(X_pred)