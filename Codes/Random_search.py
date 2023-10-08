import tensorflow as tf
import h5py
from tensorflow import keras
from keras.constraints import max_norm
import numpy as np

#fix random see for reproducibility
np.random.seed(42)

#dataset
# Open the HDF5 file in read mode
with h5py.File(r'C:\Users\My PC\Documents\DATA ANALYSIS\LGBTQ project\LGBTQ_project2\Check\Check_6000_W2V.h5') as hf:
    X_train = hf['X_train'][:]
    Y_train = hf['Y_train'][:]
    X_test = hf['X_test'][:]
    Y_test = hf['Y_test'][:]
    
#print((X_train.shape, Y_train.shape), (X_test.shape, Y_test.shape))

num_classes = 3
Y_train_coded = keras.utils.to_categorical(Y_train, num_classes = 3)

from sklearn.model_selection import train_test_split
X_rms, X_not_use, Y_rms, Y_not_use = train_test_split(X_train, Y_train_coded,  test_size = 0.8, random_state=42)
#print((X_rms.shape, Y_rms.shape), (X_not_use.shape, Y_not_use.shape))

#input_dim = X_rms.shape[1]
#print(input_dim)

def define_model(hp):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.InputLayer(input_shape=(300, 200)))  # Input layer for sequences
        
        hp_learning_rate = hp.Choice('learning_rate', values=[0.0001, 0.001, 0.01, 0.1])

        model.add(tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(64, activation='relu', return_sequences=True)
        ))

        model.add(tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(64, activation='relu', return_sequences=True)
        ))

        model.add(tf.keras.layers.Dropout(0.3))
        model.add(tf.keras.layers.Flatten())

        model.add(tf.keras.layers.Dense(3, activation='softmax'))

        model.compile(
            optimizer=tf.keras.optimizers.Adadelta(learning_rate = hp_learning_rate),
            loss= 'CategoricalCrossentropy',
            metrics = ['accuracy']
            )
        return model

import keras_tuner as kt

tuner = kt.Hyperband(define_model,
                     objective = 'val_accuracy',
                     max_epochs = 3,
                     factor = 3,
                     directory = 'dir',
                     project_name = 'LGBTQAdadelta_W2V')

tuner.search(X_rms, Y_rms, epochs = 3, validation_split = 0.2)

print(tuner.get_best_hyperparameters()[0].values)

#hp_units_lstm = hp.Int('units_lstm', min_value=32, max_value=256, step=32)  # Define units for LSTM layer
    #hp_units_dense = hp.Int('units_dense', min_value=32, max_value=256, step=32)  # Define units for Dense layers



