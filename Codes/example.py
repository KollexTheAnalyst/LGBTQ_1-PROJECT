import tensorflow as tf
import keras
from keras.layers import Embedding, Bidirectional, LSTM, Dense
from keras.models import Sequential
from keras.layers import Dropout
from keras.callbacks import LearningRateScheduler
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import RandomizedSearchCV
from keras.constraints import max_norm
from keras.optimizers import Adam
import h5py
import os

LGBTQ = pd.read_csv(r'C:\Users\My PC\Documents\DATA ANALYSIS\LGBTQ project\LGBTQ_project2\sample_ext_lgbtq_datasets.csv')

#fix random see for reproducibility
np.random.seed(42)

#dataset
# Open the HDF5 file in read mode
with h5py.File('data.h5', 'r') as hf:
    X_train = hf['X_train'][:]
    Y_train = hf['Y_train'][:]

num_classes = 3

Y_train_coded = keras.utils.to_categorical(Y_train, num_classes = 3)

#Take a subset of train for RandomSearch, we take 20%
from sklearn.model_selection import train_test_split
X_rms, X_not_use, Y_rms, Y_not_use = train_test_split(X_train, Y_train_coded,  test_size = 0.8, random_state=42)

from sklearn.utils import class_weight
from sklearn.utils.class_weight import compute_class_weight
class_weights = list(class_weight.compute_class_weight('balanced',  classes = np.unique(LGBTQ['label']), y = LGBTQ['label']))

frequencies = pd.value_counts(LGBTQ['label'])
#print(frequencies)

class_weights.sort()

#print(class_weights)

weights = {}

for index, weight in enumerate(class_weights):
    weights[index] = weight

#Build the model
input_dim = X_rms.shape[1]
#print(input_dim)

'Note: Add default optimizer, otherwise throws error optimizer not legal'
def define_model(learning_rate=0.01, momentum = 0.1):
        model = tf.keras.Sequential([])

        model.add(tf.keras.layers.InputLayer(input_shape=(370, 100)))  # Input layer for sequences
        for units in [128,128,64,32]:
            model.add(tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(units, activation='relu', return_sequences=True)
        ))
        model.add(tf.keras.layers.Dropout(0.2))
        model.add(tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(units, activation='relu', return_sequences=True)
        ))
        model.add(tf.keras.layers.Dropout(0.3))
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(3, activation='softmax'))

        cp = tf.keras.callbacks.ModelCheckpoint('Model_1/', save_best_only = True)

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate = learning_rate),
            loss= tf.keras.losses.CategoricalCrossentropy(from_logits=True),
            metrics = ['accuracy', tf.keras.metrics.AUC(name = 'auc')]
            )
        return model
#==============================================================================
import keras_tuner as kt

tuner = kt.Hyperband(define_model,
                     objective = 'val_accuracy',
                     max_epochs = 5,
                     factor = 3,
                     directory = 'dir',
                     project_name = 'LGBTQadadelta1')

learning_rate = [0.0001, 0.001, 0.01, 0.1]
momentum = [0.3, 0.5, 0.7, 0.9]

param_random_lstm = dict(learning_rate = learning_rate, momentum = momentum)
LSTM_RandomGrid = RandomizedSearchCV(estimator = tuner, param_distributions = param_random_lstm, cv = 10, verbose=21, n_jobs = 8)

random_result = LSTM_RandomGrid.fit(X_rms, Y_rms)

#summarize results
print("Best: %f using %s" % (random_result.best_score_, random_result.best_params_))

means= random_result.cv_results_['mean_test_score']
stds = random_result.cv_results_['std_test_score']
params = random_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
        print("Mean = %f (std=%f) with: %r" % (mean, stdev, param))

#===========================================================
# Create an HDF5 file



    #split data for training
train_df = LGBTQ.sample(frac=1, random_state = 1)
train_df.reset_index(drop=True, inplace=True)

split_index_1 = int(len(LGBTQ) * 0.7)
split_index_2 = int(len(LGBTQ) * 0.85)

train_df, val_df, test_df = train_df[:split_index_1], train_df[split_index_1:split_index_2], train_df[split_index_2:]
#print(len(train_df), len(val_df), len(test_df))
#=======================================================================================
def df_to_X_Y(dff):
    y = dff['label'].to_numpy().astype(int)

    #sequences = comment_vec_df.values.tolist()

    #return sequences, y
#Check
X_train_W2V_NN, Y_train_W2V_NN = df_to_X_Y(train_df)
X_test_W2V_NN, Y_test_W2V_NN = df_to_X_Y(test_df)
X_val_W2V_NN, Y_val_W2V_NN = df_to_X_Y(val_df)

print(len(X_train_W2V_NN), len(X_train_W2V_NN[0]))

#===========================================================
def df_to_X_Y(dff):
    y = dff['label'].to_numpy().astype(int)

    all_word_vector_sequences = []

    for message in dff['lemma_sentence(with POS)']:
        message_as_vector_seq = message_to_words_vectors(message)

        if message_as_vector_seq.shape[0] == 0:
            message_as_vector_seq = np.zeros(shape=(1, 100))

        all_word_vector_sequences.append(message_as_vector_seq)

    return all_word_vector_sequences, y
#Check
X_train, Y_train = df_to_X_Y(train_df)
#print(len(X_train), len(X_train[0]))
#====================================================================================
X_test, Y_test = df_to_X_Y(test_df)
#=============================================== GLOVE EMBEDDING DONE....
#print(X_train)

import h5py

# Create an HDF5 file
with h5py.File('Manual_label_3000_unpad_glove.h5', 'w') as hf:
    # Save the X arrays
    hf.create_dataset('X_train', data=X_train)
    #hf.create_dataset('X_val', data=X_val)
    hf.create_dataset('X_test', data=X_test)
    
    # Save the Y arrays
    hf.create_dataset('Y_train', data=Y_train)
    #hf.create_dataset('Y_val', data=Y_val)
    hf.create_dataset('Y_test', data=Y_test)



    