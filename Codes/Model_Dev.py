import tensorflow as tf
import pandas as pd
import h5py
import numpy as np
from tensorflow import keras
from keras.constraints import max_norm
import pickle

LGBTQ = pd.read_csv(r'C:\Users\My PC\Documents\DATA ANALYSIS\LGBTQ project\LGBTQ_project2\sample_ext_lgbtq_datasets.csv')

#dataset
# Open the HDF5 file in read mode
with h5py.File(r'C:\Users\My PC\Documents\DATA ANALYSIS\LGBTQ project\LGBTQ_project2\Manual_label_3000_W2V.h5', 'r') as hf:
    X_train = hf['X_train'][:]
    Y_train = hf['Y_train'][:]
    X_test = hf['X_test'][:]
    Y_test = hf['Y_test'][:]
    X_val = hf['X_test'][:]
    Y_val = hf['Y_test'][:]
#print((X_train.shape, Y_train.shape), (X_test.shape, Y_test.shape))

model = tf.keras.Sequential([])
    # Input layer for sequences (gotten from our padded embedding(input_shape) - this is for GLOVE)
model.add(tf.keras.layers.InputLayer(input_shape=(300, 200)))  
for units in [128,128,64,32]:
    model.add(tf.keras.layers.Bidirectional(
    tf.keras.layers.LSTM(units, activation='relu', return_sequences=True)
    ))
model.add(tf.keras.layers.Dropout(0.3))
model.add(tf.keras.layers.Bidirectional(
    tf.keras.layers.LSTM(units, activation='relu', return_sequences=True)
    ))
model.add(tf.keras.layers.Dropout(0.3))     
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(2, activation='softmax')) #SoftMax is used for multiclassification  #Output layer


model.compile(
    optimizer=tf.keras.optimizers.SGD(learning_rate = 0.1), #Here is where we input our RandomSearch Best Learning Rate for the four Optimizer
    loss= tf.keras.losses.CategoricalCrossentropy(from_logits=True),
    metrics = ['accuracy', tf.keras.metrics.AUC(name = 'auc')]
    )

#print(model.summary())
#==============================================================================
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

#print(weights)
#======================================================================================
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping

# Create an EarlyStopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=3)

Y_train_encoded = to_categorical(Y_train, num_classes=2)
Y_val_encoded = to_categorical(Y_val, num_classes=2)

model.fit(X_train, Y_train_encoded, batch_size = 100, validation_data = (X_val, Y_val_encoded), epochs = 5, callbacks=[early_stopping], class_weight = weights)

pickle.dump(model, open(r'C:\Users\My PC\Documents\DATA ANALYSIS\LGBTQ project\LGBTQ_project2\.venv\Model_Manual_SGD_W2V', 'wb'))

#for Word2vec training::: #X_train_W2V_NN = comment_vec_df.to_numpy() ----- converting the X_train to numpy format for keras to understand.
    #Input layer for sequences (gotten from our unpadded embedding(input_shape=(X_train_W2V_NN.shape[1], X_train_W2V_NN.shape[2])) - this is for Word2Vec
#============================================================================
#We built just Two(2) models each on the feature engineering chosen (Glove and Word2Vec)
#We use the same model structure for our learning rate random search ------ 
#We will eemployed the four optimizer as our chosen Hyperparameter for optimization of our model
#The best learning rate of each optimizer was randomly search using kera_tuner(random search for neural networks)
#The best learning rates were each used to train the Model.
#So, we have Bi-LSTM (GLOVE) with Optimizers 'Adam(Best Learning rate)', 'SGD(Best learning rate)', 'Adadelta(Best Learning rate)', and 'RMSprop(Best learning rate)'
#same for Bi-LSTM(Word2Vec) with Optimizers 'Adam(Best Learning rate)', 'SGD(Best learning rate)', 'Adadelta(Best Learning rate)', and 'RMSprop(Best learning rate)
#In total, our 2 models  were trained four(4) havaing us with four result in 2 places to compare.

#for SVC, were are only comparing the effective of Glove and Word2Vec on accuracy - it's a SVM classifier

#Our model development is onnly focused on Bi-LSTM as a neural network
    
#In this code, the number of units [128, 128, 64, 32] appears to be chosen as a part of the model architecture exploration. 
# It suggests that the model starts with two LSTM layers, each with 128 units, followed by two more layers with 64 and 32 units, respectively. 
# This sequence of units might have been chosen based on experimentation and the complexity of the problem you are trying to solve.

#ReLU activation functions are used to introduce non-linearity into neural networks, making them capable of learning complex patterns in data while remaining computationally efficient. 
# They have become the default choice for many deep learning applications due to their effectiveness and simplicity.
# (a regularization technique used in deep learning models, including Bidirectional Long Short-Term Memory (Bi-LSTM) models, to prevent overfitting)
