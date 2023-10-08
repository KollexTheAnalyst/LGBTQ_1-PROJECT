import tensorflow as tf
import keras
import pickle
import h5py
from keras.utils import to_categorical
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#dataset
# Open the HDF5 file in read mode
with h5py.File(r'C:\Users\My PC\Documents\DATA ANALYSIS\LGBTQ project\LGBTQ_project2\Manual_label_3000_W2V.h5', 'r') as hf:
    X_train = hf['X_train'][:]
    Y_train = hf['Y_train'][:]
    X_test = hf['X_test'][:]
    Y_test = hf['Y_test'][:]
    X_val = hf['X_test'][:]
    Y_val = hf['Y_test'][:]


model_2_loaded = pickle.load(open(r'C:\Users\My PC\Documents\DATA ANALYSIS\LGBTQ project\LGBTQ_project2\.venv\Model_SVC_3000_W2V', 'rb'))
#print(model_2_loaded.summary())

from sklearn.metrics import classification_report
test_predictions = np.argmax(model_2_loaded.predict(X_test), axis=1)
#test_predictions = (model_2_loaded.predict(X_test) > 0.5).astype(int)  #Binary Classification (0.5 = threshold)

#print(classification_report(Y_test, test_predictions))

from sklearn.metrics import confusion_matrix
confusion_matrix(Y_test, test_predictions)

from sklearn.utils.multiclass import unique_labels
unique_labels(Y_test)

import seaborn as sns
#Combine labels with confusion_matrix
def plot(Y_true, Y_pred):
    labels = unique_labels(Y_test)
    column = [f'Predicted{label}' for label in labels]
    indices = [f'Actual{label}' for label in labels]
    table = pd.DataFrame(confusion_matrix(Y_true, Y_pred),
                         columns = column, index = indices)

    return table

Final = plot(Y_test,test_predictions)

print(Final)




# Create a heatmap using Seaborn
    #sns.set(font_scale=1.2)  # Adjust the font size if needed
   # plt.figure(figsize=(8, 6))  # Adjust the figure size if needed
  #  heatmap = sns.heatmap(table, annot=True, fmt='d', cmap='viridis')

 #   return heatmap

#heatmap = plot(Y_test, test_predictions)
#plt.show()
#print(table)

