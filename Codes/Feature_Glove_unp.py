import pandas as pd
import numpy as np
import spacy
import nltk
from nltk.tokenize import word_tokenize
import gensim
import matplotlib.pyplot as plt
from gensim.scripts.glove2word2vec import glove2word2vec
import os
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import pickle


LGBTQ = pd.read_csv(r'C:\Users\My PC\Documents\DATA ANALYSIS\LGBTQ project\LGBTQ_project2\Main_3000_LGTQ_DATASET.csv')

Tokenize_comment = LGBTQ['lemma_sentence(with POS)'].apply(word_tokenize)
# Load the GloVe pretrained word vectors
glove_file = r'C:\Users\My PC\Documents\DATA ANALYSIS\LGBTQ project\LGBTQ_project2\.venv\glove.6B.100d.txt'  # Replace with your GloVe file path
word2vec_glove_file = 'glove.6B.100d.word2vec'

# Convert GloVe format to Word2Vec format if not already done
if not os.path.exists(word2vec_glove_file):
    glove2word2vec(glove_file, word2vec_glove_file)

# Load the Word2Vec GloVe model
glove_model = gensim.models.KeyedVectors.load_word2vec_format(word2vec_glove_file, binary=False)

# Function to get the GloVe vector for a list of tokens
def glove_comment(tokens, model, vector_size):
    vectors = []
    for word in tokens:
        try:
            vector = model[word]
            vectors.append(vector)
        except KeyError:
            continue
    if not vectors:
        # Handle cases where there are no valid word vectors
        return np.zeros((1, vector_size))
    return np.mean(vectors, axis=0).reshape((1, vector_size))

# Create a NumPy array to hold the GloVe vectors for each comment
comment_arr = np.zeros((len(Tokenize_comment), 100))  # Assuming you are using 100-dimensional GloVe vectors

# Generate GloVe vectors for each comment
for i in range(len(Tokenize_comment)):
    comment_arr[i, :] = glove_comment(Tokenize_comment[i], glove_model, 100)

# Create a DataFrame to store the GloVe vectors
comment_vec_df = pd.DataFrame(comment_arr)

# Now, comment_vec_df contains the GloVe vectors for your comments.
#print(comment_vec_df)

X_train_SVC, X_test_SVC, Y_train_SVC, Y_test_SVC = train_test_split(comment_vec_df, LGBTQ['label'],test_size = 0.2)
#========================================================================================
model_svc = SVC()
#parameters in SVC
# c_list=list(range(1,51))
param_grid_svc = {'C': [1, 10, 100, 1000],
                'kernel': ['linear','poly','rbf','sigmoid'],
                'degree': [1,2,3,4]}
print(param_grid_svc)

SVC_RandomGrid = RandomizedSearchCV(estimator = model_svc, param_distributions = param_grid_svc, cv = 10, verbose=2, n_jobs = 4)
SVC_RandomGrid.fit(X_train_SVC, Y_train_SVC)
print(SVC_RandomGrid.best_params_)

#Model_svc Training
model_svc = SVC(C=10, kernel='rbf', degree=2)
model_svc = model_svc.fit(X_train_SVC,Y_train_SVC)

#pickle.dump(model_svc, open(r'C:\Users\My PC\Documents\DATA ANALYSIS\LGBTQ project\LGBTQ_project2\.venv\Model_Check_SVC_6000_Glove', 'wb'))

prediction_svc = model_svc.predict(X_test_SVC)

from sklearn.metrics import confusion_matrix
confusion_matrix(Y_test_SVC, prediction_svc)

from sklearn.utils.multiclass import unique_labels
unique_labels(Y_test_SVC)

import seaborn as sns
#Combine labels with confusion_matrix
def plot(Y_true, Y_pred):
    labels = unique_labels(Y_test_SVC)
    column = [f'Predicted{label}' for label in labels]
    indices = [f'Actual{label}' for label in labels]
    table = pd.DataFrame(confusion_matrix(Y_true, Y_pred),
                         columns = column, index = indices)

    return table

Final = plot(Y_test_SVC,prediction_svc)

print(Final)