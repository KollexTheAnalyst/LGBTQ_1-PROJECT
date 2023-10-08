from sklearn.ensemble._forest import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV
import h5py
import keras

model_svc = SVC()
#===============================================================================================
from nltk.tokenize import word_tokenize
import gensim
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
import keras
import pickle



LGBTQ = pd.read_csv(r'C:\Users\My PC\Documents\DATA ANALYSIS\LGBTQ project\LGBTQ_project2\Check\Check_F_1500_data.csv')

Tokenize_comment = LGBTQ['lemma_sentence(with POS)'].apply(word_tokenize)
#print(Tokenize_comment)
#=============================================================

Model_W2V = gensim.models.Word2Vec(Tokenize_comment, vector_size=200, #features
                                   window=5, 
                                   min_count=1, 
                                   sg=1,  #skip-gram model
                                   hs=0,
                                   negative=10, 
                                   workers=2, 
                                   seed=34 )

#Each word can get its own vector. The representation of a tweets can the vector sum of each word divided by the total number(average) 
#or just the sum of each word vector
def word2vec_comment(tokens, vector_size):
    vector=np.zeros(vector_size).reshape((1,vector_size))
    vector_cnt = 0
    for word in tokens:
        vector += Model_W2V.wv[word].reshape((1, vector_size))
        vector_cnt += 1
    return vector/vector_cnt  #average for tweets

def word2vec_comment_2(tokens, vector_size):
    vector=np.zeros(vector_size).reshape((1,vector_size))
    vector_cnt = 0
    for word in tokens:
        vector += Model_W2V.wv[word].reshape((1, vector_size))
    return vector  #sum of tweets

comment_arr=np.zeros((len(Tokenize_comment), 200))

for i in range (len(Tokenize_comment)):
    comment_arr[i,:] = word2vec_comment(Tokenize_comment[i], 200)
comment_vec_df = pd.DataFrame(comment_arr)

#print(comment_vec_df)

X_train_SVC, X_test_SVC, Y_train_SVC, Y_test_SVC = train_test_split(comment_vec_df, LGBTQ['label'],test_size = 0.2)
#========================================================================================

#parameters in SVC
# c_list=list(range(1,51))
param_grid_svc = {'C': [1, 10, 100, 1000],
                  #'kernel': ['linear','poly','rbf','sigmoid'],
                  #'degree': [1,2,3,4]}
}
print(param_grid_svc)

SVC_RandomGrid = RandomizedSearchCV(estimator = model_svc, param_distributions = param_grid_svc, cv = 10, verbose=2, n_jobs = 4)
SVC_RandomGrid.fit(X_train_SVC, Y_train_SVC)
print(SVC_RandomGrid.best_params_)


model_svc = SVC(C=1000, kernel='rbf', degree=2)
model_svc = model_svc.fit(X_train_SVC,Y_train_SVC)

#pickle.dump(model_svc, open(r'C:\Users\My PC\Documents\DATA ANALYSIS\LGBTQ project\LGBTQ_project2\.venv\Model_Check_SVC_6000_W2V', 'wb'))

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

