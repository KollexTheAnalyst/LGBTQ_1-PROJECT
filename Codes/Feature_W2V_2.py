from sklearn.ensemble._forest import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV
import h5py
import keras
import matplotlib.pyplot as plt

model_svc = SVC()
#===============================================================================================
from nltk.tokenize import word_tokenize
import gensim
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
import keras


LGBTQ = pd.read_csv(r'C:\Users\My PC\Documents\DATA ANALYSIS\LGBTQ project\LGBTQ_project2\Main_3000_LGTQ_DATASET.csv')

Tokenize_comment = LGBTQ['lemma_sentence(with POS)'].apply(word_tokenize)
#print(Tokenize_comment)

#==============================================================================
#raining of Model_W2V
Model_W2V = gensim.models.Word2Vec(Tokenize_comment, vector_size=200, #features
                                   window=5, 
                                   min_count=1, 
                                   sg=1,  #skip-gram model
                                   hs=0,
                                   negative=10, 
                                   workers=2, 
                                   seed=34 )

# Save the Word2Vec model
Model_W2V.save("word2vec_model.model")

#Splitting the dataset into dataframes
train_df,temp_df = train_test_split(LGBTQ, test_size = 0.3, random_state=42)
test_df, val_df = train_test_split(temp_df, test_size=0.5, random_state=42)

# Function to get word embeddings for a sentence
def get_sentence_embeddings(sentence, model):
    words = sentence.split()
    embeddings = []
    for word in words:
        if word in model.wv:
            embeddings.append(model.wv[word])
    return embeddings

# Apply the function to the training, test, and validation dataframes
train_df['Embeddings'] = train_df['lemma_sentence(with POS)'].apply(lambda x: get_sentence_embeddings(x, Model_W2V))
test_df['Embeddings'] = test_df['lemma_sentence(with POS)'].apply(lambda x: get_sentence_embeddings(x, Model_W2V))
val_df['Embeddings'] = val_df['lemma_sentence(with POS)'].apply(lambda x: get_sentence_embeddings(x, Model_W2V))

# Define a function to pad embeddings in each sentence
def pad_embeddings(embeddings, max_length):
    padded_embeddings = []
    for emb_list in embeddings:
        if len(emb_list) >= max_length:
            padded_embeddings.append(emb_list[:max_length])
        else:
            padding = [np.zeros(200)] * (max_length - len(emb_list))
            padded_emb = emb_list + padding
            padded_embeddings.append(padded_emb)
    return padded_embeddings
#KeyError: 'Padded_Embeddings'
#Define maximum sequence
max_sequence_length = 300

# Apply padding to the training, test, and validation dataframes
train_df['Padded_Embeddings'] = pad_embeddings(train_df['Embeddings'], max_sequence_length)
test_df['Padded_Embeddings'] = pad_embeddings(test_df['Embeddings'], max_sequence_length)
val_df['Padded_Embeddings'] = pad_embeddings(val_df['Embeddings'], max_sequence_length)

# Extract the padded embeddings and labels from the training dataframe
X_train = list(train_df['Padded_Embeddings'])
Y_train = list(train_df['label'])
X_test = list(test_df['Padded_Embeddings'])
Y_test = list(test_df['label'])
X_val = list(val_df['Padded_Embeddings'])
Y_val = list(val_df['label'])

# Convert them to numpy arrays for compatibility with machine learning models
X_train = np.array(X_train)
Y_train = np.array(Y_train)
X_test = np.array(X_test)
Y_test = np.array(Y_test)
X_val = np.array(X_val)
Y_val = np.array(Y_val)

# Create an HDF5 file
with h5py.File('Manual_label_3000_W2V.h5', 'w') as hf:
    # Save the X arrays
    hf.create_dataset('X_train', data=X_train)
    hf.create_dataset('X_val', data=X_val)
    hf.create_dataset('X_test', data=X_test)
    
    # Save the Y arrays
    hf.create_dataset('Y_train', data=Y_train)
    hf.create_dataset('Y_val', data=Y_val)
    hf.create_dataset('Y_test', data=Y_test)

#print(X_train.shape, Y_train.shape)

