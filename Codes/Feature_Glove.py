import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#import gensim
#print("Gensim version:", gensim.__version__)
import spacy
import nltk
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

LGBTQ = pd.read_csv(r'C:\Users\My PC\Documents\DATA ANALYSIS\LGBTQ project\LGBTQ_project2\Check\Check_F_1500_data.csv')
#===========================================================================
words = dict()

def add_to_dict(d, filename):
    with open(filename, encoding='utf-8') as file:
        for line in file.readlines():
            line = line.split(' ')

            try:
                d[line[0]] = np.array(line[1:], dtype=float)
            except:
                continue

add_to_dict(words, r'C:\Users\My PC\Documents\DATA ANALYSIS\LGBTQ project\LGBTQ_project2\.venv\glove.6B.100d.txt')
#print(len(words)) 
#=======================================================================================
#tokenize text
tokenizer = nltk.RegexpTokenizer(r"\w+")
#print(tokenizer.tokenize('How are you'))
lemmatizer = WordNetLemmatizer()

lemmatizer.lemmatize('good')

def message_to_token_list(s):
  tokens = tokenizer.tokenize(s)
  lowered_tokens = [t.lower() for t in tokens]
  lemmatized_tokens = [lemmatizer.lemmatize(t) for t in lowered_tokens]
  useful_tokens = [t for t in lemmatized_tokens if t in words]
  return useful_tokens
#print(message_to_token_list('How are you'))
#=============================================================
def message_to_words_vectors(message, word_dict=words):
    processed_list_of_tokens = message_to_token_list(message)

    vectors = []

    for token in processed_list_of_tokens:
        if token not in word_dict:
           continue

        token_vector = word_dict[token]
        vectors.append(token_vector)

    return np.array(vectors, dtype = float)

print(message_to_words_vectors('How are you').shape)
#===================================================================================  
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
#simple Data Analysis
sequence_length = []

for i in range(len(X_train)):
 sequence_length.append(len(X_train[i]))

plt.hist(sequence_length)

print(pd.Series(sequence_length).describe())
#===========================================================================================

from copy import deepcopy

def pad_X(X, desired_sequence_length = 286):
    X_copy = deepcopy(X)

    for i, x in enumerate(X):
        x_seq_len = x.shape[0]
        sequence_length_difference = desired_sequence_length - x_seq_len

        pad = np.zeros(shape = (sequence_length_difference, 100))

        X_copy[i] = np.concatenate([x, pad])

    return np.array(X_copy).astype(float)

X_train = pad_X(X_train)
#==========================================================

#===================================================================
X_val, Y_val = df_to_X_Y(val_df)
X_val = pad_X(X_val)

#print(X_val.shape, Y_val.shape)

X_test, Y_test = df_to_X_Y(test_df)
X_test = pad_X(X_test)

print(X_test.shape, Y_test.shape)
#=============================================== GLOVE EMBEDDING DONE....

import h5py

# Create an HDF5 file
with h5py.File('Check_1500_NN.h5', 'w') as hf:
    # Save the X arrays
    hf.create_dataset('X_train', data=X_train)
    hf.create_dataset('X_val', data=X_val)
    hf.create_dataset('X_test', data=X_test)
    
    # Save the Y arrays
    hf.create_dataset('Y_train', data=Y_train)
    hf.create_dataset('Y_val', data=Y_val)
    hf.create_dataset('Y_test', data=Y_test)



