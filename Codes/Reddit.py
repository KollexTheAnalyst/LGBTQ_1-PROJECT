#Import Statements
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import spacy
import re

from xgboost import XGBClassifier
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer,HashingVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
from tqdm import tqdm

#Extract twitter lgbtq data
df = pd.read_csv('C:\\Users\\My PC\\Documents\\DATA ANALYSIS\\LGBTQ project\\LGBTQ_project2\\LGBTQ Dataset2.csv')
df.rename(columns = {'Comment': 'text'}, inplace = True)

#DATA MANIPULATION AND CLEANING
# Remove duplicate rows
df = df.drop_duplicates()
# Remove empty rows
df = df.dropna()
# Reset the index
df = df.reset_index(drop=True)

#DATA PREPROCESSING
#create a copy of the original dataset (df)
LGBTQ = df.copy()
#Create another column for text called texts
LGBTQ['texts'] = df['text']
#Delete the unneeded columns
LGBTQ = LGBTQ.drop(['Timestamp', 'Created_timestamp', 'Created_date', 'Created_time', 'Ups', 'Downs'], axis=1)
#Create a Boolean to check if there's @ in comment
rt_mask = LGBTQ.text.apply(lambda x: "RT @" in x)

# standard text preprocessing 
LGBTQ.texts = LGBTQ.texts.str.lower()
#Remove tweet handlers if any
LGBTQ.texts = LGBTQ.texts.apply(lambda x:re.sub('@[^\s]+','',x))
#remove hashtags
LGBTQ.texts = LGBTQ.texts.apply(lambda x:re.sub(r'\B#\S+','',x))
# Remove URLS
LGBTQ.texts = LGBTQ.texts.apply(lambda x:re.sub(r"http\S+", "", x))
# Remove all the special characters
LGBTQ.texts = LGBTQ.texts.apply(lambda x:' '.join(re.findall(r'\w+', x)))
#remove all single characters
LGBTQ.texts = LGBTQ.texts.apply(lambda x:re.sub(r'\s+[a-zA-Z]\s+', '', x))
# Substituting multiple spaces with single space
LGBTQ.texts = LGBTQ.texts.apply(lambda x:re.sub(r'\s+', ' ', x, flags=re.I))
# Remove numbers and punctuation
LGBTQ['texts'] = LGBTQ['texts'].apply(lambda x: re.sub(r"[^a-zA-Z]", " ", x))

# Check the data type of a specific column
#column_name = 'texts'  # Replace with the name of the column you want to check
#data_type = LGBTQ['texts'].dtype
#print(f"Data type of column '{column_name}': {data_type}")

#Getting Stopwords and list of punctuations
stopwords = list(STOP_WORDS)
punct = list(punctuation)
#print("Length of punctuations:\t {} \nLength of stopwords:\t {}".format(len(punct), len(stopwords)))

import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA
nltk.download('vader_lexicon')
from spellchecker import SpellChecker

# For sentiment analysis 
sia = SIA() 

# To identify misspelled words
spell = SpellChecker() 

def label_sentiment(x:float):
    if x < -0.05 : return '-1' #Negative
    if x > 0.35 : return '1' #Positive
    return '0' #Neutral

# Feature Extraction
LGBTQ['words'] = LGBTQ.texts.apply(lambda x:re.findall(r'\w+', x ))
LGBTQ['errors'] = LGBTQ.words.apply(spell.unknown)
LGBTQ['errors_count'] = LGBTQ.errors.apply(len)
LGBTQ['words_count'] = LGBTQ.words.apply(len)
LGBTQ['sentence_length'] = LGBTQ.texts.apply(len)

# Extract Sentiment Values for each tweet 
LGBTQ['sentiment'] = [sia.polarity_scores(x)['compound'] for x in tqdm(LGBTQ['texts'])]
LGBTQ['overall_sentiment'] = LGBTQ['sentiment'].apply(label_sentiment)

# Count the occurrences of each sentiment label
sentiment_counts = LGBTQ['overall_sentiment'].value_counts()

# Print the counts
#print("Sentiment Counts:")
#print(sentiment_counts)

fig , ax = plt.subplots(figsize = (10,10))
ax = LGBTQ['overall_sentiment'].value_counts().plot(kind = 'bar')

plt.xticks(rotation = 0, size = 14)
plt.yticks(size = 14, color = 'white')
plt.title('Distribution of Sentiment', size = 20)

ax.annotate(text = LGBTQ['overall_sentiment'].value_counts().values[0], xy = (-0.13,12821), size = 18)
ax.annotate(text = LGBTQ['overall_sentiment'].value_counts().values[1], xy = (0.87,12025), size = 18)
ax.annotate(text = LGBTQ['overall_sentiment'].value_counts().values[2], xy = (1.87,7610), size = 18)

#print(plt.show())


# Create a Vectorizer Object using default parameters
hash_vectorizer = HashingVectorizer()

# Convert a collection of text documents to a matrix of token counts
token_count_matrix = hash_vectorizer.fit_transform(LGBTQ['texts'])
#print(f'The size of the count matrix for the texts = {token_count_matrix.get_shape()}')
#print(f'The sparse count matrix is as follows:')
#print(token_count_matrix)

# Create a tf_idf object using default parameters
tf_idf_transformer = TfidfTransformer(use_idf=True, smooth_idf=True, sublinear_tf=False) 

# Fit to the count matrix, then transform it to a normalized tf-idf representation
tf_idf_matrix = tf_idf_transformer.fit_transform(token_count_matrix)

#print(f'The size of the tf_idf matrix for the texts = {tf_idf_matrix.get_shape()}')
#print(f'The sparse tf_idf matrix is as follows:')
#print(tf_idf_matrix)

#Getting X and y

X = tf_idf_matrix
y = LGBTQ.overall_sentiment

#Splitting the data into training and testing
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.1)

#Creating, fitting and scoring classifier
classifier = LinearSVC()
classifier.fit(X_train, y_train)
print(f"Accuracy: {classifier.score(X_test, y_test) * 100:.3f}%", )