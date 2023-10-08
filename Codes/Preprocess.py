import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer as SIA
import nltk
from nltk import pos_tag
nltk.download('averaged_perceptron_tagger')
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
stem_lemmatizer = WordNetLemmatizer()
import numpy as np

LGBTQ = pd.read_csv(r'C:\Users\My PC\Documents\DATA ANALYSIS\LGBTQ project\LGBTQ_project2\clean_dataset_lgbtq.csv')

#tokenization
def comment_token(text):
    words_set=text.str.split()
    tokens=[word for word in words_set]
    return tokens

LGBTQ['tokens'] = comment_token(LGBTQ['clean_comment'])
#print(LGBTQ.head())

#lemmatize or stem
word_tokenizer = nltk.tokenize.WhitespaceTokenizer()
word_lemmatizer = nltk.stem.WordNetLemmatizer()

def lemmatize_text(text):
    return [word_lemmatizer.lemmatize(word) for word in word_tokenizer.tokenize(text)]
LGBTQ['lemma'] = LGBTQ['clean_comment'].apply(lemmatize_text)
#print(LGBTQ.head())

#lemma_sentence
LGBTQ['lemma_sentence'] = LGBTQ['lemma'].apply(lambda x: ' '.join(x))
#print(LGBTQ.head())

#POS for clean comments
#reference:
#https://www.machinelearningplus.com/nlp/lemmatization-examples-python/
#https://stackoverflow.com/questions/51267166/lemmatization-pandas-python

def convert_wordnet_tag(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return None

def pos_tag_set(Comment):
    tagged_words = nltk.pos_tag(nltk.word_tokenize(Comment))
    new_tag=[]
    for word, tag in tagged_words:
        new_tag.append(tuple([word, convert_wordnet_tag(tag)]))
    return new_tag
LGBTQ['pos_tag'] = LGBTQ['clean_comment'].apply(pos_tag_set)
#print(LGBTQ)

#create lemma sentence with pos-tags
def handle_lemma(pos_comment):
    lemma_set = " "
    for word, pos in pos_comment:
        if not pos: 
            lemma = word
            lemma_set = lemma_set + " " + lemma
        else:  
            lemma = stem_lemmatizer.lemmatize(word, pos=pos)
            lemma_set = lemma_set + " " + lemma
    return lemma_set
LGBTQ['pos_tag'].apply(handle_lemma)
    
LGBTQ['lemma_sentence(with POS)'] = LGBTQ['pos_tag'].apply(handle_lemma)
#print(LGBTQ)

#Sentiment_analysis using VADER--Lexicon-based analysis (--Textblob and Sentiword)
def vaderSentiment_method(LGBTQ):
    sentiment_analyzer = SIA()
    snt_score = sentiment_analyzer.polarity_scores(LGBTQ['lemma_sentence(with POS)'])
    return snt_score['compound'] 

LGBTQ['vader_score'] = LGBTQ.apply(vaderSentiment_method, axis=1)
#print(LGBTQ.head())

#Labelling with VADER
def senti_label_2(score):
    if score <= -0.05:
        return -1 #==37818
    elif score >= 0.05:
        return 1 #==29914
    else:
        return 0 #24916
    
#using senti_label_2
LGBTQ['label'] = LGBTQ['vader_score'].apply(senti_label_2)
label_1 = LGBTQ.groupby('label').count()
#print('vader_step1:',label_1["vader_score"])

#extreme positive
slight_pos_vader = LGBTQ[LGBTQ['vader_score'].between(0.5,1)] #13225
#print("vader",len(slight_pos_vader))

#extreme negative 
slight_pos_vader = LGBTQ[LGBTQ['vader_score'].between(-1,-0.5)] #20758
#print("vader",len(slight_pos_vader))

print(LGBTQ)
print(len(LGBTQ))
#LGBTQ.to_csv('supervised_lgbtq_datasets.csv',index = False, encoding='utf_8_sig')