import pandas as pd
import emoji
import re
import nltk
from nltk.corpus import stopwords
from nltk.corpus import wordnet
import numpy as np
#===============================================================================================================
LGBTQ = pd.read_csv(r'C:\Users\My PC\Documents\DATA ANALYSIS\LGBTQ project\LGBTQ_project2\LGBTQ Dataset2.csv')
#print(len(LGBTQ)) #273017 comments
#empty_comment=LGBTQ['Comment'].isna().value_counts()
#print(empty_comment) #no empty comments
LGBTQ['clean_comment'] = LGBTQ['Comment'].copy()
#print(LGBTQ)
#================================================================================================================
#handle emoji
def convert_emoji(text):
    text=[emoji.demojize(tw) for tw in text]  #run slowly
    new_df= pd.DataFrame(text, columns=['Comment'])
    return new_df
new_df=convert_emoji(LGBTQ['clean_comment'])
#print(new_df)

LGBTQ[['clean_comment']]=new_df[['Comment']].copy()  
#print(LGBTQ)
#================================================================================================================
#simple data cleaning
#reference:https://github.com/ugis22/analysing_twitter/blob/master/Jupyter%20Notebook%20files/Analysis%20of%20Twitter.ipynb
def preprocessing_one(lgbtq):
    #lower all characters
    lgbtq['clean_comment'] = lgbtq['clean_comment'].str.lower()
    #remove all the mentions: @username
    lgbtq['clean_comment'] = lgbtq['clean_comment'].replace(r'@\w+', '', regex=True)
    #remove all the links in the original tweets (start with "www" and "http")
    lgbtq['clean_comment'] = lgbtq['clean_comment'].replace(r'http\S+|rhttps\S+|rwww\S+', '', regex=True)   
    return lgbtq
LGBTQ = preprocessing_one(LGBTQ)

#Notice: Remove punctuation and special characters after handling contraction words

#Handling repeated characters
#reference: https://github.com/ugis22/analysing_twitter/blob/master/Jupyter%20Notebook%20files/Analysis%20of%20Twitter.ipynb
#https://stackoverflow.com/questions/3788870/how-to-check-if-a-word-is-an-english-word-with-python
#re.sub(pattern, repl, string, count): pattern(Eligible pattern)，repl(replace to...), string
def repeated_char(word):
    repeat_pattern = re.compile(r'(\w*)(\w)\2(\w*)')
    repl_word = r'\1\2\3'
    if wordnet.synsets(word):
        return word  #test for words existence
    #repl_new= repeat_pattern.sub(repl_word, word)
    repl_new = re.sub(repeat_pattern, repl_word, word)
    if repl_new != word:
        return repeated_char(repl_new)
    else:
        return repl_new
word1='loooove'
#print(repeated_char(word1))

def check_repeated (Comment):
    repeat_pattern = r'(\w*)(\w+)(\2)(\w*)'
    word_set = [''.join(i) for i in re.findall(repeat_pattern, Comment)]  #find all the words with repeated characters
    for word in word_set:
        if not wordnet.synsets(word):
            Comment=re.sub(word, repeated_char(word), Comment)
    return Comment
#test the function
comment1='I looove you, soooo much'
#print(check_repeated (comment1))

#replace words contraction
#reference:https://github.com/kiran-bal/Disaster_tweets_classifier/blob/2e6d648f5ef9cbe67024ad5cf032582fc4dc3a75/version2/notebooks/Disaster_tweet_classifier.ipynb
#re.sub(pattern, repl, string, count=0, flags=0): count=0: all matched will be replaced
contraction_dict=[(r'I\'m', 'I am'),(r'i\'m', 'i am'),(r'ain\'t', 'am not'),(r'(\w+)\'s', '\g<1> is'),(r'(\w+)\'re', '\g<1> are'),(r'(\w+)n\'t', '\g<1> not'),
                  (r'can\'t', 'cannot'),(r'won\'t', 'will not'), (r'wont', 'will not'), (r'(\w+)\'ll', '\g<1> will'), (r'(\w+)\'d', '\g<1> would'), (r'(\w+)\'ve', '\g<1> have'),
                 (r'I\’m', 'I am'),(r'i\’m', 'i am'),(r'ain\’t', 'am not'),(r'(\w+)\’s', '\g<1> is'),(r'(\w+)\’re', '\g<1> are'),(r'(\w+)n\’t', '\g<1> not'),
                  (r'can\’t', 'cannot'),(r'won\’t', 'will not'), (r'(\w+)\’ll', '\g<1> will'), (r'(\w+)\’d', '\g<1> would'), (r'(\w+)\’ve', '\g<1> have')]
#Notice: The quotation of some texts are not in English
def handle_contraction(text):
    patterns_set=[(re.compile(pattern), repl) for (pattern, repl) in contraction_dict]
    for (pattern, repl) in patterns_set:
        text=re.sub(pattern, repl, text)
    return text
#test
comment2 = "I'm not around and won't come today"
#print(comment2)
#print(handle_contraction(comment2))

def preprocessing_two(lgbtq):
    lgbtq['clean_comment'] = lgbtq['clean_comment'].apply(lambda x: check_repeated(x)) #remove repeated charaters
    lgbtq['clean_comment'] = lgbtq['clean_comment'].apply(lambda x: handle_contraction(x)) #handle constraction
    return lgbtq
LGBTQ = preprocessing_two(LGBTQ)
#print(LGBTQ)

#only English character
def replace_non_alphabetical(lgbtq):
    lgbtq['clean_comment'] = lgbtq['clean_comment'].replace('[^a-zA-Z]',' ', regex=True)
    lgbtq['clean_comment'] = lgbtq['clean_comment'].replace('\s+', ' ', regex=True)
    return lgbtq
LGBTQ = replace_non_alphabetical(LGBTQ)
#print(LGBTQ)

#remove less than two-character words, but keep "no" if len(w)>2 or w=="no"
def short_words(lgbtq):
    lgbtq['clean_comment'] = lgbtq['clean_comment'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>2 or w=="no"]))
    return lgbtq
LGBTQ = short_words(LGBTQ)
#print(LGBTQ.head())

#remove stopwords
#can't remove words like "not" or "no"
#create own stopwords
my_stopwords = [x for x in open('C:\\Users\\My PC\\Documents\\DATA ANALYSIS\\LGBTQ project\\LGBTQ_project2\\stopwords.txt','r', encoding="utf-8").read().split('\n')]

def remove_stopwords(lgbtq):
    lgbtq['clean_comment'] = lgbtq['clean_comment'].apply(lambda x: ' '.join([word for word in x.split() if word not in my_stopwords]))
    return lgbtq
LGBTQ = remove_stopwords(LGBTQ)
#print(my_stopwords)
#print(LGBTQ.head())

#remove empty commentss
def remove_empty(lgbtq):
    lgbtq = lgbtq[lgbtq['clean_comment']!='']
    return lgbtq
LGBTQ = remove_empty(LGBTQ)
LGBTQ = LGBTQ.reset_index(drop=True) #271377
#print(LGBTQ)
#print(len(LGBTQ))

# Remove duplicate rows
LGBTQ = LGBTQ.drop_duplicates()
# Remove empty rows
LGBTQ = LGBTQ.dropna()
LGBTQ = LGBTQ.reset_index(drop=True) #67334
# Remove duplicate rows based on the "Comment" column
LGBTQ.drop_duplicates(subset=['Comment'], keep='first', inplace=True)
LGBTQ.reset_index(drop=True, inplace=True)
#print(LGBTQ)
#print(len(LGBTQ))

#LGBTQ.to_csv('clean_dataset_lgbtq.csv',index = False, encoding='utf_8_sig')
