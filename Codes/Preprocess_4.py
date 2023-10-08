import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
from collections import Counter
import numpy as np
import seaborn as sns

#==============================================================================================

LGBTQ_1 = pd.read_csv(r'C:\Users\My PC\Documents\DATA ANALYSIS\LGBTQ project\LGBTQ_project2\filtered_lgbtq_datasets.csv')

# Remove duplicate rows based on the "Comment" column
LGBTQ_1.drop_duplicates(subset=['Comment'], keep='first', inplace=True)   #10358rows
LGBTQ_1.reset_index(drop=True, inplace=True)
#print(len(LGBTQ_1))
#print(LGBTQ_1.head())

#filter the data with lgbtq related words
s = LGBTQ_1['lemma_sentence(with POS)']
lgbtq_2 = LGBTQ_1[s.str.contains('lgbtq|lgbt|community|transgender', case=False)]

# Drop rows that contain the word "people,transgender,lesbian,gay" in the 'text' column
lgbtq_2 = lgbtq_2[~lgbtq_2['lemma_sentence(with POS)'].str.contains('people|gay|lesbian', case=False)]
print(len(lgbtq_2))    #1830rows
print(lgbtq_2) 

#count row of extracted comments
label_2 = lgbtq_2.groupby('label').count() #"-1" = 803 (847), "0" = 263 (289), "1" = 649
print(label_2) 

#lgbtq_2.to_csv('first_ext_lgbtq_datasets.csv',index = False, encoding='utf_8_sig')