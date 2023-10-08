import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
from collections import Counter
import numpy as np
import seaborn as sns

first = pd.read_csv(r'C:\Users\My PC\Documents\DATA ANALYSIS\LGBTQ project\LGBTQ_project2\first_ext_lgbtq_datasets.csv')
second = pd.read_csv(r'C:\Users\My PC\Documents\DATA ANALYSIS\LGBTQ project\LGBTQ_project2\second_ext_lgbtq_datasets.csv')
#LGBTQ =  pd.read_csv(r'C:\Users\My PC\Documents\DATA ANALYSIS\LGBTQ project\LGBTQ_project2\Main_3000_LGTQ_DATASET.csv')
#==========================================================================================

# Concatenate the two dataframes
df = pd.concat([first, second], ignore_index=True)

# Print the length of the combined dataframe and the dataframe itselfprint(len(df))
print(len(df))
#print(df)
# Count rows of extracted comments for each label
label_counts = df['label'].value_counts()
print(label_counts)

df['lemma'].value_counts()[0:10]

#word cloud map
#https://www.datacamp.com/community/tutorials/wordcloud-python
def word_cloud(words):
    wordcloud = WordCloud(width=800, height=600, random_state=21, relative_scaling=0.5, background_color="white").generate(words)
    plt.figure(figsize=(8, 6))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis('off')
    plt.show()
word_set = ' '.join([comment_word for comment_word in df['lemma_sentence(with POS)']])
word_cloud(word_set)
#plt.savefig('word_cloud.jpg')

#Word frequency
texts = df['lemma_sentence(with POS)']
word_counts = Counter(word_tokenize('\n'.join(texts)))
word_top=word_counts.most_common(n=20)
#print(word_top)

#Word frequency graph
count_all = df['lemma_sentence(with POS)'].str.len().sum()
#print(count_all)

words=[count[0] for count in word_top]
frac_value=[int(count[1])/count_all for count in  word_top]
words = words[: :-1]
frac_value=sorted(frac_value)

#plot line chart
plt.figure(figsize=(10, 6))
plt.plot(words,frac_value)
plt.xticks(rotation=40, fontsize=13)
plt.xlabel('Words',fontsize=15)
plt.ylabel('Word Fraction',fontsize=15)
plt.title('Word Frequency Chart',fontsize=16)
#plt.show()
#df.to_csv('extraxt3_1000_lgbtq_datasets.csv',index = False, encoding='utf_8_sig')