import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
from collections import Counter
import numpy as np
import seaborn as sns

LGBTQ = pd.read_csv(r'C:\Users\My PC\Documents\DATA ANALYSIS\LGBTQ project\LGBTQ_project2\supervised_lgbtq_datasets.csv')



#filter the data with lgbtq related words
s = LGBTQ['lemma_sentence(with POS)']
# Drop rows that contain certain words in the 'lemma_sentence(with POS)' column
#lgbtq_1 = s[~LGBTQ['lemma_sentence(with POS)'].str.contains('people|lgbtq|lgbt|community|transgender', case=False)]

lgbtq_1 = LGBTQ[s.str.contains('lgbtq|lesbian|gay|community|equality|antilgbtq|transgender|right', case=False)]
#print(len(lgbtq_1))
#print(lgbtq_1)

#count row of extracted comments
#label_1 = s.groupby('label').count() #"-1" = 6478 , "0" = 2083 , "1" = 5004
label_counts = lgbtq_1['label'].value_counts()
#print(label_counts)

lgbtq_1['lemma'].value_counts()[0:10]

#word cloud map
#https://www.datacamp.com/community/tutorials/wordcloud-python
def word_cloud(words):
    wordcloud = WordCloud(width=800, height=600, random_state=21, relative_scaling=0.5, background_color="white").generate(words)
    plt.figure(figsize=(8, 6))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis('off')
    plt.show()
word_set = ' '.join([comment_word for comment_word in lgbtq_1['lemma_sentence(with POS)']])

#word_cloud(word_set)
#plt.savefig('word_cloud.jpg')

#Word frequency
texts = lgbtq_1['lemma_sentence(with POS)']
word_counts = Counter(word_tokenize('\n'.join(texts)))
word_top=word_counts.most_common(n=20)
#print(word_top)

#Word frequency graph
count_all = lgbtq_1['lemma_sentence(with POS)'].str.len().sum()
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

#print(lgbtq_1)
#lgbtq_1.to_csv('filtered_lgbtq_datasets.csv',index = False, encoding='utf_8_sig')

# Taking 1000rows each from the lgbtq_1 dataframe
# Group the data by the 'label' column and sample 1000 rows from each group
#extract_1000 = lgbtq_1.groupby('label', group_keys=False).apply(lambda x: x.sample(min(len(x), 1000)))
#extract_1000 = extract_1000.reset_index(drop=True)
#print(extract_1000)
