import pandas as pd
from nltk.tokenize import word_tokenize
from collections import Counter
import numpy as np
#==============================================================================================
LGBTQ = pd.read_csv(r'C:\Users\My PC\Documents\DATA ANALYSIS\LGBTQ project\LGBTQ_project2\filtered_lgbtq_datasets.csv')

# Remove duplicate rows based on the "Comment" column
LGBTQ.drop_duplicates(subset=['Comment'], keep='first', inplace=True)   #10358rows
LGBTQ.reset_index(drop=True, inplace=True)

# Filter the data with LGBTQ-related words
s = LGBTQ['lemma_sentence(with POS)']
lgbtq_1 = LGBTQ[s.str.contains('gay|lesbian', case=False)]

# Drop rows that contain certain words in the 'lemma_sentence(with POS)' column
lgbtq_1 = lgbtq_1[~lgbtq_1['lemma_sentence(with POS)'].str.contains('people|lgbtq|lgbt|community|transgender', case=False)]

# Specify the number of rows for each label
rows_per_label = {
    '-1': 153,
    '0': 711,
    '1': 306
}

# Create an empty DataFrame to store the final result
final_dfs = []

# Iterate through the labels and add the specified number of rows for each
for label, count in rows_per_label.items():
    label_rows = lgbtq_1[lgbtq_1['label'] == int(label)].sample(n=count, replace=True)
    final_dfs.append(label_rows)

# Concatenate the DataFrames in the final_dfs list
final_df = pd.concat(final_dfs, ignore_index=True)

# Print the length of the final dataframe and the dataframe itself
#print(len(final_df))
#print(final_df)

# Count rows of extracted comments for each label
label_counts = final_df['label'].value_counts()
print(label_counts)
#final_df.to_csv('second_ext_lgbtq_datasets.csv',index = False, encoding='utf_8_sig')

