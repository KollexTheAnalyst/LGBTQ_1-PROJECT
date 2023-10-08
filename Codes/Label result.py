import pandas as pd

LGBTQ = pd.read_csv(r'C:\Users\My PC\Documents\DATA ANALYSIS\LGBTQ project\LGBTQ_project2\Main_3000_LGTQ_DATASET.csv')

LGBTQ = LGBTQ.drop(['Timestamp', 'Created_timestamp', 'Created_date', 'Created_time', 'Ups', 'Downs'], axis=1)
LGBTQ = LGBTQ.groupby('label').count()

print(len(LGBTQ))
print(LGBTQ.head())