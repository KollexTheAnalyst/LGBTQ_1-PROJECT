import praw
import datetime
import csv
import pandas as pd

# Create a Reddit instance using your API credentials
reddit = praw.Reddit(
    client_id="cRXkG30dsj5GlrbocwCReg",
    client_secret="gB63U8biGmtAeUqt86cnaTkD2AZ_lw",
    user_agent="my_app by u/Cholericity"
)

# Specify the submission ID
subreddit = reddit.subreddit('lgbtq')

post = reddit.submission(url='https://www.reddit.com/r/NoStupidQuestions/comments/1496c3w/how_did_gay_and_lesbian_people_find_each_other_in/')
# Replace 'LIMIT' with the number of comments you want to load
post.comments.replace_more(limit=None)

# Create an empty list to store the comment data
comment_data = []

# Function to recursively gather comment data
def gather_comment_data(comment):
    comment_data.append({
        'Comment': comment.body,
        'Timestamp': comment.created_utc,
        'Ups': comment.ups,
        'Downs': comment.downs
    })

    for reply in comment.replies:
        gather_comment_data(reply)

# Iterate through all comments, including the loaded ones
for comment in post.comments.list():
    gather_comment_data(comment)

# Check if there are comments
if len(comment_data) > 0:
    # Create a DataFrame from the comment data
    df = pd.DataFrame(comment_data)

    # Save the DataFrame to a CSV file
    df.to_csv('./60th extract.csv', index=False)
else:
    print("No comments found.")