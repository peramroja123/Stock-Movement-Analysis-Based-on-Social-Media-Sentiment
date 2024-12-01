
# Stock-Movement-Analysis-Based-on-Social-Media-Sentiment


Stock-Movement-Analysis-Based-on-Social-Media-Sentiment refers to leveraging the opinions and emotions expressed on social media platforms to predict or understand stock price changes. By analyzing posts, tweets, or discussions about specific stocks, algorithms classify sentiments as positive, negative, or neutral. Positive sentiments often correlate with price increases, while negative sentiments can signal potential declines. This analysis combines real-time public opinion with financial data to gain insights into market behavior, aiding investors in making informed trading decisions.

## Process to do that workflow:
* Scraping the data from social medial platforms
* Data Preprocessing
* Sentiment Analysis
* Feature Engineering
* Predictive modeling
* Evaluation

## How to scrap the data from Social media platforms

* I choose "Reddit" platform to scrap the data
First we import all libraries whatever we want like,

* numpy -> It is an open-source python library for scientific and mathematical computing
* pandas -> Pandas is a python library for data analysis and it mostly works on datasets.
* matplotlib.pyplot-> It is a python library that allows users to create 2D graphics.
* seaborn-> It helps to create statistical graphics for daya visualization.
* sklearn-> A popular library for machine learning algorithms and tools.
* train_test_split-> Splits datasets into training and testing subsets.
* LogisticRegression-> Implements logistic regression for classification problems.
* GaussianNB-> Naive Bayes classifier based on Gaussian distributions.
* KNeighborsClassifier-> Classifier implementing the k-nearest neighbors algorithm.
* DecisionTreeClassifier-> Classifier using a decision tree structure for classification tasks.
* accuracy_score-> Measures the accuracy of predictions.
* confusion_matrix-> Evaluates model performance by comparing actual vs predicted classes.
* classification_report-> Generates a report with precision, recall, F1-score, etc.

      import numpy as np
      import pandas as pd
      import matplotlib.pyplot as plt
      import seaborn as sns
      import sklearn
      from sklearn.model_selection import train_test_split
      from sklearn.linear_model import LogisticRegression
      from sklearn.naive_bayes import GaussianNB
      from sklearn.neighbors import KNeighborsClassifier
      from sklearn.tree import DecisionTreeClassifier
      from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# For data scraping we install praw by using 
    pip install praw
The praw (Python Reddit API Wrapper) library is used to interact with Reddit's API, enabling developers to programmatically access and manage Reddit content. It simplifies the process of retrieving posts, comments, and user data, as well as performing actions like submitting posts or comments.
        
    import praw # we need to import the library

    reddit = praw.Reddit(
    client_id= 'Sf695hto-aPmp_vJvG5VRA',
    client_secret='l7FWeWpdjkhUu9UMNI2BvnUeDXai0w',
    user_agent='MyStockBot/1.0 by Salt-Interest6663'
    )

* reddit = praw.Reddit -> This line initializes a Reddit instance using the praw library. The reddit object serves as a connection to the Reddit API, enabling interactions with Reddit (e.g., fetching posts, submitting comments).
* client_id='Sf695hto-aPmp_vJvG5VRA' -> client_id is a unique identifier provided by Reddit when you register an application.It represents your application and is used to authenticate API requests.
* client_secret='l7FWeWpdjkhUu9UMNI2BvnUeDXai0w' -> client_secret is a confidential key assigned to your Reddit app.It acts as a password to verify your application during API authentication.
* user_agent='MyStockBot/1.0 by Salt-Interest6663' -> user_agent is a string that identifies your application to Reddit's servers.It typically includes:A description of your app (e.g., MyStockBot),Version information (e.g., 1.0),Your Reddit username (e.g., Salt-Interest6663).

      subreddit = reddit.subreddit('stocks')  # Change to 'investing' or other relevant subreddits
      posts = []
      for submission in subreddit.new(limit=500):  # Fetch 500 latest posts
      posts.append({
        'title': submission.title,
        'selftext': submission.selftext,
        'score': submission.score,
        'comments': submission.num_comments,
        'created_utc': submission.created_utc
       })
#### Mainly,This code collects data from the newest 500 posts in the stocks subreddit, organizing it into a structured list of dictionaries. The collected data includes the title, body, score, comment count, and creation time of each post. This data can be used for further analysis, such as sentiment analysis or time-series modeling.
* subreddit = reddit.subreddit('stocks'):
This accesses the subreddit stocks via the reddit instance created earlier.
The subreddit object allows you to interact with posts and comments in that subreddit.You can replace 'stocks' with other subreddit names (e.g., 'investing', 'cryptocurrency') to fetch data from those communities. 
* posts = []:
Initializes an empty list called posts.
This list will be used to store the data extracted from the subreddit posts in a structured format.
* for submission in subreddit.new(limit=500):
Loops through the latest 500 posts in the stocks subreddit.
subreddit.new(limit=500):
Fetches the newest posts from the subreddit.The limit=500 specifies that up to 500 posts should be retrieved.submission represents a single post object in each iteration, allowing access to its attributes.
* posts.append({ }):
Adds a dictionary containing details of each post to the posts list.
Each dictionary has the following key-value pairs:

->'title': submission.title:
    Stores the post's title (headline).

->'selftext': submission.selftext:
Stores the post's body text (content).
If the post contains only a link, this field will be empty.

->'score': submission.score:
Stores the post's score (upvotes minus downvotes).

->'comments': submission.num_comments:
Stores the number of comments on the post.

->'created_utc': submission.created_utc:
Stores the time the post was created in UTC format (as a timestamp).

    df = pd.DataFrame(posts)
    df.to_csv('reddit_stocks.csv', index=False)
    
Step 1: Converts the list of Reddit post data (posts) into a structured format (Pandas DataFrame).

Step 2: Exports the DataFrame to a CSV file, which can then be used for further analysis, sharing, or archiving.

     import re
     df = pd.read_csv('reddit_stocks.csv')
     df['text'] = df['title'].fillna('') + ' ' + df['selftext'].fillna('')
     def clean_text(text):
        text = re.sub(r'http\S+', '', text)  # Remove URLs
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Remove special characters
        text = text.lower()  # Lowercase
        return text
     df['clean_text'] = df['text'].apply(clean_text)
     df.head()
* import re
Imports the re module, which is Python's built-in regular expression library. This module is used to search, match, and manipulate strings using patterns (regular expressions).
* df = pd.read_csv('reddit_stocks.csv')
Load the dataset into a table format for easy analysis and manipulation.
* df['text'] = df['title'].fillna('') + ' ' + df['selftext'].fillna(''):
Creates a new column in the DataFrame called 'text'.

df['title'].fillna(''): Fills any missing values in the 'title' column with an empty string ('').

df['selftext'].fillna(''): Fills any missing values in the 'selftext' column with an empty string ('').

Combines the cleaned title and body ('title' + 'selftext') into one column called 'text'. A space (' ') is added between them to ensure proper separation.

* def clean_text(text):
Defines a function named clean_text that takes a string (text) as input.

* text = re.sub(r'http\S+', '', text): 
Removes any URLs from the text.

-> r'http\S+': This is a regular expression pattern to match URLs starting with http followed by any non-whitespace characters (\S+).

-> re.sub replaces the matched URLs with an empty string ('').

-> text = re.sub(r'[^a-zA-Z0-9\s]', '', text): Removes special characters from the text.

-> r'[^a-zA-Z0-9\s]': Matches any character that is not a letter (uppercase or lowercase), a number, or a space.

-> These matched characters are replaced with an empty string (''), effectively removing them.

-> text = text.lower(): Converts the entire text to lowercase for uniformity, making the text case-insensitive.

-> return text: Returns the cleaned text.

* df['clean_text'] = df['text'].apply(clean_text):
Applies the clean_text function to each entry in the 'text' column.

The result is stored in a new column called 'clean_text'.
The apply() function is used to apply the clean_text function row-wise to the DataFrame.
* df.head()
Display first 5 rows from df
