
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
 
    pip install Textblob

* pip install textblob is a command used to install the TextBlob library in Python using pip, Python's package manager.
## What is TextBlob?
TextBlob is a Python library for processing textual data. It provides simple APIs for common natural language processing (NLP) tasks such as:

-> Part-of-speech tagging

-> Noun phrase extraction

-> Sentiment analysis

-> Classification

-> Tokenization (splitting text into words or sentences)

-> Translation and language detection (via Google Translate API)

-> Word inflection and lemmatization

    from textblob import TextBlob

    def get_sentiment(text):
        sentiment = TextBlob(text).sentiment.polarity
        return sentiment

    df['sentiment'] = df['clean_text'].apply(get_sentiment)

* from textblob import TextBlob:
    
    Imports the TextBlob class from the TextBlob library.
* def get_sentiment(text)::

    Defines a function get_sentiment that takes a text string as input.
* TextBlob(text).sentiment.polarity:

    Creates a TextBlob object from the input text.

    Accesses the sentiment property, which contains sentiment analysis results:
    
    polarity: A float value in the range [-1.0, 1.0].
        
      -1.0: Extremely negative sentiment.
       0.0: Neutral sentiment.
       1.0: Extremely positive sentiment.
* return sentiment:

    Returns the calculated polarity score.
* df['sentiment'] = df['clean_text'].apply(get_sentiment):

    Applies the get_sentiment function to each entry in the clean_text column of the DataFrame df.
    
    Stores the resulting sentiment polarity values in a new column called sentiment.

      def count_mentions(text, tickers):
        return sum(text.count(ticker) for ticker in tickers)

      tickers = ['AAPL', 'TSLA', 'AMZN']  # List of stock tickers
      df['mention_count'] = df['clean_text'].apply(lambda x: count_mentions(x, tickers))
      df
The provided code calculates how many times specific stock tickers (like AAPL, TSLA, and AMZN) are mentioned in the clean_text column of a DataFrame. The results are stored in a new column called mention_count.

* def count_mentions(text, tickers):

    This function takes two arguments:
          
        text: A string where the mentions will be counted.
        tickers: A list of stock ticker symbols to search for in the text.
* text.count(ticker):

    For each ticker in the list, the count() method counts its occurrences in the given text.
* sum(..):

    Adds up the counts for all tickers, returning the total number of mentions for the text.
* tickers = ['AAPL', 'TSLA', 'AMZN']:

    A predefined list of stock tickers to search for.
* df['mention_count'] = df['clean_text'].apply(lambda x: count_mentions(x, tickers)):

    Applies the count_mentions function to each entry in the clean_text column.

    The lambda function passes the text from the DataFrame and the tickers list to the count_mentions function.
* The results are stored in a new column, mention_count.
## Model Building
   
    pip install yfinance

yfinance (short for Yahoo Finance) is a Python library that provides easy access to financial data from Yahoo Finance. It is commonly used to retrieve stock price data, historical prices, financial statements, and other market-related data for analysis and algorithmic trading.

    import yfinance as yf
    stock_data = yf.download("X", start="2023-01-01", end="2023-12-31")
    * Here X refers twitter
The download method is used to fetch historical data."X": The stock ticker symbol (ensure it's valid for the intended stock).start="2023-01-01": Specifies the start date for the data.end="2023-12-31": Specifies the end date for the data.
* stock_data will stores twitter one year data. It returns,
->Date: The index of the DataFrame.

->Open: Opening price for the day.

->High: Highest price during the day.

->Low: Lowest price during the day.

->Close: Closing price for the day.

->Adj Close: Adjusted closing price (corrected for corporate actions like splits and dividends).

->Volume: Number of shares traded.

    stock_data['price_change'] = stock_data['Close'].pct_change()  # Percentage change
    stock_data['movement'] = stock_data['price_change'].apply(lambda x: 1 if x > 0 else 0)
    stock_data
* price_change : This column computes the percentage change in the closing price from one day to the next.The method .pct_change() calculates the relative change between the current and the previous row:
Percentage Change = (Current Close Price−Previous Close Price)/P
previous Close Price

* movement :  This column indicates whether the stock's closing price increased or not:
    
    ->1 if the percentage change (price_change) is positive (price increased).
    ->0 if the percentage change is zero or negative (price did not increase).
The transformation is applied using a lambda function with .apply()
* For the first row (price_change is NaN), there is no previous day to compare with, so it remains NaN.
* The movement column encodes whether the price went up (1) or stayed flat/went down (0).
* stock_data will gives the total dataset regarding stock movements in twitter.
     
      df=stock_data.copy()
      df.head()
This creates a deep copy of the stock_data DataFrame, ensuring that changes made to df do not affect stock_data (and vice versa).
     
      mean_value = df['price_change'].mean()
      * Fill NaN values with the mean
      df['price_change'].fillna(mean_value, inplace=True)
-> This computes the average (mean) of the non-NaN values in the 'price_change' column

->The fillna() method replaces NaN values with the provided value (in this case, mean_value). The inplace=True ensures the operation modifies the original DataFrame instead of returning a new one. 

    df.isna().sum()
* It indicates whether df is having null values or not,it shows as either 0 or 1.
    
      df['movement'].value_counts
*  The df['movement'].value_counts() snippet is used in Pandas to count the occurrences of unique values in the 'movement' column of a DataFrame. It returns a Series with the unique values as the index and their respective counts as the values. 
  ->This is helpful for analyzing categorical data or checking the distribution of values in a column.
    
    df=df[1:]
    df.shape

->This slices the DataFrame, dropping the first row (index 0).
df[1:] selects all rows starting from index 1 up to the end.

->Returns a tuple indicating the dimensions of the DataFrame as (number_of_rows, number_of_columns
* we need to intsall seaborn library by using pip install seaborn
    
      pip install seaborn
      import seaborn as sns 
      for i in df.select_dtypes(include='number').columns:
        sns.boxplot(df[i])
      plt.show()

->Filters columns with numeric data types from the DataFrame.

->Creates a boxplot for the current numeric column i.

->plt.show(): Ensures that each plot is displayed before moving to the next.

### Handling Outliers and fill with a value using IQR(Interquantile range)

    def out(col):
        q1,q3=np.percentile(col,[25,75])
        iqr=q3-q1 
        ll=q1-1.5*iqr 
        uw=q3+1.5*iqr 
        return ll,uw 
    for i in df.select_dtypes(include='number').columns:
        ll,uw=out(df[i])
        df[i]=np.where(df[i]<ll,ll,df[i])
        df[i]=np.where(df[i]>uw,uw,df[i])

* Computes the IQR and calculates the lower limit (ll) and upper limit (uw) for outlier detection.

    ->q1 and q3 are the 25th and 75th percentiles, respectively.
    
    ->IQR is calculated as q3 - q1.

    ->Lower limit (ll) is q1 - 1.5 * IQR.

    ->Upper limit (uw) is q3 + 1.5 * IQR.
* For loop over numeric columns:

    ->Calculates ll and uw for each numeric column.
    
    ->Replaces values below ll with ll and values above uw with uw using np.where.

      for i in df.select_dtypes(include='number').columns:
        sns.boxplot(df[i])
        plt.show()
* Here we show the plots having no outliers that means ready to predictions or model building.  

      df = df.astype(int) # convert all columns to int

      df.columns = df.columns.set_levels(
      #The method set_levels() is used on df.columns without first verifying that df.columns has a multi-level index (MultiIndex).
      df.columns.levels[1].str.replace('X', 'Twitter'), level='Ticker'
      ) 
* We replace x with twitter because of our convenient.
      
      df=df.drop(columns=['price_change']) 
      # we drop the price_change column because of in that column the values are 0 only so there is no use for that we drop that column.
      
      x=df.drop(columns='movement')
      y=df['movement'] 
      x
-> Drop the movement column from the dataframe and returns new dataframe without movement column.

-> Extracts the movement column as a series and assign it to y,y contains only the movement column.

    from sklearn.model_selection import train_test_split
    from sklearn.metrics import confusion_matrix,accuracy_score,precision_score,recall_score,f1_score
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=42)

->The train_test_split function of the sklearn. model_selection package in Python splits arrays or matrices into random subsets for train and test data, respectively.

-> we will import accuracy_score,precision_score,recall_score,f1_score from sklearn.metrics.
   
   * accuracy_score:- Calculates the ratio of correctly predicted instances to the total instances.

   * precision_score:- Measures the proportion of true positive predictions among all positive predictions (relevant in imbalanced datasets).
   * recall_score:- Measures the proportion of actual positives correctly predicted (also called sensitivity).
   * f1_score:- Combines precision and recall into a single metric (harmonic mean) to balance false positives and false negatives.

    from sklearn.linear_model import LogisticRegression
    li = LogisticRegression()
    li.fit(x_train, y_train)
* we need to import LogisticRegression from sklearn, we create an object for the LogisticRegression, fit is used to train a model on a given dataset.

      ac=accuracy_score(y_test,y_pred)
      print(ac)
      ps=precision_score(y_test,y_pred)
      print(ps)
      f1=f1_score(y_test,y_pred)
      print(f1)
      rc=recall_score(y_test,y_pred)
      print(rc)
* from the above code we get accuracy_score is 0.52, precision_score is 0.5416666666666666, f1_score is 0.6842105263157895 and recall_score is 0.9285714285714286.

      from sklearn.neighbors import KNeighborsClassifier
      knn=KNeighborsClassifier()
      knn.fit(x_train,y_train)
      y_pred1=knn.predict(x_test)
* we need to import KNeighborsClassifier from sklearn, we create an object for the KNeighborsClassifier, fit is used to train a model on a given dataset.

      ac1=accuracy_score(y_test,y_pred1)
      print(ac1)
      ps1=precision_score(y_test,y_pred1)
      print(ps1)
      f2=f1_score(y_test,y_pred1)
      print(f2)
      rc1=recall_score(y_test,y_pred1)
      print(rc1)

* from the above code we get accuracy_score is 0.66, precision_score is 0.6410256410256411, f1_score is 0.746268656716418 and recall_score is 0.8928571428571429.

      from sklearn.ensemble import RandomForestClassifier
      rd=RandomForestClassifier()
      rd.fit(x_train,y_train)
      y_pred2=rd.predict(x_test)

* we need to import RandomForestClassifier from sklearn, we create an object for the RandomForestClassifier, fit is used to train a model on a given dataset.

      ac2=accuracy_score(y_test,y_pred2)
      print(ac2)
      ps2=precision_score(y_test,y_pred2)
      print(ps2)
      f3=f1_score(y_test,y_pred2)
      print(f3)
      rc2=recall_score(y_test,y_pred2)
      print(rc2)
* from the above code we get accuracy_score is 0.54, precision_score is 0.631578947368421, f1_score is 0.5106382978723404 and recall_score is 0.42857142857142855.
### Same process as for Xgboost,DecisionTreeClassifier and svm mosels,the code as shown below
* For xgboost

      pip install xgboost
      from xgboost import XGBClassifier
      xg=XGBClassifier()
      xg.fit(x_train,y_train)
      y_pred3=xg.predict(x_test)

      ac4=accuracy_score(y_test,y_pred3)
      print(ac4)
      ps4=precision_score(y_test,y_pred3)
      print(ps4)
      f11=f1_score(y_test,y_pred3)
      print(f11)
      rc4=recall_score(y_test,y_pred3)
      print(rc4)

* from the above code we get accuracy_score is 0.58, precision_score is 0.64, f1_score is 0.6037735849056604 and recall_score is 0.5714285714285714.

* For DecisionTree
      
      #DecisionTreeClassifier
      from sklearn.tree import DecisionTreeClassifier
      dt=DecisionTreeClassifier()
      dt.fit(x_train,y_train)
      y_preda=dt.predict(x_test)

      a=accuracy_score(y_test,y_preda)
      print(a)
      p=precision_score(y_test,y_preda)
      print(p)
      f=f1_score(y_test,y_preda)
      print(f)
      r=recall_score(y_test,y_preda)
      print(r)
    
* from the above code we get accuracy_score is 0.48, precision_score is 0.55, f1_score is 0.4583333333333333 and recall_score is 0.39285714285714285.
    
* For svm
      
      # Import the SVC model
      from sklearn.svm import SVC

      # Create an SVC model instance
      svm_model = SVC()

      # Fit the model to training data
      svm_model.fit(x_train, y_train)

      # Predict on test data
      y_pred_4= svm_model.predict(x_test)

      a1=accuracy_score(y_test,y_pred_4)
      print(a1)
      p1=precision_score(y_test,y_pred_4)
      print(p1)
      f1=f1_score(y_test,y_pred_4)
      print(f1)
      r1=recall_score(y_test,y_pred_4)
      print(r1)

* from the above code we get accuracy_score is 0.56, precision_score is 0.56, f1_score is 0.717948717948718 and recall_score is 1.0.

#### Based on the above outcomes means based on accuracy and recall score I think the best model is LogisticRegression.

## How to run the code:
* Download and install vs code from the official website.
* Download and install Python from the official Python website. Add Python to the system PATH during installation.
* Open vs code, go to extensions we install some extensions which is used for running like python extension,jupyter etc...
* Install all ML libraries like numpy, pandas, scikit-learn, matplotlib etc..
* Clock on Run all to run all cells.


## Deployment :
* First we nedd to save all the files in one folder,upload that folder into vs code open app.py file and run the code and take new terminal, type python -m run app.py in the terminal.
* It open edge sheet where we enter the values enter all the values.
* It gives prediction either Positive or Negative. 
