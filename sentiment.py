import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from jupyterthemes import jtplot
from wordcloud import WordCloud, STOPWORDS
import string
import nltk 
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

# set jupyter theme
jtplot.style(theme='monokai', context='notebook',ticks = True, grid=False)

# Read in the csv file
x_df = pd.read_csv('tweets.csv')

# Drop the column ID
x_df = x_df.drop(['id'],axis=1)

# Optional: Get a summary of the info in the dataframe & check for empty values
#print(x_df.info())

#  Optional: Print the descriptive statistics
#print(x_df.describe())

# Plot histogram to analyze distribution
x_df.hist(bins = 30, figsize = (10,5), color = 'b')

# Create new column for the length of the message
x_df['length'] = x_df['tweet'].apply(len)

# Create a list of positive words
positive = x_df[x_df['label'] == 0]
positive_list = positive['tweet'].tolist()
positive_sentences_joined = " ".join(positive_list)

# Create a list of negative words
negative = x_df[x_df['label'] == 1]
negative_list = negative['tweet'].tolist()
negative_sentences_joined = " ".join(negative_list)

# Define a list of stopwords
stopwords = set(STOPWORDS)
stopwords.update(["user", "amp"]) 

def message_cleaning(message):
    # Remove punctuation
    cleaned_message = ''.join([char for char in message if char not in string.punctuation])
    # Return cleaned message
    return cleaned_message

# Apply message cleaning function to the 'tweet' column of x_df
x_df_clean = x_df['tweet'].apply(message_cleaning)

# Create a WordCloud object with custom stopwords
wordcloud = WordCloud(stopwords=stopwords)

# Plot positive word cloud
positive_cleaned = ' '.join(positive_list)
plt.figure(figsize=(10,10))
plt.imshow(wordcloud.generate(positive_cleaned))
plt.show()

# Plot the negative word cloud
negative_cleaned = ' '.join(negative_list)
plt.figure(figsize=(10,10))
plt.imshow(wordcloud.generate(negative_cleaned))
plt.show()

# Define the cleaning pipeline 
vectorizer = CountVectorizer(analyzer = message_cleaning, dtype = np.uint8)
x_countvectorizer = vectorizer.fit_transform(x_df['tweet'])

# x is a df containing the features for each tweet
X = pd.DataFrame(x_countvectorizer.toarray())

# y is a series containing the corresponding target labels for each tweet
y = x_df['label']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train the Multinomial Naive Bayes classifier
NB_classifier = MultinomialNB()
NB_classifier.fit(X_train, y_train)

# Predicting the Test set results
y_predict_test = NB_classifier.predict(X_test)
cm = confusion_matrix(y_test, y_predict_test)

# Visualize the confusion matrix
sns.heatmap(cm, annot=True)

# Print classification report
print(classification_report(y_test, y_predict_test))

