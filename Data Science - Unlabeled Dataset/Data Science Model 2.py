#!/usr/bin/env python
# coding: utf-8

# In[1]:


import csv
import tweepy
import pandas as pd
import re
import string
import nltk
import time
from nltk.corpus import words
from textblob import TextBlob

import seaborn as sns
import matplotlib.pyplot as plt

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer,TfidfTransformer
from sklearn import preprocessing

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.naive_bayes import MultinomialNB


from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# In[2]:


# set the path of the data set
df1 = pd.read_csv("/Users/sachinim/Desktop/TelecommunicationData1.csv" ,header = None)
df1.columns = ['date', 'tweet']
df2 = pd.read_csv("/Users/sachinim/Desktop/TelecommunicationData2.csv" ,header = None)
df2.columns = ['date', 'tweet']

df = df1.append(df2, ignore_index = True)


# In[3]:


df.columns = ['date', 'tweet']


# In[4]:


df=df.drop_duplicates(subset='tweet', keep="last")


# In[5]:


pd.set_option('display.max_colwidth', None)
df


# In[6]:


words = set(nltk.corpus.words.words())


# In[7]:


def text_lemmatizer(text):
  # This function is used to lemmatize the given sentence
    lemmatizer =  WordNetLemmatizer ()
    token_words = word_tokenize(text)
    stem_sentence = []
    for word in token_words:
        stem_sentence.append(lemmatizer.lemmatize(word))
    return " ".join(stem_sentence)


# In[8]:


# Remove special characters from the tweet
def clean_text(tweet):
    
    tweet = tweet.lower()
    
    # have to remove "b'RT @endaburke81" at the begining of the tweet
    if(tweet[:4]=="b'rt"):
        tweet = tweet.split(":", 1)[1]

    # splitting the tweet
    tweet = tweet.split()
    
    # Joining the tweet
    tweet = " ".join(tweet)
    
    #Removing digits and numbers
    tweet = "".join([i for i in tweet if not i.isdigit()])
    
    # Removing special characters from the tweet
    tweet = re.sub(f'[{re.escape(string.punctuation)}]', "", tweet)
    
    # cleaning = nltk.tokenize.wordpunct_tokenize(tweet)
    tweet = " ".join(w for w in nltk.wordpunct_tokenize(tweet) if w.lower() in words or not w.isalpha())
    
    tweet = text_lemmatizer(tweet)
    
    return tweet


# In[9]:


df["tweet_clean"] = df["tweet"].apply(clean_text)
df


# In[10]:


# sentiment analysis using polarity

df['sentiment'] = ' '
df['polarity'] = None
for i,tweets in enumerate(df.tweet_clean) :
    blob = TextBlob(tweets)
    df['polarity'][i] = blob.sentiment.polarity
    if blob.sentiment.polarity > 0 :
        df['sentiment'][i] = 'positive'
    elif blob.sentiment.polarity < 0 :
        df['sentiment'][i] = 'negative'
    else :
        df['sentiment'][i] = 'neutral'
df.head()


# In[11]:


df.loc[df['sentiment'] == "negative"].shape


# In[12]:


df.loc[df['sentiment'] == "positive"].shape


# In[13]:


df.loc[df['sentiment'] == "neutral"].shape


# In[14]:


df.loc[df['sentiment'] == "negative"]


# In[15]:


#Split the data

x_train, x_test, y_train, y_test = train_test_split(df["tweet_clean"], df["sentiment"],test_size=0.2, random_state=0)


# In[16]:


# Tokenize Words in each tweets (Encoding)
# TfidfVectorizer converts text to word frequency vectors

tfidf_vectorizer = TfidfVectorizer(tokenizer=word_tokenize, token_pattern=None)
tfidf_vectorizer.fit(df["tweet_clean"])
x_train_vector = tfidf_vectorizer.transform(x_train)
x_test_vector = tfidf_vectorizer.transform(x_test)


# In[17]:


def evaluate_metrics(y_test, y_hat, model_type,time):
    
    accuracy = accuracy_score(y_hat, y_test)
    print("Model Type : ", model_type)
    print("\nAccuracy : ", format(accuracy, '.2f'))
    print("\nTraining Time : ", format(time, '.2f'), "s" )
    print("\n", classification_report(y_hat, y_test))

    
    plt.figure(figsize=(10,6))
    sns.heatmap(confusion_matrix(y_hat, y_test), annot=True, fmt=".2f")
    plt.show()
    return accuracy


# In[18]:


# Maximum Entropy (Logistic Regression Algorithm)

logisticR_model = LogisticRegression()
logisticR_start = time.time()
logisticR_model.fit(x_train_vector, y_train)
logisticR_stop = time.time()
logisticR_time = (logisticR_stop - logisticR_start)
logisticR_preds = logisticR_model.predict(x_test_vector)


# In[19]:


logisticR_accuracy = evaluate_metrics(logisticR_preds, y_test, "Logistic Regression Classifier", logisticR_time)


# In[20]:


# Random Forest Algorithm

RandomForest_model = RandomForestClassifier()
RandomForest_start = time.time()
RandomForest_model.fit(x_train_vector, y_train)
RandomForest_stop = time.time()
RandomForest_time = (RandomForest_stop - RandomForest_start)
RandomForest_preds = RandomForest_model.predict(x_test_vector)


# In[21]:


RandomForest_accuracy = evaluate_metrics(RandomForest_preds, y_test, "Random Forest Classifier", RandomForest_time)


# In[22]:


# Decision Tree Algorithm

DecisionTree_model = DecisionTreeClassifier(max_depth=20,random_state=0)
DecisionTree_start = time.time()
DecisionTree_model.fit(x_train_vector, y_train)
DecisionTree_stop = time.time()
DecisionTree_time = (DecisionTree_stop - DecisionTree_start)
DecisionTree_pred = DecisionTree_model.predict(x_test_vector)


# In[23]:


DecisionTree_accuracy = evaluate_metrics(DecisionTree_pred, y_test, "DecisionTree Classifier", DecisionTree_time)


# In[24]:


# Support Vector Machine Algorithm (Linear SVC classifier)

svm_model = svm.SVC(kernel='linear')
svm_start = time.time()
svm_model.fit(x_train_vector, y_train)
svm_stop = time.time()
svm_time = (svm_stop - svm_start)
svm_preds = svm_model.predict(x_test_vector)


# In[25]:


svm_accuracy = evaluate_metrics(svm_preds, y_test, "Support Vector Machine", svm_time)


# In[26]:


# Multinomial Naive Bayes Algorithm

MultinomialNB_model = MultinomialNB()
MultinomialNB_start = time.time()
MultinomialNB_model.fit(x_train_vector, y_train)
MultinomialNB_stop = time.time()
MultinomialNB_time = (MultinomialNB_stop - MultinomialNB_start)
MultinomialNB_preds = MultinomialNB_model.predict(x_test_vector)


# In[27]:


MultinomialNB_accuracy = evaluate_metrics(MultinomialNB_preds, y_test, "Multinomial NB Classifier", MultinomialNB_time)


# In[28]:


# Graph view of Accuracy

x = ["SVM","Logistic Regression","Random Forest", "DecisionTree Classifier", "Multinomial NB"]
y = [svm_accuracy,logisticR_accuracy,RandomForest_accuracy,DecisionTree_accuracy, MultinomialNB_accuracy]

plt.bar(x=x, height=y)
plt.title("Algorithms Accuracy")
plt.ylabel("Accuracy")
plt.xticks(rotation=15)
plt.xlabel("Algorithm Model")

plt.show()


# In[29]:


import pickle


# In[30]:


#save tfidf
with open("tfidf.pkl", 'wb') as file:
    pickle.dump(tfidf_vectorizer, file)


# In[31]:


#save model
with open("model.pkl", 'wb') as file:
    pickle.dump(svm_model, file)


# In[32]:


########### Load and get Predictions - Should goes to the backend 

with open("model.pkl", 'rb') as file:
    model = pickle.load(file)
    
with open("tfidf.pkl", 'rb') as file:
    tfidf = pickle.load(file)


# In[33]:


tweet = "better Service"
token=tfidf.transform([tweet])

print(model.predict(token)[0])


# In[34]:


tweet = "better Service"
token=tfidf.transform([tweet])

print(model.predict(token)[0])


# In[ ]:




