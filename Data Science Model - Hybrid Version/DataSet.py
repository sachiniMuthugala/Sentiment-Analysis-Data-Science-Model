import tweepy
import csv

# credentials
consumer_key = "Kvn4yF69hy36nHaLO0nrt3Ppo"
consumer_secret = "EFspi2PIfQZb11tH7bVnbU1Xt8bV3AVkJTDPpr9q4QqBvXmyR6"
access_token = "1021170438582415360-9CqvAE16o5LSSEj18WeCzqljY3jpzg"
access_token_secret = "2chS3ZRpC3GIBU1zQB4U99jBJDAy9WeHIGKv9s21OF0UG"

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth, wait_on_rate_limit=True)

# Open/Create a file to append data
csvFile = open('Twylist Telecom Dataset.csv', 'a')
# Use csv Writer
csvWriter = csv.writer(csvFile)

keywords = ["wifi", "dialog", "mobitel", "airtel", "router", "#SLT", "broadband", "telecom"]


for i in range(len(keywords)):
    for tweet in tweepy.Cursor(api.search,q=i,count=10000,
                               lang="en",
                               since="2017-04-03").items():
        print (tweet.created_at, tweet.text)
        csvWriter.writerow([tweet.created_at, tweet.text.encode('utf-8')])


