# Step 1 - Authenticate
consumer_key= 'consumer_key'
consumer_secret= 'consumer_secret'

access_token='access_token'
access_token_secret='access_token_secret'

import tweepy

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth)

print('authenticated - Twitter')

public_tweets = api.search('corona')

print( len(public_tweets) )

print( public_tweets[0]._json)

# for key , value in public_tweets[0].items():
#     print(key,value,'\n')
