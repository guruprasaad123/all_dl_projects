# settings.py
from dotenv import load_dotenv
import os
from pathlib import Path  # python3 only

env_path = Path('.') / '.env'
load_dotenv(dotenv_path=env_path)

import json

consumer_key = os.getenv("consumer_key")
consumer_secret = os.getenv("consumer_secret")
access_token = os.getenv("access_token")
access_token_secret = os.getenv("access_token_secret")

from twitter import Api

from googletrans import Translator
from textblob import TextBlob

from sklearn.feature_extraction.text import TfidfVectorizer
from corextopic import corextopic as ct
import pandas as pd

class twitter_api():

    def __init__(self):
        self.api = Api(consumer_key=consumer_key,
          consumer_secret=consumer_secret,
          access_token_key=access_token,
          access_token_secret=access_token_secret)
        self.translator = Translator()

    def topic_modelling(self , statuses):
        
        dataframe = pd.DataFrame(statuses)

        vectorizer = TfidfVectorizer(
            max_df=.5,
            min_df=1,
            max_features=None,
            ngram_range=(1, 2),
            norm=None,
            binary=True,
            use_idf=False,
            sublinear_tf=False
        )
        vectorizer = vectorizer.fit(dataframe['text'])
        tfidf = vectorizer.transform(dataframe['text'])
        vocab = vectorizer.get_feature_names()

        # Anchors designed to nudge the model towards measuring specific genres
        anchors = [
            ["money","fund"],
            ["emergency"],
            ["recovered"],
            ["treatment"]
        ]
        anchors = [
            [a for a in topic if a in vocab]
            for topic in anchors
        ]

        model = ct.Corex(n_hidden=4, seed=42)
        
        model = model.fit(
            tfidf,
            words=vocab,
            anchors=anchors, # Pass the anchors in here
            anchor_strength=3 # Tell the model how much it should rely on the anchors
        )

        topic_df = pd.DataFrame(
            model.transform(tfidf), 
            #columns=["topic_{}".format(i+1) for i in range(4)]
            columns=  ["money","emergency","recovered","treatment"]
        ).astype(float)

        topic_df.index = dataframe.index

        print(dataframe.shape)
        print(topic_df.shape)


        def filter(x):
            columns = ["money","emergency","recovered","treatment"]
            str = []
            for col in columns:
                if bool(x[col]):
                    str.append(col)
            return str

        obj= {}

        obj["topics"] = topic_df.apply( filter , axis=1 )

        obj =  pd.DataFrame(obj)

        final = pd.concat([dataframe, obj], axis=1)

        print('final_topic_modelling => ',final.shape)
        
        # return final
        return final.to_json(orient="records" )

    def translate(self , statuses ):

        tweets = []

        for status in statuses:
            
            if status['lang'] =='en':
                                    
                blob = TextBlob( status['text'] )
                    
                status['sentiment'] = {
                    'polarity' : blob.sentiment.polarity,
                    'subjectivity' : blob.sentiment.subjectivity
                    }
                
                print('sentiment => ',blob.sentiment)

                tweets.append(status)
            else:

                print('status => language ', status['lang']  )
                try :

                    # An attempt to translate using 'googletrans'

                    # success = self.translator.translate( status['text'] , src=status['lang'] , dest='en')
                    # print('success => ',success)
                    # status['text'] = success.text
                    
                    blob = TextBlob( status['text'] )
                    
                    status['sentiment'] = {
                        'polarity' : blob.sentiment.polarity,
                        'subjectivity' : blob.sentiment.subjectivity
                    }
                    
                    print('sentiment => ',blob.sentiment)
                    
                    text = blob.translate(to='en')
                    print('text =>' ,text , type(text))
                    status['text'] = str(text)

                    tweets.append(status)

                except Exception as Error :
                    print('error : ',Error)

            
            # tweets.append(status)

        return tweets 

    
    def search_tweets(self,  q='corona' , geocode=(), max_id=None , count=10, result_type="recent" ):

        results = self.api.GetSearch( 
                        raw_query="q=%s&count=%d&include_entities=1&result_type=%s"%(q , count , result_type) ,
                        max_id = max_id ,
                        geocode= geocode ,
                        return_json=True 
                        )

        length = len(results['statuses'] )

        statuses = results['statuses']

        statuses = self.translate(statuses)

        statuses = self.topic_modelling(statuses)

        print('tweet => ',statuses[0])

        return {
            'tweets' : statuses ,
            'length' : length,
            'search_metadata' : results['search_metadata']
        }

