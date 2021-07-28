# Step 1 - Authenticate
consumer_key= 'consumer_key'
consumer_secret= 'consumer_secret'

access_token='access_token'
access_token_secret='access_token_secret'


from twitter import Api

api = Api(consumer_key=consumer_key,
          consumer_secret=consumer_secret,
          access_token_key=access_token,
          access_token_secret=access_token_secret)

results = api.GetSearch( raw_query="max_id=1244222306924351487&q=corona&count=10&include_entities=1&result_type=recent" , return_json=True )

print(len(results['statuses']))
# print(type(results[0]))
print(results )

# for key , value in results[0].AsDict().items() :
#     print(key , value ,"\n" )
