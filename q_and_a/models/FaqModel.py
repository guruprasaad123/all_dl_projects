import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import os
import pandas as pd
import re
graph = tf.get_default_graph()

class FaqModel():

    model = ""

    def __init__(self,db_url,model_url="https://tfhub.dev/google/universal-sentence-encoder/4"):

        # "../data/zeamed-web/zeamed-faq.csv"
        self.init(model_url)
        self.get_data(db_url)
        self.init_model()


    def init(self , module_url):
        
        try:
        
            # module_url = "https://tfhub.dev/google/universal-sentence-encoder/4" #@param ["https://tfhub.dev/google/universal-sentence-encoder/4", "https://tfhub.dev/google/universal-sentence-encoder-large/5"]
            # module_url = "../data"
            print('tf =>',tf.__version__)
            print('module_url => ',module_url)
            self.model = hub.load(module_url)
            print ("module %s loaded" % module_url)
            # def embed(input):
            # return model(input)
            return self.model
        
        except (OSError, IOError) as e:
            print('Error => ',e)

    def get_data(self , db_url):

        try:

            self.df = pd.read_csv(db_url)
            print("DB loaded %s " % (db_url) )
            return self.df

        except (OSError, IOError) as e:
            print('Error => ',e)

    def init_model(self):
        try :
            
            query = tf.placeholder( tf.string , name="Query")
            self.embed_query = self.model( query )
            
            question = tf.placeholder( tf.string , name="Question" )
            self.embed_questions = self.model( self.df.Question  )
            
        except (OSError, IOError) as e:
            print('Error => ',e)


    def predict(self , Query ):
        try :

            # query = tf.placeholder( tf.string )
            # embed_query = self.model( query )
            
            # question = tf.placeholder( tf.string )
            # embed_questions = self.model( question )

            questions = []
            for ques in self.df.Question:
                questions.append(ques)

            print(questions)
            
            with tf.Session(graph=graph) as session:
                session.run( tf.global_variables_initializer() )
                session.run( tf.tables_initializer() )

                question_matrix , query_matrix = session.run([self.embed_questions , self.embed_query] , feed_dict = {
                    'Query:0' : [Query]
                })

                print( 'question runned ',question_matrix )
                print( query_matrix.shape )
                product = np.inner(query_matrix , question_matrix)
                # product = product.reshape(-1,1)
                
                max_score = 1e-10
                choosen_question = "Please provide a valid response"

                for question , score in zip( self.df.Question , product[0].tolist()):
                                        
                    if ( round(score,2) > 0.50 and max_score < round(score,2) ):
                        max_score = score
                        choosen_question = question

                print('choosen question => ',choosen_question,max_score)
                if max_score == 1e-10 :
                    return { 
                        'question' : Query , 
                        'answer' : choosen_question , 
                        'score' : max_score 
                        }
                else :
                    choosen = self.df.Question == choosen_question
                    ques_ans = self.df[choosen]
                    
                    print('ques_ans => ',ques_ans.Answer.tolist()[0])

                    return {
                        'question' : ques_ans.Question.tolist()[0] ,
                        'answer' :ques_ans.Answer.tolist()[0] ,
                        'score' : max_score
                        }

        except (OSError, IOError) as e:
            print('Error => ',e)