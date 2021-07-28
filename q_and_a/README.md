
## Task

This app will try to answer to your question , if its able to find any match with the existing question (in our application).

if we ask any question related to our app , this app would try to match the that question with the question & answer mapping based on the semantic similarity

like *what is your age ?*  will be matching with *how much old are you?* and its relative answer will be fetched & returned as response


## Running the application


## frontend

- move to [faqapp](faqapp) in the root folder

### Installation

- install all the neccesary packages for running the application using the command below
  - `npm i`

### Running

- to run the application issue the command below
  - `npm run start`
- the application will start running locally in `http://localhost:4200`

## backend

note : we are using `python3` in the project , all versions >= 3.6 are supported

### Installation

- install all the packages for our backend application using the command below
  - `pip3 install -r requirements.txt`

### Running

- to run the application , use this command below
  - `python3 server.py`
- the application will start running locally in `http://localhost:4000`

### Test

Query as `JSON` from the frontend

```javascript
{
    'query':'describe about the productA ?'
}
```

Processing the `JSON` input in the backend

```python
choosen question =>  What is productA 0.6889529228210449
ques_ans =>  productA is a popular online price and quality transparency platform that helps consumers to shop for healthcare services like MRI
{'question': 'What is productA', 'answer': 'productA is a popular online price and quality transparency platform that helps consumers to shop for healthcare services like MRI', 'score': 0.6889529228210449}
```

## Reference

- [Universal Sentence Encoder](https://www.tensorflow.org/hub/tutorials/semantic_similarity_with_tf_hub_universal_encoder)

- [universal-sentence-encoder | tfhub](https://tfhub.dev/google/universal-sentence-encoder/1)

- [Use-cases of Google’s Universal Sentence Encoder in Production | towardsdatascience](https://towardsdatascience.com/use-cases-of-googles-universal-sentence-encoder-in-production-dd5aaab4fc15)

- [Semantic Similarity with TF-Hub Universal Encoder | colab](https://colab.research.google.com/github/tensorflow/hub/blob/master/examples/colab/semantic_similarity_with_tf_hub_universal_encoder.ipynb)

- [What is Universal Sentence Encoder and how it was trained](https://www.dlology.com/blog/keras-meets-universal-sentence-encoder-transfer-learning-for-text-data/)
