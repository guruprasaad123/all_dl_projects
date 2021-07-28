from flask import Flask, render_template, send_from_directory, request, redirect, jsonify, url_for, flash
import json
from deeppavlov import configs, train_model
from deeppavlov.core.common.file import read_json
from flask_cors import CORS

# model_config = read_json(configs.faq.tfidf_logreg_en_faq)

faq = train_model('./configs/tfidf_logreg_en_faq.json')

# initializing a flask app
app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
CORS(app)

@app.route('/api',methods=['POST'])
def api():
  # checking if request is POST , is of type JSON
  if request.method == 'POST' and request.is_json:
    json_val = request.json
    query = json_val['query']
    print('query => ',query)
    ans = faq([query])
    print(ans)
    return json.dumps({'response' : ans}) , 200

  else:
    return json.dumps({'error':True}) , 401

if __name__ == '__main__':
  app.secret_key = 'super_secret_key'
  app.debug = True
  app.use_reloader=False
  app.run(host='0.0.0.0', port=4000)