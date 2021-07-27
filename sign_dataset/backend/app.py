from flask import Flask, render_template, send_from_directory, request, redirect, jsonify, url_for, flash
import json
from flask_cors import CORS , cross_origin
from werkzeug.utils import secure_filename
import os
from signs_api.model import api
import base64
import numpy as np
from PIL import Image
from io import BytesIO

def parse_img(image,to_numpy=False):

    image = image[image.find(",") + 1 :]
    dec = base64.b64decode(image + "===")

    image = Image.open( BytesIO( dec ) )
    image = image.convert("RGB")

    if to_numpy == True:
        image = np.asarray(image)

    return image

sign_api = api()

# initializing a flask app
app = Flask(__name__)
# app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
# CORS(app ,  resources={r"/api/*": {"origins": "*"}} )
CORS(app)

UPLOAD_FOLDER = './uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/upload',methods=['GET', 'POST'])
# @cross_origin(allow_headers=['Content-Type'])
def upload():

    """Returns the tweets based on user's search

        If the argument `sound` isn't passed in, the default Animal
        sound is used.

        Parameters
        ----------
        search : str
            String that you want to twitter_api searches

        next_pg : int
            1 , 0

        max_id : int
            For navigating to next_pg page , tweets

        Raises
        ------
        NotImplementedError
            If no sound is set for the animal or passed in as a
            parameter.
        """
    if request.method == 'POST':
        print('inside upload')
        # check if the post request has the file part

        data = request.json

        image = data['image']

        image = parse_img( image )

        prediction = sign_api.predict(image)

        return jsonify({'success':True , 'response' : prediction }) , 201
    else:
        return jsonify({'error':True}) , 401

@app.route('/test',methods=['GET'])
def test():

    # ip = request.remote_addr or '106.51.240.30'
    # access_key = os.getenv('ipstack_access')

    # r = requests.get('http://api.ipstack.com/{}?access_key={}&format=1'.format(ip,access_key))

    # ipstack_response = r.json()

    # print('ipstack_response : ',ipstack_response)

    return jsonify({
        'success':True , 'response' : 'working Alright'
        # , 'ip' : ip ,'ip_response' : ipstack_response
     }) , 201


@app.route("/")
def hello():
    return "<h1 style='color:blue'>Hello There!</h1>"


if __name__ == '__main__':
  # upload location
  app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
  # file size for uploading
  app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
  app.secret_key = 'super_secret_key'
  app.debug = True
  app.use_reloader=False
  # app.run(host='0.0.0.0', port=4000)
  app.run()
