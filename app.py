from __future__ import division, print_function
import sys
import os
import glob
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
global graph
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
from tensorflow.keras import backend
from keras.models import Sequential
from skimage.transform import resize
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
from flask import Flask, render_template

app = Flask(__name__, template_folder='template')
MODEL_PATH = 'skindisease.h5'
model = load_model(MODEL_PATH) 
@app.route('/', methods=['GET']) 
def index():
	return render_template('index.html')
@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
       f = request.files['file']
       basepath = os.path.dirname(__file__)
       file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
       f.save(file_path)
       img = image.load_img(file_path, target_size=(64, 64)) 
       x = image.img_to_array(img) 
       x = np.expand_dims(x, axis=0)
       with graph.as_default():  
           preds = model.predict_classes(x) 
       index = ['Eczema','melanoma','psoriasis','rosacea','Skin cancer']
       text = "prediction : "+index[preds[0]]
       return text 
   
if __name__ == '__main__':
           app.run(debug=False,threaded=False)
           


