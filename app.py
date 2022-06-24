import os
#import tensorflow as tf
import keras
import numpy as np
#from skimage import io
import cv2
import keras



# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
#from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()

# You can also use pretrained model from Keras
# Check https://keras.io/applications/

model =keras.models.load_model('av_model.h5',compile=False)
print('Model loaded. Check http://127.0.0.1:1000/')

def model_predict(img_path, model):
    img = cv2.imread(img_path)
    new_arr = cv2.resize(img,(100, 100))
    new_arr = np.array(new_arr/255)
    new_arr = new_arr.reshape(-1, 100, 100, 3)
    return model.predict(new_arr)


    '''img = keras.utils.load_img(img_path, grayscale=False, target_size=(100,100))
    show_img = keras.utils.load_img(img_path, grayscale=False, target_size=(100,100))
    x = keras.utils.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = np.array(x, 'float32')
    x /= 255
    preds = model.predict(x)
    return preds'''


@app.route('/plant-disease-diagnosis/', methods=['GET'])
def plantDiseaseDiagnosis():
    return render_template('plant-disease-diagnosis.html')

@app.route('/weather-forecasting/', methods=['GET'])
def weatherForecasting():
    return render_template('weather-forecasting.html')

@app.route('/plant-dictionary/', methods=['GET'])
def plantDictionary():
    return render_template('plant-dictionary.html')

@app.route('/farmer-assistant/', methods=['GET'])
def farmerAssistant():
    return render_template('farmer-assistant.html')


@app.route('/')
def index():
    # Main page
    return render_template('index.html')
    #return 'HEllo'


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)
        print(preds[0])

        # x = x.reshape([64, 64]);

        

        disease_class = ['Pepper___Bacterial_spot/काली मिर्च__बेल___बैक्टीरियल_स्पॉट', 'Pepper___healthy/स्वस्थ काली मिर्च ', 'Potato___Early_blight/आलू का अगेंती अंगमारी रोग',
                         'Potato___Late_blight/आलू का उत्तरभावी अंगमारी रोग', 'Potato___healthy/_स्वस्थ आलू ', 'Tomato_Bacterial_spot/टमाटर_बैक्टीरिया_स्पॉट ', 'Tomato_Early_blight/टमाटर का अगेंती अंगमारी रोग',
                         'Tomato_Late_blight/ टमाटर का उत्तरभावी अंगमारी रोग ', 'Tomato_Leaf_Mold/टमाटर_पत्ती_मोल्ड', 'Tomato_Septoria_leaf_spot/टमाटर_सेप्टोरिया_पत्ती_स्पॉट',
                         'Tomato Ring Spot Virus / टमाटर का रिंग स्पॉट वायरस', 'Tomato__Target_Spot / टमाटर का टारगेट स्पॉट ',
                         'Tomato__Tomato_YellowLeaf__Curl_Virus/ टमाटर का लीक कर्ल ', 'Tomato_mosaic_virus/ टमाटर का मोज़ाइक वायरस ', 'Tomato_healthy/टमाटर स्वस्थ ']
        #a = preds[0]
        #ind=np.argmax(a)
        print('Prediction:',disease_class[preds.argmax()])
        result=disease_class[preds.argmax()]
        return result
    return None


#if __name__ == '__main__':
    # app.run(port=5002, debug=True)

    # Serve the app with gevent
    #http_server = WSGIServer(('', 1000), app)
    #http_server.serve_forever()

    #http_server = WSGIServer(("0.0.0.0", 5000), app)
    #http_server.serve_forever()
    #app.run()

