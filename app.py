# app.py

# app dependency setup
import os
import io
from flask import Flask
from flask import render_template, request, jsonify
import logging
from werkzeug.utils import secure_filename

# tensorflow dependency setup
# import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import img_to_array
# from tensorflow.keras.applications import imagenet_utils
# from PIL import Image
import numpy as np

# dimensions of chest x-ray images
img_width, img_height = 224, 224

# load the model
# model = load_model('static/example.h5')
# model.compile(loss='binary_crossentropy',
#               optimizer='rmsprop',
#               metrics=['accuracy'])

# load images
# img = image.load_img('chest_xray/val/VIRAL/person994_virus_1672.jpeg', target_size=(img_width, img_height))
# x = image.img_to_array(img)
# x = np.expand_dims(x, axis=0)

# predict classes
# images = np.vstack([x])
# classes = model.predict_classes(images, batch_size=10)
# print(classes)

# set the upload folder
UPLOAD_FOLDER = 'static/upload/'

# initialize the Flask app with config
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if 'DYNO' in os.environ:
    app.logger.addHandler(logging.StreamHandler(sys.stdout))
    app.logger.setLevel(logging.ERROR)

# initialize model
model = None

# default route - load landing page
@app.route("/")
def initial():
    app.logger.info("Loading initial route")
    return render_template("index.html", msg="Before Prediction")

def load_model():
    # load the pre-trained Keras Pneumonia model which was
    # previously trained, and saved
    global model
    model = load_model('static/model_xray.h5')

def prepare_image(image, target):
    # target: size of the target image
    # if the image mode is not RGB, convert it
    # if image.mode != "RGB":
    #     image = image.convert("RGB")

    # resize the input image and preprocess it
    image = image.resize(target)
    # convert Image to numpy array
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    # image = imagenet_utils.preprocess_input(image)

    # return the processed image
    return image

# route to handle the uploaded image
@app.route("/predict", methods=["POST"])
def predict():
    # initialize the data dictionary that will be returned
    # from the view
    data = {"success": False}

    # ensure an image was properly uploaded to our endpoint
    if request.method == "POST":
        if request.files.get("img"):
            # read the image in PIL format
            # image = request.files["img"].read()
            # image = Image.open(io.BytesIO(image))

            # store image in directory and load it back to model
            image = request.files["img"]
            filename = secure_filename(image.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            image.save(filepath)

            image = image.load_img(filepath, target_size=(img_width, img_height))

            # preprocess the image and prepare it for classification
            image = prepare_image(image, target=(img_width, img_height))

            # get an array having the image
            image = np.vstack([image])

            # classify the input image and then initialize the list
            # of predictions to return to the client
            preds = model.predict_classes(image, batch_size=10)
            # results = imagenet_utils.decode_predictions(preds)
            print(preds)
            # data["predictions"] = []

            # # loop over the results and add them to the list of
            # # returned predictions
            # for (imagenetID, label, prob) in results[0]:
            #     r = {"label": label, "probability": float(prob)}
            #     data["predictions"].append(r)

            # indicate that the request was a success
            data["success"] = True
            data["preds"] = preds
            data["msg"] = "Prediction done."

    # return the data dictionary as a JSON response
    # return Flask.jsonify(data)
    return render_template("index.html", data)

# run the Flask application
if __name__ == "__main__":
    app.logger.info("Loading the Keras model and starting Flask server, please wait...")
    # load model
    load_model()
    # run server
    app.run(
        host='0.0.0.0', 
        port=9000, 
        debug=True)