# app.py

# app dependency setup
import os
import io
import sys
from flask import Flask
from flask import render_template, request, jsonify
import logging
from werkzeug.utils import secure_filename

# tensorflow dependency setup
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import img_to_array
# from tensorflow.keras.applications import imagenet_utils
from PIL import Image
import numpy as np

# dimensions of chest x-ray images
img_width, img_height = 224, 224

# set the path variables
PATH_UPLOAD_FOLDER = 'static/upload/'
PATH_MODEL = 'static/models/model_xray.h5'

# initialize the Flask app with config
app = Flask(__name__)
app.config['PATH_UPLOAD_FOLDER'] = PATH_UPLOAD_FOLDER

# configure the logger
if 'DYNO' in os.environ:
  app.logger.addHandler(logging.StreamHandler(sys.stdout))
  app.logger.setLevel(logging.ERROR)

# initialize model
model = None

# default route - load landing page
@app.route("/")
def initial():
  app.logger.info("Loading initial route")
  data = {"msg": "Before prediction."}
  return render_template("index.html", data=data)

# def load_model_custom():
  # load the pre-trained Keras Pneumonia model which was
  # previously trained, and saved
  
  # global model

  # load the Pneumonia Keras model
  # model = load_model('static/models/model_xray.h5')
  # compile the model for prediction
  # model.compile(
  #   loss='categorical_crossentropy',
  #   optimizer='adam',
  #   metrics=['accuracy'])
  # model.summary()

# route to handle the uploaded image
@app.route("/predict", methods=["POST"])
def predict():
  # initialize the data dictionary that will be
  # returned
  data = {"success": False}

  # ensure an image was properly uploaded to our endpoint
  if request.method == "POST":
    if request.files.get("img"):

      # load the stored Pneumonia Keras model, which was
      # previously trained separately
      model = load_model(PATH_MODEL)
      
      # compile the model for prediction
      model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy'])

      # summary of the model
      model.summary()

      # read the image in PIL format directly from 
      # the obtained Flask request object
      input_image = request.files["img"].read()
      input_image = Image.open(io.BytesIO(input_image))

      # convert image to Keras readable format
      # which is a Numpy array
      input_image = img_to_array(input_image)
      input_image = np.resize(input_image, (img_width, img_height, 3))
      input_image = np.expand_dims(input_image, axis=0)

      # classify the input image and then initialize the list
      # of predictions to return to the client
      preds = model.predict_classes(input_image)

      # indicate that the request was a success, and load 
      # the predictions for the template
      data["success"] = True
      # convert Numpy Array to Python list to make it JSON serializable
      data["predictions"] = preds.tolist()
      data["class"] = "No class predicted"
      data["msg"] = "Prediction done."

      # give the class name
      if preds[0] == 0:
        data["class"] = "Bacterial"
      elif preds[0] == 1:
        data["class"] = "Normal"
      elif preds[0] == 2:
        data["class"] = "Viral"

      # log the data variable
      app.logger.debug(data);

  # return render_template("index.html", data=data)
  return jsonify(data)

# run the Flask application
if __name__ == "__main__":

  app.logger.info("Loading the Keras model and starting Flask server, please wait...")

  app.run(
    host='0.0.0.0', 
    port=9000, 
    debug=True)



# ------------------
# ---- OLD CODE ----
# ------------------
# load images
# img = image.load_img('chest_xray/val/VIRAL/person994_virus_1672.jpeg', target_size=(img_width, img_height))
# x = image.img_to_array(img)
# x = np.expand_dims(x, axis=0)

# predict classes
# images = np.vstack([x])
# classes = model.predict_classes(images, batch_size=10)
# print(classes)
# ------------------
# ---- OLD CODE ----
# ------------------

# -----------------------------
# ---- STORE & LOAD METHOD ----
# -----------------------------
# # read the image, and store in filesystem
# input_image = request.files["img"]
# filename = secure_filename(input_image.filename)
# filepath = os.path.join(app.config['PATH_UPLOAD_FOLDER'], filename)
# input_image.save(filepath)
# input_image = image.load_img(filepath, target_size=(img_width, img_height))
# -----------------------------
# ---- STORE & LOAD METHOD ----
# -----------------------------