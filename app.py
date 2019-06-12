# Flask app dependency setup 
import os
import io
import sys
import logging
from werkzeug.utils import secure_filename
from flask import Flask, render_template, request, jsonify

# TensorFlow dependency setup
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.backend import get_session
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import img_to_array

# dimensions of chest x-ray images
img_width, img_height = 224, 224

# set the path variables
PATH_UPLOAD_FOLDER = 'static/upload/'

# initialize the Flask app instance
app = Flask(__name__)
app.config['PATH_UPLOAD_FOLDER'] = PATH_UPLOAD_FOLDER

# configure the logger
if True:
  app.logger.addHandler(logging.StreamHandler(sys.stdout))
  app.logger.setLevel(logging.ERROR)

# Load Model - NORMAL vs BACTERIAL vs VIRAL
# model_ALL = load_model("static/models/model_xray_v2.h5")

# Load Keras Model -  NORMAL vs PNEUMONIA
model_NP = load_model("static/models/NormalvsPneumonia-model_xray1.h5")
model_NP.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])
# model_NP = load_model("static/models/model_xray1.h5")
print("---------------First model loaded---------------")

# Load Keras Model - BACTERIAL vs VIRAL
model_BV = load_model("static/models/model_xray2.h5")
model_BV.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])
print("---------------Second model loaded---------------")

# load TensorFlow Graph for global model loading
graph = tf.get_default_graph()

# default route - load landing page
@app.route("/")
def initial():
  app.logger.debug("Loading initial route")
  data = {"msg": "Before prediction."}
  return render_template("index.html", data=data)

# route to handle the uploaded image
@app.route("/predict/NP", methods=["POST"])
def predictNP():
  # initialize the data dictionary that will be
  # returned back to the view
  data = {
    "class": "No Class Predicted",
    "confidence": 0.0,
    "error": "",
    "msg": "Prediction 1 Pending",
    "prediction": -1,
    "success": False
  }

  # ensure an image was properly uploaded to our endpoint
  if request.method == "POST":
    if request.files.get("img"):

      try:
        # read the image in PIL format directly from 
        # the obtained Flask request object
        input_image = request.files["img"].read()
        fn = request.files["img"].filename

        input_img = Image.open(io.BytesIO(input_image))
        # input_img = image.load_img(PATH_UPLOAD_FOLDER+fn, target_size=(img_width, img_height))

        # convert image to Keras readable format
        # which is a Numpy array
        input_image = img_to_array(input_img)
        input_image = np.resize(input_image, (img_width, img_height, 3))
        input_image = np.expand_dims(input_image, axis=0)
        final_image = np.vstack([input_image])

        # load the graph object, and models that are stored globally,
        # and use that graph for getting the model in each thread
        # global graph, model_NP
        with graph.as_default():
          # initialize parameters required for Tensor processing
          # get_session().run(tf.global_variables_initializer())

          model_NP = load_model("static/models/NormalvsPneumonia-model_xray1.h5")
          # model_NP.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
          # classify whether the image class is normal or pneumonia
          preds = model_NP.predict(final_image)
          print(preds)
          predClass = int(np.argmax(preds))
          # predClass = model_NP.predict_classes(input_image)[0]
          print(predClass)

          # data["confidence"] = '%.2f' % (preds[predClass]*100)
          data["msg"] = "Prediction done"
          
          # If predicted class is NORMAL
          if predClass == 0:
            data["success"] = True
            data["class"] = "Normal"
            data["prediction"] = 0
            app.logger.debug(data); # log the data variable
            return jsonify(data), 200 # send data back to view

          # If predicted class is PNEUMONIA
          elif predClass == 1:
            data["success"] = True
            data["class"] = "Pneumonia"
            data["prediction"] = 1
            app.logger.debug(data); # log the data variable
            return jsonify(data), 200 # send data back to view

      # If Exception or Error in server
      except Exception as e:
        data["error"] = "Some Server Error Occurred."
        app.logger.error(str(e))
        return jsonify(data), 500

  else:
    app.logger.debug("Non POST request at /predict")
    data["msg"] = "Not a POST request"
    data["error"] = "Forbidden"
    return jsonify(data), 403

# route to handle the uploaded image
@app.route("/predict/BV", methods=["POST"])
def predictBV():
  # initialize the data dictionary that will be
  # returned back to the view
  data = {
    "success": False,
    "class": "No Class Predicted",
    "msg": "Prediction 2 Pending",
    "prediction": -1,
    "error": ""
  }

  # ensure an image was properly uploaded to our endpoint
  if request.method == "POST":
    if request.files.get("img"):

      try:
        # read the image in PIL format directly from 
        # the obtained Flask request object
        input_image = request.files["img"].read()
        input_image = Image.open(io.BytesIO(input_image))
        fn = request.files["img"].filename

        # convert image to Keras readable format
        # which is a Numpy array
        input_image = img_to_array(input_image)
        input_image = np.resize(input_image, (img_width, img_height, 3))
        input_image = np.expand_dims(input_image, axis=0)
        final_image = np.vstack([input_image])

        # load the graph object, and models that are stored globally,
        # and use that graph for getting the model in each thread
        # global graph, model_BV
        with graph.as_default():
          # initialize parameters required for Tensor processing
          # get_session().run(tf.global_variables_initializer())
          model_BV = load_model("static/models/model_xray2.h5")
          # classify whether the image class is normal or pneumonia
          preds = model_BV.predict(final_image)
          print(preds)
          predClass = int(np.argmax(preds))
          print(predClass)
          
          # data["confidence"] = '%.2f' % (preds[predClass]*100)
          data["msg"] = "Prediction done"
          
          # If predicted class is NORMAL
          if predClass == 0:
            data["class"] = "Bacterial"
            data["success"] = True
            data["prediction"] = 0
            app.logger.debug(data); # log the data variable
            return jsonify(data), 200 # send data back to view

          # If predicted class is PNEUMONIA
          elif predClass == 1:
            data["class"] = "Viral"
            data["success"] = True
            data["prediction"] = 1
            app.logger.debug(data); # log the data variable
            return jsonify(data), 200 # send data back to view

      # If Exception or Error in server
      except Exception as e:
        data["error"] = "Some Server Error Occurred in /predict/BV"
        app.logger.error(str(e))
        return jsonify(data), 500

  else:
    app.logger.debug("Non POST request at /predict/BV")
    data["msg"] = "Not a POST request"
    data["error"] = "Forbidden"
    return jsonify(data), 403

# run the Flask application
if __name__ == "__main__":

  app.logger.debug("Loading the Pneumonia Keras models and starting Flask server, please wait...")

  app.run(
    host='0.0.0.0', 
    port=9001, 
    debug=True)

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
