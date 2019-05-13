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

# Load Model - NORMAL vs BACTERIAL vs VIRAL
# model_ALL = load_model("static/models/model_xray_v2.h5")

# Load Keras Model -  NORMAL vs PNEUMONIA
model_NP = load_model("static/models/model_xray1.h5")

# Load Keras Model - BACTERIAL vs VIRAL
model_BV = load_model("static/models/model_xray2.h5")

# load TensorFlow Graph for global model loading
graph = tf.get_default_graph()

# initialize the Flask app instance
app = Flask(__name__)
app.config['PATH_UPLOAD_FOLDER'] = PATH_UPLOAD_FOLDER

# configure the logger (for Heroku Dynos)
if 'DYNO' in os.environ:
  app.logger.addHandler(logging.StreamHandler(sys.stdout))
  app.logger.setLevel(logging.ERROR)

# default route - load landing page
@app.route("/")
def initial():
  app.logger.info("Loading initial route")
  data = {"msg": "Before prediction."}
  return render_template("index.html", data=data)

# route to handle the uploaded image
@app.route("/predict", methods=["POST"])
def predict():
  # initialize the data dictionary that will be
  # returned back to the view
  data = {
    "success": False,
    "class": "No Class Predicted",
    "msg": "Prediction Pending",
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

        # convert image to Keras readable format
        # which is a Numpy array
        input_image = img_to_array(input_image)
        input_image = np.resize(input_image, (img_width, img_height, 3))
        input_image = np.expand_dims(input_image, axis=0)

        # load the graph object, and models that are stored globally,
        # and use that graph for getting the model in each thread
        global graph, model_NP, model_BV
        with graph.as_default():
          # initialize parameters required for Tensor processing
          get_session().run(tf.global_variables_initializer())

          # classify whether the image class is normal or pneumonia
          preds = model_NP.predict_classes(input_image)
          
          # If predicted class is NORMAL
          if preds[0] == 0:
            data["success"] = True
            data["class"] = "Normal"
            data["msg"] = "Prediction done"
            data["prediction"] = 0
            app.logger.debug(data); # log the data variable
            return jsonify(data), 200 # send data back to view

          # If predicted class is PNEUMONIA
          elif preds[0] == 1:
            # predict class of pneumonia using second model
            preds_bv = model_BV.predict_classes(input_image)
            data["success"] = True
            data["msg"] = "Prediction done"

            # If predicted class is BACTERIAL PNEUMONIA
            if preds_bv[0] == 0:
              data["class"] = "Bacterial"
              data["prediction"] = 1
              app.logger.debug(data); # log the data variable
              return jsonify(data), 200 # send data back to view

            # If predicted class is VIRAL PNEUMONIA
            elif preds_bv[0] == 1:
              data["class"] = "Viral"
              data["prediction"] = 2
              app.logger.debug(data); # log the data variable
              return jsonify(data), 200 # send data back to view

      # If Exception or Error in server
      except Exception as e:
        data["error"] = "Some Server Error Occurred."
        app.logger.error(str(e))
        return jsonify(data), 500

  else:
    app.logger.info("Non POST request at /predict")
    data["msg"] = "Not a POST request"
    data["error"] = "Forbidden"
    return jsonify(data), 403

# run the Flask application
if __name__ == "__main__":

  app.logger.info("Loading the Pneumonia Keras models and starting Flask server, please wait...")

  app.run(
    host='0.0.0.0', 
    port=9000, 
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