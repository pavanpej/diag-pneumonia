# app.py
from flask import Flask
from flask import render_template, request

# tensorflow dependencies
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# dimensions of our images
img_width, img_height = 224, 224

# model load
# model = load_model('model_xray.h5')
# model.compile(loss='binary_crossentropy',
#               optimizer='rmsprop',
#               metrics=['accuracy'])

# predicting images
# img = image.load_img('chest_xray/val/VIRAL/person994_virus_1672.jpeg', target_size=(img_width, img_height))
# x = image.img_to_array(img)
# x = np.expand_dims(x, axis=0)

# images = np.vstack([x])
# classes = model.predict_classes(images, batch_size=10)
# print(classes)

app = Flask(__name__)

@app.route("/")
def initial():
    print("This is a debug message")
    print("Image height: ", img_height)
    return render_template("index.html", msg="Initial")

@app.route("/predict", methods=["POST"])
def predictHandler():
    data = dict(request.data)
    print(data)
    return prediction()

def prediction():
    return render_template("index.html", msg="Prediction done")

# run the application
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=9000, debug=True)