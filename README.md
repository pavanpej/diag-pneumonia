# **DiagPneumonia** - Pneumonia Diagnosis Project

> **Authors:**
> [Pavan Rao](https://github.com/pavanpej), [Kushal Ramakanth](https://github.com/kushalramakanth), [Kunal Desai](https://github.com/kunaldesai97), [Onkar Madli](https://github.com/onkarl5)

This project takes in a chest X-ray image input, and predicts whether there is pneumonia or not.

The model is custom built using [Keras](https://keras.io/).

This project uses [Flask](http://flask.pocoo.org/) for the backend where our pre-trained model is loaded.

***Input**: A chest X-ray scanned image*

***Output**: "Normal", "Bacterial", or "Viral"*

## Requirements
Make sure you have the following packages installed on your local (or remote) environment before running 
- [TensorFlow](https://www.tensorflow.org/install)
- [Flask](flask.pocoo.org/docs/1.0/installation/)
- [Pillow](https://pillow.readthedocs.io/en/stable/installation.html)
- [NumPy](https://pypi.org/project/numpy/)

## Steps to run the project
Clone the repo or download the zip, then extract as
```
$ cd /path/to/diag-pneumonia
```
Then run,
```
$ python app.py
```
