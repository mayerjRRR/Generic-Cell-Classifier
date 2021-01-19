# Generic Cell Classifier -- WIP
This is the repository containing the glorious Generic Cell Classifier that was totally not copy and pasted together within 2 hours.
Written in Python with Tensorflow 2.

## Installation

- clone the repository
- create a virtual environment (or don't, i'm not your mom)
- install requirements via pip: `pip install -r requirements.txt`

## Training your model

- copy your training data to the `data` directory
- run `python classifier.py`
- check your training progress with `tensorboard --logdir==logs`
- find your trained model `model.h5` in the intuitively named model directory in `logs`

###### "Jesus christ i should really eat something right now!" - Jonas