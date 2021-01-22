# Generic Cell Classifier
This is the repository containing the glorious Generic Cell Classifier that was totally not copy and pasted together within 2 hours.
Written in Python with Tensorflow 2.

## Installation

- clone the repository
- create a virtual environment (or don't, i'm not your mom)
- install requirements via pip: `pip install -r requirements.txt`

## Training your model

- copy your training data to the `data` directory
  - a `data.csv` that contains three columns: `[file_name; class_number; microscope_type]`
  - directories containing the actual cell images (.jpg, .jpeg and .png should work)
- simply run `python trainer.py` or use custom setting:
    -  `python trainer.py --help` to see possible command line arguments 
    -   (`q`to quit from the help window ( ͡° ͜ʖ ͡°)
  
- check your training progress with `tensorboard --logdir==logs`
- find your trained model `model.h5` in the intuitively named model directory in `logs` (saved every 500 steps)

## Using your model
- abort the training when you feel confident enough about your progress
- load your model with `model = tf.keras.models.load_model("model.h5")`
- prolly read some tensorflow documentation if you haven't figured out how to move on from here lol

###### "Jesus christ i should really eat something right now!" - Jonas