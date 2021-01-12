# Facial-Expression-Recognition
This project is based on real time Facial expression recognition using convolutional neural network. The goal of this project is to classify each facial expression into seven universal facial expressions.

# About Project
This project usses a CNN for the real time facial expression recognition of seven most basic human expressions: **ANGER, DISGUST, FEAR, HAPPY, NEUTRAL SAD, SURPRISE**. The facial expression recognition model is trained on 28702 images belonging to 7 classes for 30 epoches with 8 layer model architecture having total 9,534,855 parameters.

# Dataset
The model is trained and tested on the data set from the [Kaggle Facial Expression Recognition](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/overview). The data consists of 48x48 pixel grayscale images of faces. The faces have been automatically registered so that the face is more or less centered and occupies about the same amount of space in each image.  

# Dependencies
- Python 3.8
- Tensorflow-gpu
- Numpy
- Opencv
- Keras

# Run the model
To run the code-  

1- Download the project by writing the following command on git bash
```
git clone https://github.com/itsGauravJoshi/Facial-Expression-Recognition.git
```
2- To run the model, open terminal and navigate to the project folder and run camera.py file.
```
python camera.py
```
