import cv2 
from tensorflow.python.keras.backend import set_session
import numpy as np

import tensorflow as tf

config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.15
session = tf.compat.v1.Session(config=config)


class FacialExpressionModel(object):

	def __init__(self, model):
		self.model = tf.keras.models.load_model(model)
		self.face_detection_model = face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
		self.EMOTIONS_LIST = ["Angry", "Disgust",
                     "Fear", "Happy",
                     "Neutral", "Sad",
                     "Surprise"]


	def getFaces(self, frame):

		self.grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		self.faces = self.face_detection_model.detectMultiScale(self.grayFrame, 1.3, 5)
		
		return self.faces
	

	def predictExpression(self, cordinates):
		
		set_session(session)
		(x, y, w, h) = cordinates
		self.face = self.grayFrame[y:y+h, x:x+w]
		self.face = cv2.resize(self.face, (48, 48))
		self.face = np.expand_dims(self.face, axis=(0,-1))
		self.pred = self.model.predict(self.face)
		
		return self.EMOTIONS_LIST[np.argmax(self.pred)]
		 		