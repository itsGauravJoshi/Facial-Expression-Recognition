import cv2
from model import FacialExpressionModel

video_capture = cv2.VideoCapture(0)
video_capture.set(3,1080)
video_capture.set(4, 1150)

model = FacialExpressionModel("model.h5")

while True:
	"""
	This captures a frame by accessing the person's webcam and predicts the expression of the person.
	"""
	
	_, frame = video_capture.read() #Reading frame from webcam        
	faces = model.getFaces(frame)
	
	# Going through each face in frame.
	for face in faces:
		expression = model.predictExpression(face)
		x, y, w, h = face
		cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,0), 2)
		cv2.putText(frame, expression, (x+20, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2)


	cv2.putText(frame, "Press 'q' to quit", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2)
	cv2.imshow("Video Feed", frame)
	if cv2.waitKey(1) & 0xFF == ord("q"):
		break