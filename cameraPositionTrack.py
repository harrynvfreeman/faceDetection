import cv2
import detect36C as det
from scipy import misc
from PIL import Image
import numpy as np
from PIL import ImageDraw
from joblib import dump, load
import face_recognition

#dlib stuff
import sys
import os
import dlib
#predictor_model = "shape_predictor_68_face_landmarks.dat"
predictor_model = "shape_predictor_5_face_landmarks.dat"
face_pose_predictor = dlib.shape_predictor(predictor_model)
detector = dlib.get_frontal_face_detector()
facerec = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

win = dlib.image_window()

clf = load('harryEmbTrain.joblib')
#

#cv2.namedWindow("preview")
vc = cv2.VideoCapture(-1)

if vc.isOpened(): # try to get the first frame
    rval, frame = vc.read()
else:
    rval = False

#image comes in 480x640
count = 0

#arrayToSave = []

facePred = np.zeros((1,128))
tracker = None
while rval:
	image = misc.imresize(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB), [150, 200])
	#image=cv2.resize(frame, (200,150))	
	#print(np.array(image).shape)
	#image=cv2.resize(frame, (150,200))
	
	#squares = det.detect(np.array(image))
	dets = detector(image,1)
	#boxes = face_recognition.face_locations(image,model="hog")
	#dets = detector(frame)
	#
	win.clear_overlay()
	#win.set_image(frame)
	win.set_image(image)
	#lmFaces = dlib.full_object_detections()
	#
	#imShow = Image.fromarray(image) 
	#d = ImageDraw.Draw(imShow)

	#npIm = np.array(image)

	#for (w, y0, y1, x0, x1) in squares:
	#shapes = dlib.full_object_detections()
	#for idx, d in enumerate(dets):
	if len(dets) == 0:
		tracker = None
	for d in dets:
		if tracker is None:
			tracker = dlib.correlation_tracker()
			tracker.start_track(image, d)
			win.add_overlay(d)
		else:
			tracker.update(image)
			pos = tracker.get_position()
			win.add_overlay(pos)
			win.add_overlay(d, color=dlib.rgb_pixel(0,255,0))
		#d.rectangle(((x0,y0),(x1,y1)), outline = (0, 255, 0))
		#
		#detd = dlib.rectangle(left=int(x0), top = int(y0), right = int(x1), bottom = int(y1))
		#detd = d
		#shape = face_pose_predictor(image, detd)
		#shapes.append(face_pose_predictor(image, detd))
		#win.add_overlay(shape)
		#win.add_overlay(detd)
		
		#facePred[0] = facerec.compute_face_descriptor(image, shape)
		#if (clf.predict(facePred)==1):
		#win.add_overlay(detd)
		#arrayToSave.append(face_descriptor)
		#print(face_descriptor)
		#lmFaces.append(shape)
		#
		#Image.fromarray(npIm[int(y0):int(y1), int(x0):int(x1)]).save('./tahmidPhotos/tahmidPhoto' + str(count) + '.jpg')
	#encodings = face_recognition.face_encodings(image, boxes)
	#for encoding in encodings:
	#	facePred[0] = encoding
	#	clf.predict(facePred)
	#count = count + 1
	#facePreds = facerec.compute_face_descriptor(image, shapes)
	#
	#if(len(lmFaces) > 0):
		#alImages = dlib.get_face_chips(image, lmFaces, size=320)
	#
	#cv2.imshow("preview", cv2.cvtColor(np.array(misc.imresize(imShow, [480, 640])),cv2.COLOR_RGB2BGR))
	rval, frame = vc.read()
	key = cv2.waitKey(20)
	if key == 27: # exit on ESC
        		break
cv2.destroyWindow("preview")
#arrayToSave = np.array(arrayToSave)
#np.save('harryVector.npy', arrayToSave)

