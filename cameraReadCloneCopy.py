import cv2
import detect36C as det
from scipy import misc
from PIL import Image
import numpy as np
from PIL import ImageDraw
from joblib import dump, load
import face_recognition
import flattenImage
import numpy as np


cv2.namedWindow("preview")
vc = cv2.VideoCapture(-1)

if vc.isOpened(): # try to get the first frame
    rval, frame = vc.read()
else:
    rval = False

#image comes in 480x640
while rval:
	#flatIm = np.array(frame)
	#flattenImage.flattenImage(frame, flatIm)
	#image = misc.imresize(cv2.cvtColor(flatIm,cv2.COLOR_BGR2RGB), [200, 200])
	image = misc.imresize(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB), [200, 200])
	
	squares = det.detect(np.array(image))

	imShow = Image.fromarray(image) 
	d = ImageDraw.Draw(imShow)

	for (w, y0, y1, x0, x1) in squares:
		d.rectangle(((x0,y0),(x1,y1)), outline = (0, 255, 0))

	cv2.imshow("preview", cv2.cvtColor(np.array(misc.imresize(imShow, [240, 320])),cv2.COLOR_RGB2BGR))

	rval, frame = vc.read()
	key = cv2.waitKey(20)
	if key == 27: # exit on ESC
        		break
cv2.destroyWindow("preview")