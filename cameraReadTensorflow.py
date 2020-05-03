import cv2
import detect36C as det
from scipy import misc
from PIL import Image
import numpy as np
from PIL import ImageDraw

import tensorflowV1 as tens
import tensorflow as tf
from tensorflow.python.framework import ops

cv2.namedWindow("preview")
vc = cv2.VideoCapture(-1)

if vc.isOpened(): # try to get the first frame
    rval, frame = vc.read()
else:
    rval = False

#image comes in 480x640
count = 0

config = tf.ConfigProto()
ops.reset_default_graph()
X, _ = tens.create_placeholders(36, 36, 3, 2)
parameters = tens.initialize_parameters()
D2 = tens.forward_propagation(X, parameters, 0)
out = tf.argmax(input=D2, axis=1)
saver = tf.train.Saver()

with tf.Session(config = config) as sess:        
	saver.restore(sess, "./isHarryFaceNonOverfitDropOut.ckpt")

	while rval:
		image = misc.imresize(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB), [150, 200])

		squares = det.detect(np.array(image))

		imShow = Image.fromarray(image) 
		d = ImageDraw.Draw(imShow)

		npIm = np.array(image)
		plzWork = np.zeros((1,36,36,3))
		for (w, y0, y1, x0, x1) in squares:
			plzWork[0] = misc.imresize(npIm[int(y0):int(y1), int(x0):int(x1)], [36, 36])
			HMean = np.mean(plzWork, axis=(1,2,3))
			HStd = np.std(plzWork, axis = (1,2,3))
			HNew = (plzWork-HMean[:, None, None, None])/HStd[:, None, None, None]
			testerest = out.eval(feed_dict={X:HNew})
			if testerest[0] == 0:
				d.rectangle(((x0,y0),(x1,y1)), outline = (255, 0, 0))
			else:
				d.rectangle(((x0,y0),(x1,y1)), outline = (0, 255, 0))
			
			#d.rectangle(((x0,y0),(x1,y1)), outline = (0, 255, 0))
			#Image.fromarray(npIm[int(y0):int(y1), int(x0):int(x1)]).save('./tahmidPhotos/tahmidPhoto' + str(count) + '.jpg')
			#count = count + 1

	
		cv2.imshow("preview", cv2.cvtColor(np.array(misc.imresize(imShow, [480, 640])),cv2.COLOR_RGB2BGR))
		rval, frame = vc.read()
		key = cv2.waitKey(20)
		if key == 27: # exit on ESC
        		break
	cv2.destroyWindow("preview")
