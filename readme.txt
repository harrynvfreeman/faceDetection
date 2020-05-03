Face Detection Project I worked on.  Optimized to run on Gigabyte Brix.
To run, call cameraRead.py.

Files:
prepData.py - deprecated file.  Was original used to preprocess images.  Experimented with 
			  different datasets and different image sizes.

prepData36.py - File used to preprocess images.  Files were split into training and testing
				sets from LFW dataset.  Files were centre-cropped and resized 
				for processing.  Output file stored as h5py.
				Also has functionality used for hard negative mining.
				
prepDataTriplet.py - File used to preprocess images when attempting to use Triplet Loss 
					 (see FaceNet algorithm).  Ended up not using as SVM classifier 
					 proved to be sufficient.  Will re-attempt when doing face recognition
					 on top of face detection.
				
script.py - A file I used to call functions while developing and testing.

createHogFeatures36.py - File with method to run HOG algorithm on images and create outputs
						 to train the SVM classifier.  Training did not have to done in 
						 real time which is why python was used.  Code was later moved to 
						 Python/Cython/C hybrid to run real time.
						 
createHogFetures36Vec.py - deprecated file.  Was used to vectorize HOG code to make it faster 
						   to run in real time, but was still too slow.  Moved to Cython
						   
trainData36.py - Trained the SVM classifier.  Was also used to run grid search to determine
				 Classifier parameters.  
				 
bSave.npy - Saved b output of SVM classifier used to run prediction.

wSave.npy - Saved w outputs of SVM classifier used to run prediction.

nmsHOG.py - My implementation of NMS algorithms to reduce overlapping detections at runtime.
			Experimented with various IOU thresholds.  

detect36.py - Code to run face detection on images.  Incorporates code from
			  createHogFeatures36.py and nmsHOG.py.  Experimented with various
			  resizing ratios and window stride lengths.  Speed was too slow for realtime 
			  performance
			  
detect36Vec.py - deprecated file.  Was an attempt to speed up detect36.py using vectorized 
				 code, but was still too slow for real time performance.

clfVar.h - header file for SVM outputs

clfVar.c - source file for SVM outputs

detect36C.pyx - Cython equivalent of detect36.py but refactored for speed.  Has the method
				used to detect images in real time.
				
detect.c - deprecated file.  Attempt to run detect36C.pyx in C for greater speed, but not
		   implemented in the end
		   
createHogFeatures36C.pyx - deprecated file.  Was used as Cython equivalent of 
						   createHogFeatures36.py, but was completely re-written into C

createHogFeatures36GxGyVoteMod.py - deprecated file.  Was used as an intermediate step 
									to convert createHogFeatures36C.pyx to c.
									
hog.h - header file for hog calculations

hog.c - source file for hog calculations.  Replaces createHogFeatures36.py and
		createHogFeatures36C.pyx

nmsHOGC.pyx - Cython equivalent of nmsHOG.py

testing.py - File I used to test speed of detecting a face.  Calls detect36C detect

tensorflowV1.py - deprecated file.  Was used to experiment with face recognition on top of
				  detection.  Significantly hurt performance.
				  
setup.py - File used to setup Cython files.  Important to run before calling detect
		   in detect36C.pyx
		   
flattenImage.py - File use to flatten fish eye distortion from camera.  Works but is 
				  not fast enough for real time.  I believe could improve performance 
				  as images were trained on non-fish eye camera.  Need to implement in C
				  for better speed.
				  
flattenImage.pyx - deprecated file.  Was used to try to improve speed from flattenImage.py
				   but was still not fast enough.  Need to implement in c
				   
flattenImage.c - Current work in progress.  Convert flattenImage.py to c to improve speed.
									
cameraRead.py - Main file used for face detection.  Uses CV2 to read image from camera, then
				calls detect in detect36C.pyx
				
cameraReadTensorflow.py - deprecated file. Was used along with tensorflowV1.py 
					   to experiment wth face recognition on top
					   detection.
					   
cameraReadOrig.py - original file to test out reading CV2 camera image.

cameraPositionTrack.py - File to experiment with dlib's facial tracking to see how my
						 HOG system compares.  Does not call detect from my HOG system,
						 just used for comparison.
						 

