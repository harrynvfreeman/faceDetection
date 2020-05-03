import numpy as np
cimport numpy as np

cdef float strength = 0.01
cdef int zoom = 1

cpdef np.ndarray flattenImage(np.ndarray sourceImage, np.ndarray destImage):
	cdef int imageWidth = sourceImage.shape[1]
	cdef int imageHeight = sourceImage.shape[0]
	cdef int halfWidth = imageWidth/2
	cdef int halfHeight = imageHeight/2
	cdef float correctionRadius = np.sqrt(np.square(imageWidth) + np.square(imageHeight))/strength
	

	cdef int i;
	cdef int j
	cdef int newY
	cdef int newX
	cdef float distance
	cdef float r
	cdef float theta
	cdef int sourceX
	cdef int sourceY
	for i in range(destImage.shape[0]):
		newY = i - halfHeight
		for j in range(destImage.shape[1]):
			newX = j - halfWidth
			distance = np.sqrt(np.square(newX) + np.square(newY))
			r = distance/correctionRadius
			if (r == 0):
				theta = 1
			else:
				theta = np.arctan(r)/r
			sourceX = (int)(halfWidth + theta*newX*zoom)
			if sourceX < 0:
				sourceX = 0
			elif sourceX >= imageWidth:
				sourceX = imageWidth - 1
			sourceY = (int)(halfHeight + theta*newY*zoom)
			if sourceY < 0:
				sourceY = 0
			elif sourceY >= imageHeight:
				sourceY = imageHeight - 1
			destImage[i,j] = sourceImage[sourceY, sourceX]
	


