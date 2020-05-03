import numpy as np
#from PIL import Image

#strength = 3
#zoom = 1.3

def flattenImage(sourceImage, destImage, strength, zoom):
	imageWidth = sourceImage.shape[1]
	imageHeight = sourceImage.shape[0]
	halfWidth = imageWidth/2
	halfHeight = imageHeight/2
	correctionRadius = np.sqrt(np.square(imageWidth) + np.square(imageHeight))/strength

	'''
	sourceRow = np.tile(np.arange(sourceImage.shape[0]), (sourceImage.shape[1],1)).T
	sourceCol = np.tile(np.arange(sourceImage.shape[1]), (sourceImage.shape[0],1))

	destRow = np.tile(np.arange(destImage.shape[0]), (destImage.shape[1],1)).T
	destCol = np.tile(np.arange(destImage.shape[1]), (destImage.shape[0],1))
	
	newY = destRow - halfHeight
	newX = destCol - halfWidth
	distance = np.sqrt(np.square(newX) + np.square(newY))
	r = distance/correctionRadius
	theta = np.where(r==0, 1, np.arctan(r)/r)
	#theta = np.zeros(r.shape)
	#theta[r==0] = 1
	#theta[r!=0] = np.arctan(r)/r
	sourceX = halfWidth + theta*newX*zoom
	sourceY = halfHeight + theta*newY*zoom

	sourceY = np.where(sourceY<0, 0, sourceY)
	sourceX = np.where(sourceX<0, 0, sourceX)
	sourceY = np.where(sourceY>=imageHeight, imageHeight-1, sourceY).astype(np.int)
	sourceX = np.where(sourceX>=imageWidth, imageWidth-1, sourceX).astype(np.int)
	

	destImage[destRow.ravel(), destCol.ravel()] = sourceImage[sourceY.ravel(), sourceX.ravel()]
	
	'''
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
			sourceX = (halfWidth + theta*newX*zoom)
			if sourceX < 0:
				sourceX = 0
			elif sourceX >= imageWidth:
				sourceX = imageWidth - 1
			sourceY = (halfHeight + theta*newY*zoom)
			if sourceY < 0:
				sourceY = 0
			elif sourceY >= imageHeight:
				sourceY = imageHeight - 1
			destImage[i,j] = sourceImage[int(sourceY), int(sourceX)]


#testImage = Image.open('./flattenImage/unflat0.jpg')
#outImage = np.array(testImage)
#flattenImage(np.array(testImage), outImage, 3, 1.3)
#Image.fromarray(outImage).save('./flattenImage/flat0.jpg')	
	


