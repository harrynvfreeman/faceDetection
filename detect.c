#include <stdio.h>
#include <math.h>
#include "clfVar.h"
#include "hog.h"
#include <stdlib.h>
#include <time.h>
#include "testImage.h"

//gcc hog.c clfVar.c testImage.c  detect.c -lm -o detectInC
//chmod +x detectInC
//./detectInC

const int hN = 36;
const int wN = 36;
const float sR = 1.3;
const int windowStride = 6;
const float detectThresh = 0.7;
const float sS = 1;
const int hogOutSize = 1296;

typedef struct FaceStruct Face;
typedef struct StackStruct Stack;

Stack * detect(int * image, int hI, int hN);
float findMin(float a, float b);
void stackPush(Stack * stack, float detectionVal, int startRow, int endRow, int startCol, int endCol);
Stack * createStack();
void bilinearInterpolate(int * image, int rowSize, int colSize, float * newImage, int newRowSize, int newColSize);
float dotProduct(float * a, float * b, int length);

struct FaceStruct {
	float detectionVal;
	int startRow;
	int endRow;
	int startCol;
	int endCol;
};

struct StackStruct {
	int size;
	int capacity;
	Face * faces;
};

Stack * createStack() {
	Stack * stack = (Stack*)malloc(sizeof(Stack));
	if (stack == NULL) {
        printf("failed to allocate stack memory. \n");
    }
	Face * faces = (Face*)malloc(10*sizeof(Face));
	if (faces == NULL) {
        printf("failed to allocate faces memory. \n");
    }
	stack->faces = faces;
	stack->size = 0;
	stack->capacity = 10;
}

void stackPush(Stack * stack, float detectionVal, int startRow, int endRow, int startCol, int endCol) {
	if (stack->size >= stack->capacity) {
		Face * newFaces = (Face*)malloc(stack->capacity *2*sizeof(Face));
		if (newFaces == NULL) {
			printf("failed to allocate newFaces memory. \n");
		}
		for (int i = 0; i < stack->size; i++) {
				(newFaces + i)->detectionVal = (stack->faces + i)->detectionVal;
				(newFaces + i)->startRow = (stack->faces + i)->startRow;  
				(newFaces + i)->endRow = (stack->faces + i)->endRow; 
				(newFaces + i)->startCol = (stack->faces + i)->startCol; 
				(newFaces + i)->endCol = (stack->faces + i)->endCol; 
		}
		free(stack->faces);
		stack->faces = newFaces;
		stack->capacity = 2*stack->capacity;
	}
	
	(stack->faces + stack->size)->detectionVal = detectionVal;
	(stack->faces + stack->size)->startRow = startRow;
	(stack->faces + stack->size)->endRow = endRow;
	(stack->faces + stack->size)->startCol = startCol;
	(stack->faces + stack->size)->endCol = endCol;
	stack->size = stack->size + 1;
}

float findMin(float a, float b) {
	if (a <= b) {
		return a;
	}
	return b;
}

//image is type uint8, 0 -> 255
Stack * detect(int * image, int hI, int wI) {
	float sE = findMin(((float)wI) / ((float)wN), ((float)hI) / ((float)hN));
	int sN = (int)floor(log(sE/sS)/log(sR) + 1);
	
	//create stack for returned values
	Stack * stack = createStack();
	
	float * normalizedBlocks = (float*)malloc(hogOutSize*sizeof(float));
	if (normalizedBlocks == NULL) {
		printf("failed to allocate normalizedBlocks memory. \n");
	}
	
	float scale = sS;
	for (int i = 0; i < sN; i++) {
		
		//rescale image, do we need to do bytescale?  Doing bilinear interpolation
		int newRowSize = (int)floor(hI/scale); //DO THESE WORK CASTING???
		int newColSize = (int)floor(wI/scale);
		float * newImage = (float *)malloc(newRowSize*newColSize*3*sizeof(float));
		if (newImage == NULL) {
			printf("failed to allocate newImage memory. \n");
		}
		bilinearInterpolate(image, hI, wI, newImage, newRowSize, newColSize);
		
		int iStart = 0;
		int iEnd = hN;
		while(iEnd < newRowSize) {
			int jStart = 0;
			int jEnd = 0;
			while(jEnd < newColSize) {
				//WARNING WARNING WARNING re comment when needed
				//calcHog(newImage, hI, wI, iStart, jStart, normalizedBlocks);
				float detectionVal = dotProduct(w, normalizedBlocks,hogOutSize) + b;
				
				if (detectionVal > detectThresh) {
					stackPush(stack, detectionVal, round(iStart*hI/newRowSize), round(iEnd*hI/newRowSize), round(jStart*wI/newColSize), round(jEnd*wI/newColSize));
				}
				
				jStart = jStart + windowStride;
				jEnd = jEnd + windowStride;
			}
			
			iStart = iStart + windowStride;
			iEnd = iEnd + windowStride;
		}
		
		free(newImage);
		scale = scale * sR;
	}
	
	free(normalizedBlocks);
	return stack;
}

void bilinearInterpolate(int * image, int rowSize, int colSize, float * newImage, int newRowSize, int newColSize) {

	float sR = ((float)(rowSize))/((float)(newRowSize));
	float sC = ((float)(colSize))/((float)(newColSize));
	
	for (int k = 0; k < 3; k++) {
		for (int i = 0; i < newRowSize; i++) {
			float rf = i*sR;
			float r = floor(rf);
			float deltaR = rf - r;
			for (int j = 0; j < newColSize; j++) {
				float cf = j*sC;
				float c = floor(cf);
				float deltaC = cf - c;
			
				*(newImage + k*newRowSize*newColSize + i*newColSize + j) = (float)*(image + k*rowSize*colSize + (int)r*colSize + (int)c)*(1-deltaR)*(1-deltaC) + 
				(float)*(image + k*rowSize*colSize + ((int)r+1)*colSize + (int)c)*(deltaR)*(1-deltaC) + 
				(float)*(image + k*rowSize*colSize + (int)r*colSize + (int)c+1)*(1-deltaR)*(deltaC) + 
				(float)*(image + k*rowSize*colSize + ((int)r+1)*colSize + (int)c+1)*(deltaR)*(deltaC);
			}
		}
	}
}

float dotProduct(float * a, float * b, int length) {
	float sum = 0;
	for (int i = 0; i < length; i++) {
		sum = sum + (*(a+i))*(*(b+i));
	}
	
	return sum;
}

float main() {
	
	//clock_t start = clock();
	//Stack * stack = detect(testImage, 256, 144);
	//clock_t end = clock();
	//printf("Program Complete.  Stack size is: %d. \n", stack->size);
	//free(stack);
	//printf("Elapsed: %f seconds\n", (double)(end - start) / CLOCKS_PER_SEC);
	
	float * normalizedBlocks = (float*)malloc(1296*sizeof(float));
	
	float * gTheta = (float*)malloc(256*144*sizeof(float));
	float * subVote = (float*)malloc(256*144*sizeof(float));
		
	prepHog(testImage, 256, 144, gTheta, subVote);
	calcHog(gTheta, subVote, 144, 0, 0, normalizedBlocks);
	
	for (int i = 0; i < 1296; i++) {
		printf("%d: %f \n", i, *(normalizedBlocks + i));
	}
	
	for (int i = 0; i < 256; i++) {
		for (int j = 0; j < 144; j++) {
			//printf("(%d, %d): %f, %f \n", i, j, *(gTheta + 144*i + j), *(subVote + 144*i + j));
		}
	}
	
	free(gTheta);
	free(subVote);
	free(normalizedBlocks);
}
