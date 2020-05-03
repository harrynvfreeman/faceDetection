#include <stdio.h>
#include <math.h>
#include <time.h>
#include <stdlib.h>

#define M_PI 3.14159265358979323846

const int binSize = 9;
const int imageSize = 36;
const int cellSize = 6;
const int numCell = 6; //imageSize/cellSize
const int blockSize = 3;
const int pixelStride = 6;
const int cellStride  = 1; //pixelStride/cellSize
const int numBlocks = 16; //calculated
const int gradientMask[3] = {-1, 0, 1};
const float maxNormFactor = 0.2;
const float epsilon = 0.001;

/**
const int binSize = 9;
const int imageSize = 72;
const int cellSize = 12;
const int numCell = 6; //imageSize/cellSize
const int blockSize = 6;
const int pixelStride = 12;
const int cellStride  = 1; //pixelStride/cellSize
const int numBlocks = 49; //calculated
const int gradientMask[3] = {-1, 0, 1};
const float maxNormFactor = 0.2;
const float epsilon = 0.001;
**/

void calcHog(float * gTheta, float * subVote, int colSize, int rowStart, int colStart, float * normalizedBlocks);
void castVotes(float * histogram, float * gTheta, float* subVote, int colSize);
void normalize(float * normalizedBlocks, float * histogram);
void createSubBlock(float * block, float * histogram, int histRowStart, int histRowEnd, int histColStart, int histColEnd);

void prepHog(float * x, int rowSize, int colSize, float * gTheta, float * subVote);
void convolveX(float * x, int rowSize, int colSize, float * gx);
void convolveY(float * x, int rowSize, int colSize, float * gy);
void calcVote(float * vote, float * gx, float * gy, int rowSize, int colSize);
void calcSubVoteAndGTheta(float * subVote, float * gTheta, float * vote, float * gx, float * gy, int rowSize, int colSize);

void prepHog(float * x, int rowSize, int colSize, float * gTheta, float * subVote) {
    
    float * gx = (float*)malloc(3*rowSize*colSize*sizeof(float));
    if (gx == NULL) {
        printf("failed to allocate gx memory. \n");
    }
    
    float * gy = (float*)malloc(3*rowSize*colSize*sizeof(float));
    if (gy == NULL) {
        printf("failed to allocate gy memory. \n");
    }
    
    float * vote = (float*)malloc(3*rowSize*colSize*sizeof(float));
    if (vote == NULL) {
        printf("failed to allocate vote memory. \n");
    }
    
    convolveX(x, rowSize, colSize, gx);
    convolveY(x, rowSize, colSize, gy);
    calcVote(vote, gx, gy, rowSize, colSize);

    calcSubVoteAndGTheta(subVote, gTheta, vote, gx, gy, rowSize, colSize);
    
    free(gx);
    free(gy);
    free(vote);
}

//input is a 3xrowSizexcolSize vector all to be convolved for dim colour
void convolveX(float * x, int rowSize, int colSize, float * gx) {
    for (int dim = 0; dim < 3; dim++) {
        int startDim = rowSize*colSize*dim;
        for (int i = 0; i < rowSize; i++) {
            float * gxTemp = gx + startDim + colSize*i;
            float * xTemp = x + startDim + colSize*i;
            
            *(gxTemp) = -*(xTemp + 1) + (*xTemp);
            *(gxTemp + colSize-1) = *(xTemp + colSize - 2) - *(xTemp + colSize - 1);
            for (int n = 2; n < colSize; n++) {
                float sum = 0;
                for (int m = 0; m < 3; m++) {
                    sum = sum + (*(gradientMask+m))*(*(xTemp + n - m));
                }
                *(gxTemp + n - 1) = sum;
            }
        }
    }
}

//input is a 3xrowSizexcolSize vector all to be convolved for dim colour
void convolveY(float * x, int rowSize, int colSize, float * gy) {
    for (int dim = 0; dim < 3; dim++) {
        int startDim = rowSize*colSize*dim;
        for (int j = 0; j < colSize; j++) {
            float * gyTemp = gy + startDim + j;
            float * xTemp = x + startDim + j;
            
            *(gyTemp) = -*(xTemp + colSize) + (*xTemp);
            *(gyTemp + (rowSize-1)*colSize) = *(xTemp + (rowSize-2)*colSize) - *(xTemp + (rowSize-1)*colSize);
            
            for (int n = 2; n < rowSize; n++) {
                float sum = 0;
                for (int m = 0; m < 3; m++) {
                    sum = sum + (*(gradientMask+m))*(*(xTemp + (n-m)*colSize));
                }
                *(gyTemp + (n-1)*colSize) = sum;
            }
        }
    } 
}

void calcVote(float * vote, float * gx, float * gy, int rowSize, int colSize) {
    for (int i = 0; i < 3*rowSize*colSize; i++) {
        *(vote + i) = sqrt((*(gx + i))*(*(gx + i)) + (*(gy + i))*(*(gy + i)));
    }
}

void calcSubVoteAndGTheta(float * subVote, float * gTheta, float * vote, float * gx, float * gy, int rowSize, int colSize) {
    float norm0 = 0;
    float norm1 = 0;
    float norm2 = 0;
    //since just comparing do not need to square root norms
    for (int i = 0; i < rowSize; i++) {
        for (int j = 0; j < colSize; j++) {
            norm0 = norm0 + (*(vote + i*colSize + j))*(*(vote + i*colSize + j));
            norm1 = norm1 + (*(vote + i*colSize + j + rowSize*colSize))*(*(vote + i*colSize + j + rowSize*colSize));
            norm2 = norm2 + (*(vote + i*colSize + j + 2*rowSize*colSize))*(*(vote + i*colSize + j + 2*rowSize*colSize));
        }
    }

    int index;
    if (norm0 >= norm1 && norm0 >= norm2) {
        index = 0;
    } else if (norm1 >= norm0 && norm1 >= norm2) {
        index = 1;
    } else {
        index = 2;
    }
    
    for (int i = 0; i < rowSize; i++) {
        for (int j = 0; j < colSize; j++) {
            *(subVote + i*colSize + j) = *(vote + index*rowSize*colSize + i*colSize + j);
        
            if (*(gy + index*rowSize*colSize + i*colSize + j) == 0) {
                *(gTheta + i*colSize + j) = 0;
            } else if (*(gx + index*rowSize*colSize + i*colSize + j) == 0) {
                *(gTheta + i*colSize + j) = 90;
            } else {
                *(gTheta + i*colSize + j) = atan((*(gy + index*rowSize*colSize + i*colSize + j))/(*(gx + index*rowSize*colSize + i*colSize + j))) * 180 / M_PI;
                if (*(gTheta + i*colSize + j) < 0) {
                    *(gTheta + i*colSize + j) = *(gTheta + i*colSize + j) + 180;
                }
            }
        }
    }
}

//First, for simplicity, get non vectorized working. x should be 3x36x36
void calcHog(float * gTheta, float * subVote, int colSize, int rowStart, int colStart, float * normalizedBlocks) {
    
    float * histogram = (float*)calloc(numCell*numCell*binSize, sizeof(float));
    if (histogram == NULL) {
        printf("failed to allocate histogram memory. \n");
    }

    castVotes(histogram, gTheta + rowStart*colSize + colStart, subVote + rowStart*colSize + colStart, colSize);

    normalize(normalizedBlocks, histogram);
    free(histogram);
    
}

void castVotes(float * histogram, float * gTheta, float* subVote, int colSize) {
    float bz = 180/binSize;
    float by = cellSize;
    float bx = cellSize;
    
    float * cz = (float*)malloc(binSize*sizeof(float));
    if (cz == NULL) {
        printf("failed to allocate cz memory. \n");
    }

    for (int i = 0; i < binSize; i++) {
        *(cz + i) = bz * (0.5 + i);
    }
    
    float * cy = (float*)malloc(numCell*sizeof(float));
    if (cy == NULL) {
        printf("failed to allocate cy memory. \n");
    }
    
    float * cx = (float*)malloc(numCell*sizeof(float));
    if (cx == NULL) {
        printf("failed to allocate cx memory. \n");
    }
    
    for (int i = 0; i < numCell; i++) {
        *(cy + i) = by * (0.5 + i) - 0.5;
        *(cx + i) = bx * (0.5 + i) - 0.5;
    }
    for (int i = 0; i < imageSize; i++) {
        for (int j = 0; j < imageSize; j++) {
            //
            int bin0 = ((int)floor((*(gTheta + i*colSize + j) - *(cz))/bz)) % binSize;
            if (bin0 < 0) {
                bin0 = bin0 + binSize;
            }
            int bin1 = (bin0 + 1) % binSize;
            //
            int y0 = ((int)floor((i - *(cy))/by)) % numCell;
            if (y0 < 0) {
                y0 = y0 + numCell;
            }
            int y1 = (y0 + 1) % numCell;
            //
            int x0 = ((int)floor((j - *(cx))/bx)) % numCell;
            if (x0 < 0) {
                x0 = x0 + numCell;
            }
            int x1 = (x0 + 1) % numCell;
            //
            float binVote1 = fmod(*(gTheta + i*colSize + j) - *(cz + bin0), bz);
            if (binVote1 < 0) {
                binVote1 = (binVote1 + bz)/bz;
            } else {
                binVote1 = binVote1/bz;
            }
            float binVote0 = 1-binVote1;
            //
            float yVote1 = fmod(i - *(cy + y0), by);
            if (yVote1 < 0) {
                yVote1 = (yVote1 + by)/by;
            } else {
                yVote1 = yVote1/by;
            }
            float yVote0 = 1-yVote1;
            //
            float xVote1 = fmod(j - *(cx + x0), bx);
            if (xVote1 < 0) {
                xVote1 = (xVote1 + bx)/bx;
            } else {
                xVote1 = xVote1/bx;
            }
            float xVote0 = 1-xVote1;
            //
            float voteVal = *(subVote + i*colSize + j);

            *(histogram + y0*numCell*binSize + x0*binSize + bin0) = *(histogram + y0*numCell*binSize + x0*binSize + bin0) + voteVal*yVote0*xVote0*binVote0;
            
            *(histogram + y0*numCell*binSize + x0*binSize + bin1) = *(histogram + y0*numCell*binSize + x0*binSize + bin1) + voteVal*yVote0*xVote0*binVote1;
            
            *(histogram + y0*numCell*binSize + x1*binSize + bin0) = *(histogram + y0*numCell*binSize + x1*binSize + bin0) + voteVal*yVote0*xVote1*binVote0;
            
            *(histogram + y0*numCell*binSize + x1*binSize + bin1) = *(histogram + y0*numCell*binSize + x1*binSize + bin1) + voteVal*yVote0*xVote1*binVote1;
            
            *(histogram + y1*numCell*binSize + x0*binSize + bin0) = *(histogram + y1*numCell*binSize + x0*binSize + bin0) + voteVal*yVote1*xVote0*binVote0;
            
            *(histogram + y1*numCell*binSize + x0*binSize + bin1) = *(histogram + y1*numCell*binSize + x0*binSize + bin1) + voteVal*yVote1*xVote0*binVote1;
            
            *(histogram + y1*numCell*binSize + x1*binSize + bin0) = *(histogram + y1*numCell*binSize + x1*binSize + bin0) + voteVal*yVote1*xVote1*binVote0;
            
            *(histogram + y1*numCell*binSize + x1*binSize + bin1) = *(histogram + y1*numCell*binSize + x1*binSize + bin1) + voteVal*yVote1*xVote1*binVote1;
        }
    }
    
    free(cz);
    free(cy);
    free(cx);
}

void normalize(float * normalizedBlocks, float * histogram) {
    
    int blockIndex = 0;
    int rowStardIndex = 0;
    int rowEndIndex = blockSize;
    while (rowEndIndex <= numCell) {
        
        int colStartIndex = 0;
        int colEndIndex = blockSize;
        while (colEndIndex <= numCell) {
            createSubBlock(normalizedBlocks + blockIndex*binSize*blockSize*blockSize, histogram, rowStardIndex, rowEndIndex, colStartIndex, colEndIndex);
            blockIndex = blockIndex + 1;

            colStartIndex = colStartIndex + cellStride;
            colEndIndex = colEndIndex + cellStride;
        }
        
        
        rowStardIndex = rowStardIndex + cellStride;
        rowEndIndex = rowEndIndex + cellStride;
    }
    
    float * normVals = (float*)malloc(numBlocks*sizeof(float));
    if (normVals == NULL) {
        printf("failed to allocate normVals memory. \n");
    }
    
    for (int i = 0; i < numBlocks; i++) {
        float sum = 0;
        for (int j = 0; j < binSize*blockSize*blockSize; j++) {
            sum = sum + (*(normalizedBlocks + i*binSize*blockSize*blockSize + j))*(*(normalizedBlocks + i*binSize*blockSize*blockSize + j));
        }
        *(normVals + i) = sqrt(sum + epsilon*epsilon);
    }
    
    for (int i = 0; i < numBlocks; i++) {
        float sum = 0;
        for (int j = 0; j < binSize*blockSize*blockSize; j++) {
            *(normalizedBlocks + i*binSize*blockSize*blockSize + j) = (*(normalizedBlocks + i*binSize*blockSize*blockSize + j)) / (*(normVals + i));
            if (*(normalizedBlocks + i*binSize*blockSize*blockSize + j) > maxNormFactor) {
                *(normalizedBlocks + i*binSize*blockSize*blockSize + j) = maxNormFactor;
            }
            sum = sum + (*(normalizedBlocks + i*binSize*blockSize*blockSize + j))*(*(normalizedBlocks + i*binSize*blockSize*blockSize + j));
        }
        *(normVals + i) = sqrt(sum + epsilon*epsilon);
    }
    
    for (int i = 0; i < numBlocks; i++) {
        for (int j = 0; j < binSize*blockSize*blockSize; j++) {
            *(normalizedBlocks + i*binSize*blockSize*blockSize + j) = (*(normalizedBlocks + i*binSize*blockSize*blockSize + j)) / (*(normVals + i));
        }
    }
    
    free(normVals);
}

void createSubBlock(float * block, float * histogram, int histRowStart, int histRowEnd, int histColStart, int histColEnd) {
    int blockCount = 0;
    for (int i = histRowStart; i < histRowEnd; i++) {
        for (int j = histColStart; j < histColEnd; j++) {
            for (int k = 0; k < binSize; k++) {
                *(block + blockCount) = *(histogram + i*numCell*binSize + j*binSize + k);
                blockCount = blockCount + 1;
            }
        }
    }
}

/**
//input is a 36x1 vector to be convolved with gradient mask
void convolve(float * x, int rowSize, int colSize, float * result, int isX) {
    int resultScaleFact;
    if (isX == 1) {
        resultScaleFact = 1;
    } else {
        resultScaleFact = colSize;
    }
    *(result) = -*(x + resultScaleFact) + (*x);
    *(result + (rowSize - 1)*resultScaleFact) = *(x+(rowSize - 2)*resultScaleFact) - *(x+(rowSize - 1)*resultScaleFact);
    for (int n = 2; n < imageSize; n++) {
        float sum = 0;
        for (int m = 0; m < 3; m++) {
            sum = sum + (*(gradientMask+m))*(*(x+(n-m)*xScaleFact));
        }
        *(result + (n-1)*resultScaleFact) = sum;
    } 
}
*/

/**
void calcSubVoteAndGTheta(float * subVote, float * gTheta, float * vote, float * gx, float * gy, int rowSize, int colSize, int rowStart, int colStart) {
    float norm0 = 0;
    float norm1 = 0;
    float norm2 = 0;
    
    //since just comparing do not need to square root norms
    for (int i = rowStart*colSize; i < (rowStart+imageSize)*colSize; i = i + colSize) {
        for (int j = colStart; j < colStart + imageSize; j++) {
            norm0 = norm1 + (*(vote + i + j))*(*(vote + i + j));
            norm1 = norm1 + (*(vote + i + j + rowSize*colSize))*(*(vote + i + j + rowSize*colSize));
            norm2 = norm2 + (*(vote + i + j + 2*rowSize*colSize))*(*(vote + i + j + 2*rowSize*colSize));
        }
    }
    
    int index;
    if (norm0 >= norm1 && norm0 >= norm2) {
        index = 0;
    } else if (norm1 >= norm0 && norm1 >= norm2) {
        index = 1;
    } else {
        index = 2;
    }
    
    for (int i = 0; i < imageSize; i++) {
        for (int j = 0; j < imageSize; j++) {
            *(subVote + i) = *(vote + index*rowSize*colSize + (i+rowStart)*colSize + j+colStart);
        
            if (*(gy + index*rowSize*colSize + (i+rowStart)*colSize + j+colStart) == 0) {
                *(gTheta + i) = 0;
            } else if (*(gx + index*rowSize*colSize + (i+rowStart)*colSize + j+colStart) == 0) {
                *(gTheta + i) = 90;
            } else {
                *(gTheta + i) = atan((*(gy + index*rowSize*colSize + (i+rowStart)*colSize + j+colStart))/(*(gx + index*rowSize*colSize + (i+rowStart)*colSize + j+colStart))) * 180 / M_PI;
                if (*(gTheta + i) < 0) {
                    *(gTheta + i) = *(gTheta + i) + 180;
                }
            }
        }
    }
}
*/
