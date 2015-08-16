//
//  SRAD.h
//  SRAD
//
//  Created by white on 15/5/13.
//  Copyright (c) 2015年 white. All rights reserved.
//

#ifndef SRAD_SRAD_h
#define SRAD_SRAD_h

#include <opencv/cv.h>
#include <opencv/highgui.h>

#define FILE_NAME   "/Users/white/Desktop/2048_t1.bmp"
#define RESULT_DIR  "/Users/white/Desktop/result.bmp"
#define KERNEL_FILE "/Users/white/Desktop/SRAD/SRAD/SRAD.cl"

IplImage*   ImageInit(int iterTime);

int         SRAD(IplImage *imgSrc,IplImage* imgDes,int iterTime);

int         SRAD_GPU(IplImage *imgSrc, IplImage* imgDes, int iterTime);

void        ShowImage(IplImage *imgSrc,IplImage *imgDes);

#endif
