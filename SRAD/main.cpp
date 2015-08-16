//
//  main.cpp
//  SRAD
//
//  Created by white on 15/5/13.
//  Copyright (c) 2015年 white. All rights reserved.
//

#include "SRAD.h"

#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <time.h>
#include <stdlib.h>
#include <stdio.h>


int main(int argc,char * argv[])
{
    int iterTime  = 100;
    IplImage *imgSrc = ImageInit(iterTime);
    IplImage *imgDes = cvCreateImage(cvSize(imgSrc->width,imgSrc->height),IPL_DEPTH_8U, 1);
    imgDes           = cvCloneImage(imgSrc);
    
    clock_t sclock,eclock;
    time_t stime,etime;
    sclock = clock();
    stime=time(NULL);
    
//    SRAD(imgSrc,imgDes,iterTime);
    SRAD_GPU( imgSrc, imgDes, iterTime);
    etime=time(NULL);
    printf("time=%ld\n",etime-stime);
    eclock = clock();
    printf("Total compute time is %fs\n",(eclock-sclock)/(double)(CLOCKS_PER_SEC));
    
    ShowImage(imgSrc,imgDes);
    cvSaveImage(RESULT_DIR,imgDes);
    printf("result image has been saved in %s. \n",RESULT_DIR);
    cvWaitKey(0);
    cvReleaseImage(&imgSrc);
    cvReleaseImage(&imgDes);
    
    return 0;
}
