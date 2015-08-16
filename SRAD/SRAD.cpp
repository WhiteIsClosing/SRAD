//
//  SRAD.cpp
//  SRAD
//
//  Created by white on 15/5/13.
//  Copyright (c) 2015年 white. All rights reserved.
//

#include "SRAD.h"

#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <OpenCL/cl.h>
#include <stdio.h>
#include <stdlib.h>

#define CL_DEBUG
#define SQUARE(t) ( (t) * (t) )

IplImage* ImageInit(int iterTime)
{
    char fileName[100]=FILE_NAME;
    IplImage *img=cvLoadImage(fileName,0);
    if(img==NULL)
    {
        fprintf(stderr,"Error when reading file '%s', does it exist?\n",fileName);
        exit(-1);
    }
    return img;
}

int SRAD(IplImage *imgSrc,IplImage* imgDes,int iterTime)
{
    if( imgSrc->nChannels != 1)
    {
        fprintf(stderr,"imgSrc is not monochromatic.\n");
        return -1;
    }
    IplImage* imgTmp=cvCreateImage( cvSize(imgSrc->width, imgSrc->height), IPL_DEPTH_8U, 1);
    imgTmp          =cvCloneImage(imgSrc);
    
    uchar *src  = (uchar *)(imgTmp->imageData);
    uchar *des  = (uchar *)(imgDes->imageData);
    
    int iWidth  = imgSrc->width;
    int iHeight = imgSrc->height;
    int iStep   = imgSrc->widthStep;
    
    double eps  = 0.0001;
    double q0   = 0.0;
    double mean = 0.0;
    for(int h=0; h < iHeight; h++)
    {
        for(int w=0; w < iWidth; w++)
        {
            mean += src[h*iStep+w];
        }
    }
    mean /= iWidth*iHeight;
    
    for( int h=0; h < iHeight; h++)
    {
        for( int w=0; w < iWidth; w++)
        {
            q0 += ( src[h*iStep+w] - mean ) * ( src[h*iStep+w] - mean );
        }
    }
    q0 = sqrt( q0/(iWidth*iHeight) )/mean;
    
    double *cq=new double[iWidth*iHeight];
    memset( cq, 0, sizeof(double)*iWidth*iHeight );
    
    for( int it=1; it<=iterTime; it++ )
    {
        printf("Running %d times iteration.\n",it);
        for( int h = 2; h < iHeight-2; h++ )
        {
            for( int w = 2; w < iWidth-2; w++ )
            {
                
                double a  = ( -0.2 ) * ( it - 1 );
                double qt = q0 * exp(a);
                
                double GradientX = ( src[w+h*iStep+1] - src[w+h*iStep-1] ) / 2;
                double GradientY = ( src[w+(h+1)*iStep] - src[w+(h-1)*iStep] ) / 2;
                
                double di1 = sqrt( GradientX * GradientX + GradientY * GradientY );
                double di2 = double(src[w+h*iStep+1] +
                                    src[w+h*iStep-1] +
                                    src[w+(h+1)*iStep+1] +
                                    src[w+(h-1)*iStep-1])/4 - src[w+h*iStep];
                
                double t1 = 0.5   * SQUARE(  di1/(src[w+h*iStep] + eps)  );
                double t2 = 0.625 * SQUARE(  di2/(src[w+h*iStep] + eps)  );
                double t3 = SQUARE( 1 + ( 0.25 *  di2/(src[w+h*iStep]+eps) ) );
                double t  = sqrt( fabs( t1 - t2 ) / fabs( t3 + eps ) );
                double dd = ( t*t - qt*qt ) / (qt*qt*(1+qt*qt)+eps);
                cq[w+h*iWidth] = 1 / (1 + dd);
            }
        }
        
        for( int h = 2; h < iHeight-2; h++ )
        {
            for( int w = 2; w < iWidth-2; w++ )
            {
                double GradientX_ =
                (cq[w+h*iWidth+1]*(src[w+h*iStep+2]-src[w+h*iStep])-cq[w+h*iWidth-1]*(src[w+h*iStep]-src[w+h*iStep-2]))/4;
                double GradientY_ =
                (cq[w+(h+1)*iWidth]*(src[w+(h+2)*iStep]-src[w+h*iStep])-cq[w+(h-1)*iWidth]*( src[w+h*iStep]-src[w+(h-2)*iStep]))/4;
                
                des[w+h*iStep] =
                (uchar)(src[w+h*iStep] + 0.035 * ( GradientX_ + GradientY_ ) );
            }
        }
        
        uchar* tmp;
        tmp = src;
        src = des;
        des = tmp;
        
    }
    if( iterTime % 2 == 0)
    {
       imgDes= cvCloneImage(imgTmp);
    }
    delete[] cq;
    cvReleaseImage(&imgTmp);
    return 0;
}

void ShowImage(IplImage *imgSrc,IplImage *imgDes)
{
    cvNamedWindow("Raw Image");
    cvNamedWindow("Result Image");
    cvShowImage("Raw Image",imgSrc);
    cvShowImage("Result Image",imgDes);
}

void openclRetTackle(cl_int retValue, const char* processInfo){
    const char* errInfo;
    switch(retValue){
        case  0  :errInfo="CL_SUCCESS";break;
        case -1  :errInfo="CL_DEVICE_NOT_FOUND";break;
        case -2  :errInfo="CL_DEVICE_NOT_AVAILABLE";break;
        case -3  :errInfo="CL_COMPILER_NOT_AVAILABLE";break;
        case -4  :errInfo="CL_MEM_OBJECT_ALLOCATION_FAILURE";break;
        case -5  :errInfo="CL_OUT_OF_RESOURCES";break;
        case -6  :errInfo="CL_OUT_OF_HOST_MEMORY";break;
        case -7  :errInfo="CL_PROFILING_INFO_NOT_AVAILABLE";break;
        case -8  :errInfo="CL_MEM_COPY_OVERLAP";break;
        case -9  :errInfo="CL_IMAGE_FORMAT_MISMATCH";break;
        case -10 :errInfo="CL_IMAGE_FORMAT_NOT_SUPPORTED";break;
        case -11 :errInfo="CL_BUILD_PROGRAM_FAILURE";break;
        case -12 :errInfo="CL_MAP_FAILURE";break;
        case -13 :errInfo="CL_MISALIGNED_SUB_BUFFER_OFFSET";break;
        case -14 :errInfo="CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";break;
        case -30 :errInfo="CL_INVALID_VALUE";break;
        case -31 :errInfo="CL_INVALID_DEVICE_TYPE";break;
        case -32 :errInfo="CL_INVALID_PLATFORM";break;
        case -33 :errInfo="CL_INVALID_DEVICE";break;
        case -34 :errInfo="CL_INVALID_CONTEXT";break;
        case -35 :errInfo="CL_INVALID_QUEUE_PROPERTIES";break;
        case -36 :errInfo="CL_INVALID_COMMAND_QUEUE";break;
        case -37 :errInfo="CL_INVALID_HOST_PTR";break;
        case -38 :errInfo="CL_INVALID_MEM_OBJECT";break;
        case -39 :errInfo="CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";break;
        case -40 :errInfo="CL_INVALID_IMAGE_SIZE";break;
        case -41 :errInfo="CL_INVALID_SAMPLER";break;
        case -42 :errInfo="CL_INVALID_BINARY";break;
        case -43 :errInfo="CL_INVALID_BUILD_OPTIONS";break;
        case -44 :errInfo="CL_INVALID_PROGRAM";break;
        case -45 :errInfo="CL_INVALID_PROGRAM_EXECUTABLE";break;
        case -46 :errInfo="CL_INVALID_KERNEL_NAME";break;
        case -47 :errInfo="CL_INVALID_KERNEL_DEFINITION";break;
        case -48 :errInfo="CL_INVALID_KERNEL";break;
        case -49 :errInfo="CL_INVALID_ARG_INDEX";break;
        case -50 :errInfo="CL_INVALID_ARG_VALUE";break;
        case -51 :errInfo="CL_INVALID_ARG_SIZE";break;
        case -52 :errInfo="CL_INVALID_KERNEL_ARGS";break;
        case -53 :errInfo="CL_INVALID_WORK_DIMENSION";break;
        case -54 :errInfo="CL_INVALID_WORK_GROUP_SIZE";break;
        case -55 :errInfo="CL_INVALID_WORK_ITEM_SIZE";break;
        case -56 :errInfo="CL_INVALID_GLOBAL_OFFSET";break;
        case -57 :errInfo="CL_INVALID_EVENT_WAIT_LIST";break;
        case -58 :errInfo="CL_INVALID_EVENT";break;
        case -59 :errInfo="CL_INVALID_OPERATION";break;
        case -60 :errInfo="CL_INVALID_GL_OBJECT";break;
        case -61 :errInfo="CL_INVALID_BUFFER_SIZE";break;
        case -62 :errInfo="CL_INVALID_MIP_LEVEL";break;
        case -63 :errInfo="CL_INVALID_GLOBAL_WORK_SIZE";break;
        case -64 :errInfo="CL_INVALID_PROPERTY";break;
        default  :errInfo="CL_SUCCESS";break;
    }
    if(retValue!=CL_SUCCESS)
    {
        #if (defined CL_DEBUG) || (defined CL_VERBOSE)
        fprintf(stderr,"%s Error! %s \n",processInfo,errInfo);
        #endif
        system("pause");
        exit(-1);
    }
    else
    {
        #ifdef CL_VERBOSE
        printf("%s Success!\n",processInfo);
        #endif
    }
}

cl_platform_id cpPlatform;
cl_device_id cdDevice;
cl_context cxGPUContext;
cl_command_queue cqCommandQueue;
cl_program cpProgram;

void printCLDeviceInfo()
{
    cl_uint uintRet;
    cl_ulong ulongRet;
    clGetDeviceInfo(cdDevice,CL_DEVICE_MAX_COMPUTE_UNITS,sizeof(uintRet),&uintRet,NULL);
    printf("CL_DEVICE_MAX_COMPUTE_UNITS %d\n",uintRet);
    clGetDeviceInfo(cdDevice,CL_DEVICE_MAX_MEM_ALLOC_SIZE,sizeof(ulongRet),&ulongRet,NULL);
    printf("CL_DEVICE_MAX_MEM_ALLOC_SIZE %d\n",(int)ulongRet);
}

int openclInit()
{
    cl_int ret;
    openclRetTackle( clGetPlatformIDs(1, &cpPlatform, NULL), "clGetPlatFormIDs");
    openclRetTackle( clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_CPU,1,&cdDevice,NULL), "clGetDeviceIDs");
    cxGPUContext = clCreateContext(0, 1, &cdDevice, NULL, NULL, &ret);
    #if (defined CL_DEBUG) || (defined CL_VERBOSE)
    printCLDeviceInfo();
    #endif
    openclRetTackle( ret , "clCreateContext" );
    cqCommandQueue = clCreateCommandQueue(cxGPUContext, cdDevice, 0, &ret);
    openclRetTackle( ret , "clCreateCommandQueue");
    return CL_SUCCESS;
}

int openclCreateKernelFromFile(cl_program* cpProgram,cl_kernel* clKernel,const char* clFileName,const char* kernelName)
{
    cl_int  ret;
    cl_uint count=1;
    size_t  sourceLength=0;
    char*   sourceString=NULL;
    FILE*   fp;
    fp  = fopen(clFileName,"rb");
    if(fp==NULL)
    {
        fprintf(stderr,"Read source file %s failed, does it exist?\n",clFileName);
        exit(-1);
    }
    fseek(fp,0,SEEK_END);
    sourceLength=ftell(fp);
    fseek(fp,0,SEEK_SET);
    sourceString=(char*)malloc((sourceLength+1)*sizeof(char));//no non-ascii characters in .cl file please
    memset(sourceString,0,(sourceLength+1)*sizeof(char));
    if(fread(sourceString,sourceLength,1,fp)!=1){
        fprintf(stderr,"Cannot read source file %s.\n",clFileName);
        exit(-1);
    }
    (*cpProgram)=clCreateProgramWithSource(cxGPUContext,count,(const char**)(&sourceString),&sourceLength,&ret);
    openclRetTackle(ret,"clCreateProgramWithSource");
    ret=clBuildProgram(*cpProgram,1,&cdDevice, NULL,NULL,NULL);
    openclRetTackle(ret,"clBuildProgram");
    (*clKernel)=clCreateKernel(*cpProgram,kernelName,&ret);
    
    char *infoString=(char*)malloc( (strlen(kernelName)+40) * sizeof(char) );
    memset(infoString,0,sizeof((strlen(kernelName)+40) * sizeof(char)));
    strcat(infoString,"clCreateKernel ");
    strcat(infoString,kernelName);
    openclRetTackle(ret,infoString);
    
    free(infoString);
    free(sourceString);
    fclose(fp);
    
    return 0;
}

struct param{
    unsigned int    iWidth;
    unsigned int    iHeight;
    unsigned int    it;
    float           mean;
    float           q0;
    float           empt1;
    float           empt2;
    float           empt3;
};

cl_mem cmImgSrc;
cl_mem cmImgDes;
cl_mem cmCqM;
cl_mem cmPar;

float imgMean(IplImage *imgSrc)
{
    int iWidth  = imgSrc->width;
    int iHeight = imgSrc->height;
    int iStep   = imgSrc->widthStep;
    uchar* src  = (uchar*)imgSrc->imageData;
    
    float mean=0;
    for(int i=0;i<iHeight;i++)
    {
        for(int j=0; j<iWidth; j++)
        {
            mean+=src[i*iStep+j];
        }
    }
    mean /= (iWidth*iHeight);
    return mean;
}
float imgQ0(IplImage *imgSrc, float mean)
{
    int     iWidth  = imgSrc->width;
    int     iHeight = imgSrc->height;
    int     iStep   = imgSrc->widthStep;
    uchar*  src     = (uchar*)imgSrc->imageData;
    float   q0      = 0;
    for( int h=0; h < iHeight; h++)
    {
        for( int w=0; w < iWidth; w++)
        {
            q0 += ( src[h*iStep+w] - mean ) * ( src[h*iStep+w] - mean );
        }
    }
    q0 = sqrt( q0/(iWidth*iHeight) )/mean;
    return q0;
}

void prepareBuffer(IplImage* imgSrc,int iterTime)
{
    struct param par;
    par.iWidth  = imgSrc->width;
    par.iHeight = imgSrc->height;
    par.it      = iterTime;
    par.mean    = imgMean(imgSrc);
    par.q0      = imgQ0(imgSrc,par.mean);
    cl_int ret;
    cmPar = clCreateBuffer(cxGPUContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,sizeof(struct param),&par,&ret);
    openclRetTackle(ret,"clCreateBuffer const");
    
    int iStep       = imgSrc->widthStep;
    float* initMem  = (float*)malloc(sizeof(float) * par.iWidth * par.iHeight);
    memset(initMem,0,sizeof(float)* par.iWidth * par.iHeight);
    for(int i=0;i<imgSrc->height;i++)
    {
        for(int j=0;j<imgSrc->widthStep;j++)
        {
            initMem[i*par.iWidth+j]=(unsigned char)imgSrc->imageData[i*iStep+j];
        }
    }
    cmImgSrc = clCreateBuffer(cxGPUContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,sizeof(float) * par.iWidth * par.iHeight,initMem,&ret);
    openclRetTackle(ret,"clCreateBuffer cmImgSrc");
    cmImgDes = clCreateBuffer(cxGPUContext, CL_MEM_READ_WRITE, sizeof(float) * par.iWidth * par.iHeight,0,&ret);
    openclRetTackle(ret,"clCreateBuffer cmImgDes");
    cmCqM = clCreateBuffer(cxGPUContext, CL_MEM_READ_WRITE, sizeof(float) * par.iWidth * par.iHeight, 0, &ret);
    openclRetTackle(ret,"clCreateBuffer cmCqM");
}

int SRAD_GPU(IplImage *imgSrc, IplImage* imgDes, int iterTime){
    cl_int ret;
    size_t iSize    = (imgSrc->width * imgSrc->height);
    int iWidth      = imgSrc->width;
    int iHeight     = imgSrc->height;
    int iStep       = imgSrc->widthStep;
    printf("Image size is %dx%d=%d\n", imgSrc->width,imgSrc->height,iSize);
    /*
     if(iSize>254*254*254){
     fprintf(stderr,"image size is too large\n");
     system("pause");
     exit(-1);
     }
     */
    openclInit();

    prepareBuffer(imgSrc,iterTime);

    cl_program  program1,program2;
    cl_kernel   kernelCQ,kernelSRAD;
    openclCreateKernelFromFile(&program1,&kernelCQ,KERNEL_FILE,"CQ");
    openclCreateKernelFromFile(&program2,&kernelSRAD,KERNEL_FILE,"SRAD");
   
    size_t global_work_size[]={2,2,1};
    if(iSize<255)
    {
        
    global_work_size[0]=iSize;
    }
    else
    {
        global_work_size[0]=256;
        if(iSize<256*256)
        {
            global_work_size[1]=(size_t)ceil(double(iSize)/256);
        }
        else
        {
            global_work_size[1]=256;
            global_work_size[2]=size_t(ceil(double(iSize)/256/256));
        }
    }
    size_t local_work_size[]={1,1,1};
    printf("global work size %zu %zu %zu\n",global_work_size[0],global_work_size[1],global_work_size[2]);
    printf("local work size %zu %zu %zu\n",local_work_size[0],local_work_size[1],local_work_size[2]);

    /*
     global_work_size[0]=254;
     global_work_size[1]=254;
     global_work_size[2]=1;
     local_work_size[0]=254;
     local_work_size[1]=1;
     local_work_size[2]=1;
     */
    struct param parSwap;
    for(int i=1;i<=iterTime;i++){
        printf("Run %d times iteration\n",i);
        
        openclRetTackle(clSetKernelArg(kernelCQ,0,sizeof(cl_mem),&cmPar),"clSetKernelArg 0");
        openclRetTackle(clSetKernelArg(kernelCQ,1,sizeof(cl_mem),&cmImgSrc),"clSetKernelArg 1");
        openclRetTackle(clSetKernelArg(kernelCQ,2,sizeof(cl_mem),&cmCqM),"clSetKernelArg 2");
        openclRetTackle(clEnqueueNDRangeKernel(cqCommandQueue,kernelCQ,3,0,global_work_size,NULL,0,0,0),"clEnqueueNDRangeKernel cq");
        openclRetTackle(clFinish(cqCommandQueue),"clFinish");
        
        
        struct param* hostmap=(struct param*)clEnqueueMapBuffer(cqCommandQueue,cmPar,CL_TRUE,CL_MAP_WRITE,0,sizeof(struct param),0,0,0,&ret);
        hostmap->it=i;
        clEnqueueUnmapMemObject(cqCommandQueue,cmPar,hostmap,0,0,0);
        
        openclRetTackle(clSetKernelArg(kernelSRAD,0,sizeof(cl_mem),&cmPar),"clSetKernelArg 0");
        openclRetTackle(clSetKernelArg(kernelSRAD,1,sizeof(cl_mem),&cmCqM),"clSetKernelArg 1");
        openclRetTackle(clSetKernelArg(kernelSRAD,2,sizeof(cl_mem),&cmImgSrc),"clSetKernelArg 2");
        openclRetTackle(clSetKernelArg(kernelSRAD,3,sizeof(cl_mem),&cmImgDes),"clSetKernelArg 3");
        openclRetTackle(clEnqueueNDRangeKernel(cqCommandQueue,kernelSRAD,3,0,global_work_size,NULL,0,0,0),"clEnqueueNDRangeKernel SRAD");
        openclRetTackle(clFinish(cqCommandQueue),"clFinish");
        
        cl_mem tmp;
        tmp         = cmImgSrc;
        cmImgSrc    = cmImgDes;
        cmImgDes    = tmp;
    }
    
    float* outBuffer=(float*)malloc(sizeof(float)*iSize);
    memset(outBuffer,0,sizeof(float)*iSize);
    clEnqueueReadBuffer(cqCommandQueue,cmImgSrc,CL_TRUE,0,sizeof(float)*iSize,outBuffer,0,0,0);
    printf("%f %f %f\n",outBuffer[2*iWidth+2],outBuffer[2*iWidth+3],outBuffer[2*iWidth+4]);
    
    for(int i=0;i<iHeight;i++)
    {
        for(int j=0;j<iWidth;j++)
        {
            imgDes->imageData[i*iStep+j]=(unsigned char)(outBuffer[i*iWidth+j]);
        }
    }
    
    return 0;
}
