#pragma once

#include <stdint.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <stdio.h>

// using namespace libconfig;

#include <string>
#include <vector>

typedef unsigned char uchar;

struct RectRoi
{
    int x;
    int y;
    int w;
    int h;
};

enum Datatype
{
    INFER_FP32 = 0,
    INFER_IN8
};

enum DataStruct
{
    INFER_NCHW_RGB = 0,
    INFER_NCHW_BGR,
    INFER_NHWC_RGB,
    INFER_NHWC_BGR
};

enum DataPreMode
{
    MODE_SCALE = 0,
    MODE_NORMAL_SCALE
};

struct videoinfoPara
{
    char URI[64];
    int interval;
};

struct dnnRTPara
{
    char engineFileName[64];
    char inputTensorName[32];
    char outputTensorNames[4][32];
    int outTensorNum;
};

struct dnnpreinfo
{
    float channel_b;
    float channel_g;
    float channel_r;
};

struct videoinferPara
{
    dnnRTPara rtPara;
    // Datatype inputTensorType;
    // DataStruct inputTensorStruct;
    int net_type;
    int batchSize;
    int primaryInferIndex; // -1 all
    int primaryInferID;    //

    int inferID;      // 10 / 20 /30
    bool primaryFlag; //detect
    DataPreMode premode;
    float scale;
    dnnpreinfo input_mean;
    dnnpreinfo input_std;
    /* data */
};

struct RoiRect
{
    int x, y, w, h;
    /* data */
};

struct RoiRects
{
    int roiNum;
    DataPreMode premode;
    dnnpreinfo input_mean;
    dnnpreinfo input_std;
    RoiRect rois[32];
    /* data */
};

struct NormalMode
{
    DataPreMode premode;
    dnnpreinfo input_mean;
    dnnpreinfo input_std;
};

void resizePlanarCropBGRA2NvTensor(uchar *dpSrc, unsigned int nSrcPitch, unsigned int nSrcWidth, unsigned int nSrcHeight,
                                   float *dpDst, unsigned int nDstPitch, unsigned int nDstWidth, unsigned int nDstHeight, RoiRects Rois, cudaStream_t stream = NULL);