#include <iostream>
#include "utils.hpp"
#include "videoDec.hpp"

#include <opencv2/opencv.hpp>

int main(int argc, char **argv)
{
    char *videoFileName = argv[1];

    videoDec *videodec = new videoDec(0, videoFileName, 0);
    bool finished = false;
    void *bgraPtr = videodec->getFrameAddr();
    if (bgraPtr == nullptr)
    {
        std::cout << "nullptr error " << std::endl;
    }

    int width = videodec->getSrcWidth();
    int height = videodec->getSrcHeight();
    // Mat
    cv::Mat img(height, width, CV_8UC4, bgraPtr);

    int dstWidth = 608;
    int dstHeight = 608;
    int channel = 3;
    int MaxBatchsize = 4;
    void *inTensor = videodec->outTensorMalloc(dstWidth, dstHeight, channel, MaxBatchsize);

    RoiRects Rois;
    Rois.roiNum = 2;
    Rois.premode = MODE_SCALE;

    Rois.input_std.channel_b = 1 / 255.0;
    Rois.input_std.channel_g = 1 / 255.0;
    Rois.input_std.channel_r = 1 / 255.0;

    Rois.input_mean.channel_b = 0;
    Rois.input_mean.channel_g = 0;
    Rois.input_mean.channel_r = 0;

    Rois.rois[0].x = 10;
    Rois.rois[0].y = 100;
    Rois.rois[0].w = 800;
    Rois.rois[0].h = 600;

    Rois.rois[1].x = 100;
    Rois.rois[1].y = 200;
    Rois.rois[1].w = 600;
    Rois.rois[1].h = 500;

    while (!finished)
    {
        finished = videodec->getSrcFrame();

        resizePlanarCropBGRA2NvTensor((unsigned char *)bgraPtr, width * 4, width, height,
                                      (float *)inTensor, dstWidth, dstWidth, dstHeight, Rois);
        cv::Mat out0(dstHeight, dstWidth, CV_32FC1, inTensor);
#define SHOW_IMG
#ifdef SHOW_IMG
        cv::imshow("src", img);

        cv::imshow("out", out0);

        cv::waitKey(10);
#endif
    }
    videodec->outTensorFree(inTensor);
    delete (videodec);

    return 0;
}
