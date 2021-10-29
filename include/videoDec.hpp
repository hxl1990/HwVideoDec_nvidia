#pragma once

class videoDec
{
public:
    videoDec(int GpuId, const char *fileURI, int video_interval);
    ~videoDec();
    bool getSrcFrame();
    int getSrcWidth();
    int getSrcHeight();

    void *getFrameAddr();
    void *outTensorMalloc(int width, int height, int channel, int MaxBatchsize);
    bool outTensorFree(void *tensorAddr);

private:
    void *res = nullptr;
};