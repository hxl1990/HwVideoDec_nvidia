# dependencies

- CUDA11.1
- FFmpeg 4.3.1
- Nvidia driver >= 456.71

# 使用说明

- videoDec \*videodec = new videoDec(GpuID, videoFileName, video_interval);
- void \*bgraPtr = videodec->getFrameAddr(); // 类内对原始图片进行了内存分配，getFrameAddr 可以获取内存地址，返回地址为显存和内存 UMA 地址，图片类型为 BGRA package 图片，与 opencv 内存分布一致
- void \*inTensor = videodec->outTensorMalloc(dstWidth, dstHeight, channel, MaxBatchsize); //分配进入 tensorrt 网络内存空间，也可以采用 cudaMalloc 自行分配管理，类型为 NCHW float
- RoiRects Rois; 定义 bgra->dnn input 的转换关系，最大支持 32 个 Roi ,处理方式为由 roi->resize->dnn input ,支持 通道归一化和 normal 归一化
- getSrcFrame 更新最近帧数据
