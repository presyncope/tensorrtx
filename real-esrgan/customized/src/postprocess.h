#pragma once
#include <cstdint>
#include <cuda_fp16.h>

struct PostprocessParams {
  // input tensor pointer and output YUV pointers
  const half* d_srcTensor;
  uint8_t* d_dstY;
  uint8_t* d_dstU;
  uint8_t* d_dstV;

  // unscaled image size
  int width;
  int height;

  // unscaled tile and padding
  int tileWidth;
  int tileHeight;
  int padH;
  int padV;

  int scale;

  // color space
  bool video_full_range_flag;
};

void postprocess(PostprocessParams& params, cudaStream_t stream);