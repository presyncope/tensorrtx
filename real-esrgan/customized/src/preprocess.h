#pragma once
#include <cstdint>
#include <cuda_fp16.h>

struct PreprocessParams {
  // input YUV pointers and output tensor pointer
  const uint8_t* d_srcY;
  const uint8_t* d_srcU;
  const uint8_t* d_srcV;
  half* d_dstTensor;

  // image size
  int width;
  int height;

  // tile and padding
  int tileWidth;
  int tileHeight;
  int padH;
  int padV;

  // color space
  bool video_full_range_flag;
};


void preprocess(PreprocessParams& params, cudaStream_t stream);