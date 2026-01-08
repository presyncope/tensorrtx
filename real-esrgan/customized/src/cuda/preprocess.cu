#include <cuda_runtime.h>
#include "preprocess.h"

struct ColorConvParams {
  float y_offset;
  float y_mult;
  float m1;
  float m21;
  float m22;
  float m3;
};

__global__ void preprocess_iyuv_kernel(
  const uint8_t* __restrict__ srcY, 
  const uint8_t* __restrict__ srcU,
  const uint8_t* __restrict__ srcV, 
  half* __restrict__ dstTensor, 
  int srcWidth, 
  int srcHeight, 
  int tileWidth, 
  int tileHeight, 
  int padH,                                 // horizontal padding
  int padV,                                 // vertical padding   
  int tiles_per_row,
  const ColorConvParams color_params
) {
  // -----------------------------------------------------------
  // 1. Thread Mapping (Parallelize over VALID tile pixels)
  // -----------------------------------------------------------
  const int outWidth = tileWidth + 2 * padH;
  const int outHeight = tileHeight  + 2 * padV;
  const int local_x = blockIdx.x * blockDim.x + threadIdx.x;
  const int local_y = blockIdx.y * blockDim.y + threadIdx.y;
  const int n = blockIdx.z; // Tile index

  if (local_x >= outWidth || local_y >= outHeight)
    return;

  // -----------------------------------------------------------
  // 2. Calculate input coordinates with padding(mirror)
  // -----------------------------------------------------------
  const int tile_idx_x = n % tiles_per_row;
  const int tile_idx_y = n / tiles_per_row;

  int x_global = tile_idx_x * tileWidth + local_x - padH;
  int y_global = tile_idx_y * tileHeight + local_y - padV;
  x_global = abs(x_global);
  y_global = abs(y_global);
  x_global = (srcWidth - 1) - abs(x_global - (srcWidth - 1));
  y_global = (srcHeight - 1) - abs(y_global - (srcHeight - 1));

  // -----------------------------------------------------------
  // 3. Read YUV, Normalize, Convert to RGB, and Write to output tensor
  // -----------------------------------------------------------
  float y_val = static_cast<float>(srcY[y_global * srcWidth + x_global]);

  int uv_x = x_global >> 1;
  int uv_y = y_global >> 1;
  int uv_offset = uv_y * (srcWidth >> 1) + uv_x;

  float u_val = static_cast<float>(srcU[uv_offset]);
  float v_val = static_cast<float>(srcV[uv_offset]);

  // convert yuv to rgb
  y_val = __fmaf(color_params.y_mult, y_val, color_params.y_offset); 
  u_val -= 128.0f;
  v_val -= 128.0f;

  float red = __fmaf(color_params.m1, v_val, y_val);
  float green = __fmaf(color_params.m21, u_val, y_val);
  green = __fmaf(color_params.m22, v_val, green);
  float blue = __fmaf(color_params.m3, u_val, y_val);
    
  // normalize
  constexpr float inv_255 = 1.0f / 255.0f;
  red = __saturatef(red * inv_255);
  green = __saturatef(green * inv_255);
  blue = __saturatef(blue * inv_255);

  // write to output tensor in NCHW format
  const size_t plane_size = outWidth * outHeight;
  half* dst_tile = dstTensor + (n * 3 * plane_size);
  const size_t pixel_idx = local_y * outWidth + local_x;

  dst_tile[pixel_idx] = __float2half(red);
  dst_tile[plane_size + pixel_idx] = __float2half(green);
  dst_tile[2 * plane_size + pixel_idx] = __float2half(blue);
}

void preprocess(PreprocessParams& params, cudaStream_t stream) {
  // 1. Calculate output tile dimensions
  int outW = params.tileWidth + 2 * params.padH;
  int outH = params.tileHeight + 2 * params.padV;
  int numTilesW = (params.width + params.tileWidth - 1) / params.tileWidth;
  int numTilesH = (params.height + params.tileHeight - 1) / params.tileHeight;
  int totalTitles = numTilesW * numTilesH;

  // 2. Setting block and grid sizes
  dim3 block(32, 32, 1);  // 1024 threads
  dim3 grid;
  grid.x = (outW + block.x - 1) / block.x;
  grid.y = (outH + block.y - 1) / block.y;
  grid.z = totalTitles;

  // 3. Color conversion parameters (BT.709)
  ColorConvParams color_params;
  if(params.video_full_range_flag) {
    color_params.y_offset = 0.0f;
    color_params.y_mult = 1.0f;
    color_params.m1 = 1.5748f;
    color_params.m21 = -0.1873f;
    color_params.m22 = -0.4681f;
    color_params.m3 = 1.8556f;
  } else {
    color_params.y_offset = -16.0f;
    color_params.y_mult = 1.164f;
    color_params.m1 = 1.793f;
    color_params.m21 = -0.213f;
    color_params.m22 = -0.533f;
    color_params.m3 = 2.112f;
  }
  // 4. Launch kernel
  preprocess_iyuv_kernel<<<grid, block, 0, stream>>>(
      params.d_srcY, params.d_srcU, params.d_srcV, params.d_dstTensor,
      params.width, params.height,
      params.tileWidth, params.tileHeight,
      params.padH, params.padV,
      numTilesW,
      color_params
  );
}