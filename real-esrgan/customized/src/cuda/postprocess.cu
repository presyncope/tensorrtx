#include <cuda_runtime.h>
#include "postprocess.h"

struct ColorConvParams {
  // Coefficients pre-scaled by 255 (Full) or 219/224 (Limited)
  float3 y_coef; float y_off;
  float3 u_coef; float u_off;
  float3 v_coef; float v_off;
};

// Helper for safe saturation and casting
__device__ __forceinline__ uint8_t clip_u8(float v) {
  return static_cast<uint8_t>(__float2int_rn(fminf(fmaxf(v, 0.0f), 255.0f)));
}

__global__ void postprocess_iyuv_kernel(
  const half* __restrict__ srcTensor,     
  uint8_t* __restrict__ dstY,
  uint8_t* __restrict__ dstU,
  uint8_t* __restrict__ dstV,
  int dstWidth, 
  int dstHeight,                          
  int tileWidth, 
  int tileHeight,                         
  int padH, 
  int padV,                               
  int scale,                              
  int tiles_per_row,
  const ColorConvParams color_params
) {
  // -----------------------------------------------------------
  // 1. Thread Mapping (Parallelize over VALID scaled tile pixels)
  // -----------------------------------------------------------
  const int scaled_tileW = tileWidth * scale;
  const int scaled_tileH = tileHeight * scale;
  const int scaled_padH = padH * scale;
  const int scaled_padV = padV * scale;

  const int x_local = blockIdx.x * blockDim.x + threadIdx.x; // coordinate within scaled tile
  const int y_local = blockIdx.y * blockDim.y + threadIdx.y;
  const int n = blockIdx.z; // Tile index

  if (x_local >= scaled_tileW || y_local >= scaled_tileH)
    return;

  // -----------------------------------------------------------
  // 2. Calculate Global Output Coordinates (Stitching)
  // -----------------------------------------------------------
  const int tile_idx_x = n % tiles_per_row;
  const int tile_idx_y = n / tiles_per_row;

  const int x_global = tile_idx_x * scaled_tileW + x_local;
  const int y_global = tile_idx_y * scaled_tileH + y_local;

  if (x_global >= dstWidth || y_global >= dstHeight)
    return;

  // -----------------------------------------------------------
  // 3. Calculate Input Tensor Coordinates (Adding Padding)
  // -----------------------------------------------------------
  const int tensorW = scaled_tileW + 2 * scaled_padH; // Width of one tensor batch item
  const int tensorH = scaled_tileH + 2 * scaled_padV;
    
  const int x_tensor = x_local + scaled_padH; 
  const int y_tensor = y_local + scaled_padV;

  const size_t plane_size = tensorW * tensorH;
  const half* src_tile = srcTensor + n * 3 * plane_size;
  const size_t pixel_idx = y_tensor * tensorW + x_tensor;

  // -----------------------------------------------------------
  // 4. Read RGB (FP16 -> Float) from Tensor
  // -----------------------------------------------------------
  float red = __half2float(src_tile[pixel_idx]);
  float green = __half2float(src_tile[plane_size + pixel_idx]);
  float blue = __half2float(src_tile[2 * plane_size + pixel_idx]);

  // -----------------------------------------------------------
  // 5. Convert RGB to YUV
  // -----------------------------------------------------------    
  // Apply affine transform: val = (R*c1 + G*c2 + B*c3) + offset
  // Coefficients are pre-calculated on host to include scaling (255 or 219/224)
  float y_val = fmaf(color_params.y_coef.x, red, fmaf(color_params.y_coef.y, green, fmaf(color_params.y_coef.z, blue, color_params.y_off)));
  float u_val = fmaf(color_params.u_coef.x, red, fmaf(color_params.u_coef.y, green, fmaf(color_params.u_coef.z, blue, color_params.u_off)));
  float v_val = fmaf(color_params.v_coef.x, red, fmaf(color_params.v_coef.y, green, fmaf(color_params.v_coef.z, blue, color_params.v_off)));
  
  // -----------------------------------------------------------
  // 6. Write Y Plane, U/V Planes
  // -----------------------------------------------------------
  dstY[y_global * dstWidth + x_global] = clip_u8(y_val);

  if (!(x_global & 1) && !(y_global & 1)) {
    int uv_idx = (y_global >> 1) * (dstWidth >> 1) + (x_global >> 1);
    dstU[uv_idx] = clip_u8(u_val);
    dstV[uv_idx] = clip_u8(v_val);
  }
}

void postprocess(PostprocessParams& params, cudaStream_t stream) {
  // 1. Calculate Scaled Dimensions
  int dstW = params.width * params.scale;
  int dstH = params.height * params.scale;
  int scaled_tileW = params.tileWidth * params.scale;
  int scaled_tileH = params.tileHeight * params.scale;

  int tiles_per_row = (params.width + params.tileWidth - 1) / params.tileWidth;
  int tiles_per_col = (params.height + params.tileHeight - 1) / params.tileHeight;
  int total_tiles = tiles_per_row * tiles_per_col;

  // 2. Setting block and grid sizes
  dim3 block(32, 32, 1);
  dim3 grid;
  grid.x = (scaled_tileW + block.x - 1) / block.x;
  grid.y = (scaled_tileH + block.y - 1) / block.y;
  grid.z = total_tiles;

  // 3. Set color conversion parameters
  ColorConvParams color_params;
  if(params.video_full_range_flag) {
    // BT.709 Full Range (0-255)
    // Y = (0.2126 R + 0.7152 G + 0.0722 B) * 255
    color_params.y_coef = make_float3(0.2126f * 255.0f, 0.7152f * 255.0f, 0.0722f * 255.0f);
    color_params.y_off  = 0.0f;
    // U = (-0.1146 R - 0.3854 G + 0.5000 B) * 255 + 128
    color_params.u_coef = make_float3(-0.1146f * 255.0f, -0.3854f * 255.0f, 0.5000f * 255.0f);
    color_params.u_off  = 128.0f;
    // V = (0.5000 R - 0.4542 G - 0.0458 B) * 255 + 128
    color_params.v_coef = make_float3(0.5000f * 255.0f, -0.4542f * 255.0f, -0.0458f * 255.0f);
    color_params.v_off  = 128.0f;
  } else {
    // BT.709 Limited Range (Y: 16-235, UV: 16-240)
    // Y = (0.2126 R + 0.7152 G + 0.0722 B) * 219 + 16
    color_params.y_coef = make_float3(0.2126f * 219.0f, 0.7152f * 219.0f, 0.0722f * 219.0f);
    color_params.y_off  = 16.0f;
    // U = (-0.1146 R - 0.3854 G + 0.5000 B) * 224 + 128
    color_params.u_coef = make_float3(-0.1146f * 224.0f, -0.3854f * 224.0f, 0.5000f * 224.0f);
    color_params.u_off  = 128.0f;
    // V = (0.5000 R - 0.4542 G - 0.0458 B) * 224 + 128
    color_params.v_coef = make_float3(0.5000f * 224.0f, -0.4542f * 224.0f, -0.0458f * 224.0f);
    color_params.v_off  = 128.0f;
  }

  postprocess_iyuv_kernel<<<grid, block, 0, stream>>>(
    params.d_srcTensor, 
    params.d_dstY, 
    params.d_dstU, 
    params.d_dstV, 
    dstW, 
    dstH, 
    params.tileWidth, 
    params.tileHeight, 
    params.padH, 
    params.padV, 
    params.scale, 
    tiles_per_row,
    color_params
  );
}