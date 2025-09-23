#include "oww.h"
#include <cstdio>

// 完全简化的OWW实现，避免所有复杂的内存管理
struct oww_handle {
  int mel_win=97, mel_bins=32;
  int det_T=41, det_D=96;
  float threshold=0.5f;
  float last=0.0f;
  
  // 使用固定大小的数组，避免动态内存管理
  float pcm_buf[10000];
  float mel_buf[10000];
  float emb_buf[10000];
  int pcm_size=0, mel_size=0, emb_size=0;
};

oww_handle* oww_create(const char* melspec_onnx,
                       const char* embed_onnx,
                       const char* detector_onnx,
                       int threads,
                       float threshold){
  // 完全简化：只创建基本结构
  auto h = new oww_handle();
  h->mel_win = 97;
  h->mel_bins = 32;
  h->det_T = 41;
  h->det_D = 96;
  h->threshold = threshold;
  h->last = 0.0f;
  h->pcm_size = 0;
  h->mel_size = 0;
  h->emb_size = 0;
  
  return h;
}

void oww_reset(oww_handle* h){
  h->pcm_size = 0;
  h->mel_size = 0;
  h->emb_size = 0;
  h->last = 0.0f;
}

float oww_last_score(const oww_handle* h){ 
  return h->last; 
}

size_t oww_recommended_chunk(){ 
  return 1280; 
}

int oww_process_f32(oww_handle* h, const float* pcm, size_t samples){
  // 简化：直接返回0，不进行任何处理
  return 0;
}

int oww_process_i16(oww_handle* h, const short* x, size_t samples){
  // 简化：直接返回0，不进行任何处理
  return 0;
}

void oww_destroy(oww_handle* h){ 
  delete h; 
}