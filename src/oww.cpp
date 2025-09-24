#include "oww.h"
#include <onnxruntime_c_api.h>
#include <cstring>
#include <vector>
#include <deque>
#include <stdexcept>
#include <cstdio>
#include <algorithm>

static const OrtApi* A() { return OrtGetApiBase()->GetApi(ORT_API_VERSION); }

struct OwwOrt {
  OrtEnv* env=nullptr;
  OrtSessionOptions* so=nullptr;
  OrtSession* mels=nullptr;
  OrtSession* det=nullptr;
  OrtAllocator* alloc=nullptr;
  std::string mels_in0, mels_out0;
  std::string det_in0, det_out0;
  
  ~OwwOrt(){
    if(det)   A()->ReleaseSession(det);
    if(mels)  A()->ReleaseSession(mels);
    if(so)    A()->ReleaseSessionOptions(so);
    if(env)   A()->ReleaseEnv(env);
  }
};

struct oww_handle {
  OwwOrt ort;
  
  // 固定形状参数
  int mel_win=97, mel_bins=32;     // mel输出 [1, mel_win, mel_bins, 1]
  int det_T=36, det_D=96;          // detector输入 [1, det_T, det_D]
  
  float threshold=0.5f;
  float last=0.0f;
  
  // 缓冲
  std::deque<float> pcm_buf;       // 原始PCM float
  std::deque<float> mel_buf;       // mel特征 (frame, mel_bins)
  
  static void ORTCHK(OrtStatus* st){ 
    if(st){ 
      const char* m=A()->GetErrorMessage(st); 
      std::string s=m?m:"ORT error"; 
      A()->ReleaseStatus(st); 
      throw std::runtime_error(s);
    } 
  }
};

static OrtSession* load_session(OrtEnv* env, OrtSessionOptions* so, const char* path){
  if (!path || strlen(path) == 0) {
    throw std::runtime_error("模型路径为空");
  }
  
  OrtSession* session = nullptr;
  oww_handle::ORTCHK(A()->CreateSession(env, path, so, &session));
  printf("✅ 加载模型: %s\n", path);
  return session;
}

static std::string ort_get_input_name(oww_handle* h, OrtSession* session, size_t index){
  char* tmp = nullptr;
  oww_handle::ORTCHK(A()->SessionGetInputName(session, index, h->ort.alloc, &tmp));
  std::string name(tmp);
  h->ort.alloc->Free(h->ort.alloc, tmp);
  return name;
}

static std::string ort_get_output_name(oww_handle* h, OrtSession* session, size_t index){
  char* tmp = nullptr;
  oww_handle::ORTCHK(A()->SessionGetOutputName(session, index, h->ort.alloc, &tmp));
  std::string name(tmp);
  h->ort.alloc->Free(h->ort.alloc, tmp);
  return name;
}

oww_handle* oww_create(const char* melspec_onnx,
                       const char* embed_onnx,
                       const char* detector_onnx,
                       int threads,
                       float threshold){
  printf("🔍 OWW两链模式初始化...\n");
  
  auto h = new oww_handle();
  h->threshold = threshold;
  
  // 初始化ONNX Runtime
  oww_handle::ORTCHK(A()->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "oww", &h->ort.env));
  oww_handle::ORTCHK(A()->CreateSessionOptions(&h->ort.so));
  oww_handle::ORTCHK(A()->SetIntraOpNumThreads(h->ort.so, threads));
  oww_handle::ORTCHK(A()->GetAllocatorWithDefaultOptions(&h->ort.alloc));
  
  // 加载模型
  h->ort.mels = load_session(h->ort.env, h->ort.so, melspec_onnx);
  h->ort.det = load_session(h->ort.env, h->ort.so, detector_onnx);
  
  // 获取输入输出名称
  h->ort.mels_in0 = ort_get_input_name(h, h->ort.mels, 0);
  h->ort.mels_out0 = ort_get_output_name(h, h->ort.mels, 0);
  h->ort.det_in0 = ort_get_input_name(h, h->ort.det, 0);
  h->ort.det_out0 = ort_get_output_name(h, h->ort.det, 0);
  
  printf("✅ OWW两链模式初始化完成, 阈值: %.3f\n", threshold);
  return h;
}

void oww_reset(oww_handle* h){
  h->pcm_buf.clear(); 
  h->mel_buf.clear(); 
  h->last=0.0f;
}

float oww_last_score(const oww_handle* h){ 
  return h->last; 
}

size_t oww_recommended_chunk(){ 
  return 1280; // ~80ms@16k
}

void oww_destroy(oww_handle* h){
  delete h;
}

// 运行mel spectrogram模型
static void run_mels(oww_handle* h, const float* pcm, size_t samples){
  OrtMemoryInfo* mi=nullptr; 
  oww_handle::ORTCHK(A()->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &mi));
  
  // 输入: [1, samples]
  OrtValue* in=nullptr;
  int64_t in_shape[2] = {1, (int64_t)samples};
  oww_handle::ORTCHK(A()->CreateTensorWithDataAsOrtValue(mi, (void*)pcm, samples*sizeof(float),
                                                         in_shape, 2, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &in));
  A()->ReleaseMemoryInfo(mi);
  
  // 推理
  const char* in_names[]={h->ort.mels_in0.c_str()}; 
  const char* out_names[]={h->ort.mels_out0.c_str()};
  OrtValue* out=nullptr; 
  oww_handle::ORTCHK(A()->Run(h->ort.mels, nullptr, in_names, (const OrtValue* const*)&in, 1, out_names, 1, &out));
  A()->ReleaseValue(in);
  
  // 读取mel输出 [1, frames, mel_bins, 1]
  float* mel_data=nullptr; 
  oww_handle::ORTCHK(A()->GetTensorMutableData(out, (void**)&mel_data));
  
  OrtTensorTypeAndShapeInfo* info=nullptr;
  oww_handle::ORTCHK(A()->GetTensorTypeAndShape(out, &info));
  size_t dim_count = 0;
  oww_handle::ORTCHK(A()->GetDimensionsCount(info, &dim_count));
  std::vector<int64_t> dims(dim_count);
  oww_handle::ORTCHK(A()->GetDimensions(info, dims.data(), dim_count));
  A()->ReleaseTensorTypeAndShapeInfo(info);
  
  if (dim_count >= 2) {
    int frames = (int)dims[1];
    int mel_bins = dim_count >= 3 ? (int)dims[2] : 32;
    
    // 添加到mel_buf
    for(int f = 0; f < frames; f++) {
      for(int b = 0; b < mel_bins; b++) {
        h->mel_buf.push_back(mel_data[f * mel_bins + b]);
      }
    }
  }
  
  A()->ReleaseValue(out);
}

// 两链检测：mel -> detector
static int try_detect_two_chain(oww_handle* h){
  int mel_frames = (int)h->mel_buf.size() / h->mel_bins;
  
  if(mel_frames < h->det_T) {
    return 0; // 帧数不足
  }
  
  // 取最后det_T帧，转换为[1, det_T, det_D]
  std::vector<float> detector_input;
  detector_input.reserve(h->det_T * h->det_D);
  
  int start_frame = mel_frames - h->det_T;
  for(int f = 0; f < h->det_T; f++) {
    int frame_idx = start_frame + f;
    
    // 每帧mel_bins维重复3次变成det_D维
    for(int repeat = 0; repeat < 3; repeat++) {
      for(int b = 0; b < h->mel_bins; b++) {
        int idx = frame_idx * h->mel_bins + b;
        if (idx < h->mel_buf.size()) {
          detector_input.push_back(h->mel_buf[idx]);
        } else {
          detector_input.push_back(0.0f);
        }
      }
    }
  }
  
  // 推理
  OrtMemoryInfo* mi=nullptr; 
  oww_handle::ORTCHK(A()->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &mi));
  
  OrtValue* in=nullptr;
  int64_t shape[3] = {1, h->det_T, h->det_D};
  oww_handle::ORTCHK(A()->CreateTensorWithDataAsOrtValue(mi, detector_input.data(), detector_input.size()*sizeof(float),
                                                         shape, 3, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &in));
  A()->ReleaseMemoryInfo(mi);
  
  const char* in_names[]={h->ort.det_in0.c_str()}; 
  const char* out_names[]={h->ort.det_out0.c_str()};
  OrtValue* out=nullptr; 
  oww_handle::ORTCHK(A()->Run(h->ort.det, nullptr, in_names, (const OrtValue* const*)&in, 1, out_names, 1, &out));
  A()->ReleaseValue(in);
  
  // 读取检测结果
  float* p=nullptr; 
  oww_handle::ORTCHK(A()->GetTensorMutableData(out, (void**)&p));
  h->last = p[0];
  
  A()->ReleaseValue(out);
  
  printf("🔍 唤醒词检测: score=%.6f, 阈值=%.3f, 结果=%s\n", 
         h->last, h->threshold, (h->last >= h->threshold) ? "触发" : "未触发");
  
  return (h->last >= h->threshold) ? 1 : 0;
}

static int feed_pcm(oww_handle* h, const float* pcm, size_t samples){
  size_t off = 0;
  int fired = 0;
  
  while(off < samples){
    size_t n = std::min(samples - off, (size_t)1280);
    
    // 运行mel spectrogram
    run_mels(h, pcm + off, n);
    
    // 两链检测
    fired |= try_detect_two_chain(h);
    
    off += n;
  }
  
  return fired ? 1 : 0;
}

int oww_feed_pcm_f32(oww_handle* h, const float* pcm, size_t samples){
  return feed_pcm(h, pcm, samples);
}

int oww_feed_pcm_s16(oww_handle* h, const int16_t* pcm, size_t samples){
  std::vector<float> f32_pcm(samples);
  for(size_t i = 0; i < samples; i++){
    f32_pcm[i] = pcm[i] / 32768.0f;
  }
  return feed_pcm(h, f32_pcm.data(), samples);
}