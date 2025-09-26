#include "oww.h"
#include <onnxruntime/core/session/onnxruntime_c_api.h>
#include <cstring>
#include <vector>
#include <deque>
#include <stdexcept>
#include <cstdio>
#include <algorithm>
#include <ctime>

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

// ==================== 新的KWS单模型实现 ====================

struct kws_handle {
  OwwOrt ort;
  
  // 固定参数
  int win = 16000;        // 1.0s窗口
  int hop = 160;          // 10ms跳步
  int smooth = 3;         // 滑动平均帧数
  int cooldown_frames = 30; // 冷却帧数
  
  float threshold = 0.65f;
  float last = 0.0f;
  
  // 环形缓冲
  std::vector<int16_t> ring_buf;
  std::vector<float> ma_buf;  // 滑动平均缓冲
  int ma_idx = 0;
  int cooldown = 0;
  
  // 输入输出名称
  std::string input_name;
  std::string output_name;
  
  static void ORTCHK(OrtStatus* st){ 
    if(st){ 
      const char* m=A()->GetErrorMessage(st); 
      std::string s=m?m:"ORT error"; 
      A()->ReleaseStatus(st); 
      throw std::runtime_error(s);
    } 
  }
};

kws_handle* kws_create(const char* model_path, int threads, float threshold){
  printf("🔍 KWS单模型初始化...\n");
  
  auto h = new kws_handle();
  h->threshold = threshold;
  h->ring_buf.resize(h->win, 0);
  h->ma_buf.resize(h->smooth, 0.0f);
  
  // 初始化ONNX Runtime
  kws_handle::ORTCHK(A()->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "kws", &h->ort.env));
  kws_handle::ORTCHK(A()->CreateSessionOptions(&h->ort.so));
  kws_handle::ORTCHK(A()->SetIntraOpNumThreads(h->ort.so, threads));
  kws_handle::ORTCHK(A()->GetAllocatorWithDefaultOptions(&h->ort.alloc));
  
  // 加载模型
  h->ort.mels = load_session(h->ort.env, h->ort.so, model_path);
  
  // 获取输入输出名称
  char* tmp = nullptr;
  kws_handle::ORTCHK(A()->SessionGetInputName(h->ort.mels, 0, h->ort.alloc, &tmp));
  h->input_name = std::string(tmp);
  h->ort.alloc->Free(h->ort.alloc, tmp);
  
  kws_handle::ORTCHK(A()->SessionGetOutputName(h->ort.mels, 0, h->ort.alloc, &tmp));
  h->output_name = std::string(tmp);
  h->ort.alloc->Free(h->ort.alloc, tmp);
  
  // 打印模型输入输出信息
  fprintf(stderr, "🔍 模型输入输出信息:\n");
  fprintf(stderr, "   - 输入名称: %s\n", h->input_name.c_str());
  fprintf(stderr, "   - 输出名称: %s\n", h->output_name.c_str());
  
  // 获取输入输出形状信息
  size_t input_count, output_count;
  kws_handle::ORTCHK(A()->SessionGetInputCount(h->ort.mels, &input_count));
  kws_handle::ORTCHK(A()->SessionGetOutputCount(h->ort.mels, &output_count));
  
  fprintf(stderr, "   - 输入数量: %zu, 输出数量: %zu\n", input_count, output_count);
  
  // 获取输入输出类型信息
  OrtTypeInfo* input_type_info = nullptr;
  OrtTypeInfo* output_type_info = nullptr;
  kws_handle::ORTCHK(A()->SessionGetInputTypeInfo(h->ort.mels, 0, &input_type_info));
  kws_handle::ORTCHK(A()->SessionGetOutputTypeInfo(h->ort.mels, 0, &output_type_info));
  
  if (input_type_info) {
    const OrtTensorTypeAndShapeInfo* input_tensor_info = nullptr;
    kws_handle::ORTCHK(A()->CastTypeInfoToTensorInfo(input_type_info, &input_tensor_info));
    
    ONNXTensorElementDataType input_type;
    kws_handle::ORTCHK(A()->GetTensorElementType(input_tensor_info, &input_type));
    fprintf(stderr, "   - 输入类型: %d (INT16=%d)\n", input_type, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16);
    
    size_t input_dim_count;
    kws_handle::ORTCHK(A()->GetDimensionsCount(input_tensor_info, &input_dim_count));
    fprintf(stderr, "   - 输入维度数: %zu\n", input_dim_count);
    
    for (size_t i = 0; i < input_dim_count; i++) {
      int64_t dim;
      kws_handle::ORTCHK(A()->GetDimensions(input_tensor_info, &dim, i));
      fprintf(stderr, "   - 输入维度[%zu]: %ld\n", i, dim);
    }
    
    A()->ReleaseTypeInfo(input_type_info);
  }
  
  if (output_type_info) {
    const OrtTensorTypeAndShapeInfo* output_tensor_info = nullptr;
    kws_handle::ORTCHK(A()->CastTypeInfoToTensorInfo(output_type_info, &output_tensor_info));
    
    ONNXTensorElementDataType output_type;
    kws_handle::ORTCHK(A()->GetTensorElementType(output_tensor_info, &output_type));
    fprintf(stderr, "   - 输出类型: %d (FLOAT=%d)\n", output_type, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
    
    size_t output_dim_count;
    kws_handle::ORTCHK(A()->GetDimensionsCount(output_tensor_info, &output_dim_count));
    fprintf(stderr, "   - 输出维度数: %zu\n", output_dim_count);
    
    for (size_t i = 0; i < output_dim_count; i++) {
      int64_t dim;
      kws_handle::ORTCHK(A()->GetDimensions(output_tensor_info, &dim, i));
      fprintf(stderr, "   - 输出维度[%zu]: %ld\n", i, dim);
    }
    
    A()->ReleaseTypeInfo(output_type_info);
  }
  
  fprintf(stderr, "✅ KWS单模型初始化完成, 阈值: %.3f\n", threshold);
  fprintf(stderr, "   - 输入: %s, 输出: %s\n", h->input_name.c_str(), h->output_name.c_str());
  fflush(stderr);
  return h;
}

void kws_reset(kws_handle* h){
  std::fill(h->ring_buf.begin(), h->ring_buf.end(), 0);
  std::fill(h->ma_buf.begin(), h->ma_buf.end(), 0.0f);
  h->ma_idx = 0;
  h->cooldown = 0;
  h->last = 0.0f;
}

float kws_last_score(const kws_handle* h){
  return h->last;
}

size_t kws_recommended_chunk(){
  return 160; // 10ms@16k
}

void kws_destroy(kws_handle* h){
  delete h;
}

int kws_process_i16(kws_handle* h, const short* pcm, size_t samples){
  // 函数入口日志 - 2024-09-26 09:45
  fprintf(stderr, "🔍 KWS函数调用: samples=%zu, 时间戳=%ld\n", samples, time(nullptr));
  fflush(stderr);
  
  int fired = 0;
  
  for(size_t i = 0; i < samples; i += h->hop){
    size_t hop_size = std::min((size_t)h->hop, samples - i);
    
    // 滑窗：右移 WIN-HOP，拷贝 HOP
    memmove(h->ring_buf.data(), h->ring_buf.data() + h->hop, (h->win - h->hop) * sizeof(int16_t));
    memcpy(h->ring_buf.data() + (h->win - h->hop), pcm + i, hop_size * sizeof(int16_t));
    
    // 验证输入数据：统计新拷贝的数据
    int16_t* new_data = h->ring_buf.data() + (h->win - h->hop);
    int16_t max_val = 0, min_val = 0;
    int non_zero_count = 0;
    for(size_t j = 0; j < hop_size; j++) {
        int16_t val = new_data[j];
        max_val = std::max(max_val, val);
        min_val = std::min(min_val, val);
        if(val != 0) non_zero_count++;
    }
    fprintf(stderr, "KWS输入验证: max=%d min=%d 非零=%d/%zu 原始peak=%d\n", 
            max_val, min_val, non_zero_count, hop_size, (i < samples) ? pcm[i] : 0);
    
    // 实验：注入测试数据看看模型是否响应
    static int test_counter = 0;
    if (test_counter < 5) {
        fprintf(stderr, "🧪 实验：注入测试数据 (第%d次)\n", test_counter);
        for(size_t j = 0; j < h->win; j++) {
            h->ring_buf[j] = (int16_t)(1000 * sin(j * 0.1)); // 注入正弦波
        }
        test_counter++;
    }
    
    // 创建输入张量
    OrtMemoryInfo* mi = nullptr;
    kws_handle::ORTCHK(A()->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &mi));
    
    OrtValue* input = nullptr;
    int64_t dims[2] = {1, h->win};
    kws_handle::ORTCHK(A()->CreateTensorWithDataAsOrtValue(mi, h->ring_buf.data(), 
                                                          h->win * sizeof(int16_t), dims, 2, 
                                                          ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16, &input));
    A()->ReleaseMemoryInfo(mi);
    
    // 推理
    const char* input_names[] = {h->input_name.c_str()};
    const char* output_names[] = {h->output_name.c_str()};
    OrtValue* output = nullptr;
    
    fprintf(stderr, "🔍 开始模型推理: 输入名称=%s, 输出名称=%s\n", 
            h->input_name.c_str(), h->output_name.c_str());
    fflush(stderr);
    
    kws_handle::ORTCHK(A()->Run(h->ort.mels, nullptr, input_names, (const OrtValue* const*)&input, 1,
                                output_names, 1, &output));
    
    fprintf(stderr, "✅ 模型推理完成\n");
    fflush(stderr);
    A()->ReleaseValue(input);
    
    // 获取分数
    float* score_ptr = nullptr;
    kws_handle::ORTCHK(A()->GetTensorMutableData(output, (void**)&score_ptr));
    float score = score_ptr[0];
    
    fprintf(stderr, "🔍 模型原始输出: score_ptr=%p, score=%.6f\n", score_ptr, score);
    fflush(stderr);
    
    A()->ReleaseValue(output);
    
    // 滑动平均
    h->ma_buf[h->ma_idx % h->smooth] = score;
    h->ma_idx++;
    
    float avg = 0.0f;
    for(int j = 0; j < h->smooth; j++){
      avg += h->ma_buf[j];
    }
    avg /= h->smooth;
    
    // 调试日志：显示原始分数和滑动平均
    fprintf(stderr, "KWS raw=%.6f avg=%.6f threshold=%.3f cooldown=%d\n", score, avg, h->threshold, h->cooldown);
    fflush(stderr);  // 强制刷新 stderr 缓冲区
    
    // 更新分数（每次循环都更新）
    h->last = avg;
    
    // 冷却处理
    if(h->cooldown > 0) h->cooldown--;
    
    // 触发检测
    if(avg > h->threshold && h->cooldown == 0){
      fprintf(stderr, "🔍 KWS触发: score=%.3f, 平均=%.3f, 阈值=%.3f\n", score, avg, h->threshold);
      h->cooldown = h->cooldown_frames;
      fired = 1;
    }
  }
  
  return fired;
}