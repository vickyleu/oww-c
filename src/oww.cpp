#include "oww.h"
#include <onnxruntime/core/session/onnxruntime_c_api.h>
#include <cstring>
#include <vector>
#include <deque>
#include <stdexcept>
#include <cstdio>
#include <algorithm>
#include <cmath>
#include <string>

static const OrtApi* A() { return OrtGetApiBase()->GetApi(ORT_API_VERSION); }

struct OwwOrt {
  OrtEnv* env=nullptr;
  OrtSessionOptions* so=nullptr;
  OrtSession* mel=nullptr;
  OrtSession* emb=nullptr;
  OrtSession* cls=nullptr;
  OrtAllocator* alloc=nullptr;
  std::string mel_in0, mel_out0;
  std::string emb_in0, emb_out0;
  std::string cls_in0, cls_out0;
  
  ~OwwOrt(){
    if(cls)   A()->ReleaseSession(cls);
    if(emb)   A()->ReleaseSession(emb);
    if(mel)   A()->ReleaseSession(mel);
    if(so)    A()->ReleaseSessionOptions(so);
    if(env)   A()->ReleaseEnv(env);
  }
};

struct oww_handle {
  OwwOrt ort;
  
  // 三链固定参数
  int mel_win=76;      // 每窗76帧
  int mel_bins=32;     // mel频谱32维
  int nwin=16;         // 16个窗
  
  // 估算参数（16kHz, 10ms hop, 25ms win）
  static const int SR = 16000;
  static const int HOP = 160;
  static const int WIN = 400;
  static const int NEED_FRAMES = 16 * 76;  // 1216帧用于完整推理
  static const int NEED_SAMPLES = 32000;   // 约2秒，快速响应唤醒词 (2*16000)
  
  float threshold=0.5f;
  float last=0.0f;
  
  // 环形缓冲区 - 扩大到支持完整的mel输入
  std::deque<float> pcm_buf;       // 原始PCM float，容量约195k
  
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
  fprintf(stderr, "✅ 加载模型: %s\n", path);
  fflush(stderr);
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

oww_handle* oww_create(const char* mel_onnx,
                       const char* emb_onnx, 
                       const char* cls_onnx,
                       int threads,
                       float threshold){
  fprintf(stderr, "🔍 OWW三链模式初始化...\n");
  fflush(stderr);
  
  auto h = new oww_handle();
  h->threshold = threshold;
  
  // 初始化ONNX Runtime
  oww_handle::ORTCHK(A()->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "oww", &h->ort.env));
  oww_handle::ORTCHK(A()->CreateSessionOptions(&h->ort.so));
  oww_handle::ORTCHK(A()->SetIntraOpNumThreads(h->ort.so, threads));
  oww_handle::ORTCHK(A()->GetAllocatorWithDefaultOptions(&h->ort.alloc));
  
  // 加载三链模型
  h->ort.mel = load_session(h->ort.env, h->ort.so, mel_onnx);
  h->ort.emb = load_session(h->ort.env, h->ort.so, emb_onnx);
  h->ort.cls = load_session(h->ort.env, h->ort.so, cls_onnx);
  
  // 获取输入输出名称
  h->ort.mel_in0 = ort_get_input_name(h, h->ort.mel, 0);
  h->ort.mel_out0 = ort_get_output_name(h, h->ort.mel, 0);
  h->ort.emb_in0 = ort_get_input_name(h, h->ort.emb, 0);
  h->ort.emb_out0 = ort_get_output_name(h, h->ort.emb, 0);
  h->ort.cls_in0 = ort_get_input_name(h, h->ort.cls, 0);
  h->ort.cls_out0 = ort_get_output_name(h, h->ort.cls, 0);
  
  fprintf(stderr, "✅ OWW三链模式初始化完成, 阈值: %.3f\n", threshold);
  fflush(stderr);
  return h;
}

void oww_reset(oww_handle* h){
  h->pcm_buf.clear(); 
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

// 与 Notebook 一致的 power→dB→[0,1]
static inline void power_to_db01(float* x, size_t n) {
  const float eps = 1e-10f;
  for (size_t i = 0; i < n; ++i) {
    float p  = fmaxf(x[i], eps);
    float db = 10.f * log10f(p);
    float y  = (db + 80.f) / 80.f;
    x[i] = y < 0.f ? 0.f : (y > 1.f ? 1.f : y);
  }
}

// 读取 ORT 输出形状，去掉所有 size=1 轴
static std::vector<int64_t> squeeze_dims(const std::vector<int64_t>& in) {
  std::vector<int64_t> d;
  d.reserve(in.size());
  for (auto v : in) if (v != 1) d.push_back(v);
  return d;
}

// 运行 mel：输入 [1, samples]，输出强制重排到 (32, T) 并做 dB01
static std::vector<float> run_mel(oww_handle* h, const float* pcm, size_t samples){
  OrtMemoryInfo* mi=nullptr; 
  oww_handle::ORTCHK(A()->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &mi));

  // in: [1, N]
  OrtValue* in=nullptr;
  const int64_t in_shape[2] = {1, (int64_t)samples};
  oww_handle::ORTCHK(A()->CreateTensorWithDataAsOrtValue(
      mi, (void*)pcm, samples*sizeof(float), in_shape, 2,
      ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &in));
  A()->ReleaseMemoryInfo(mi);

  // run
  const char* in_names[]  = {h->ort.mel_in0.c_str()};
  const char* out_names[] = {h->ort.mel_out0.c_str()};
  OrtValue* out=nullptr; 
  oww_handle::ORTCHK(A()->Run(h->ort.mel, nullptr, in_names,
                              (const OrtValue* const*)&in, 1,
                              out_names, 1, &out));
  A()->ReleaseValue(in);

  // get buffer + dims
  float* buf=nullptr; 
  oww_handle::ORTCHK(A()->GetTensorMutableData(out, (void**)&buf));
  OrtTensorTypeAndShapeInfo* info=nullptr;
  oww_handle::ORTCHK(A()->GetTensorTypeAndShape(out, &info));
  size_t dim_count = 0;
  oww_handle::ORTCHK(A()->GetDimensionsCount(info, &dim_count));
  std::vector<int64_t> raw_dims(dim_count);
  oww_handle::ORTCHK(A()->GetDimensions(info, raw_dims.data(), dim_count));
  A()->ReleaseTensorTypeAndShapeInfo(info);

  // squeeze 所有 size=1 维
  auto d = squeeze_dims(raw_dims);
  if (d.size() != 2 || (d[0] != 32 && d[1] != 32)) {
    A()->ReleaseValue(out);
    throw std::runtime_error("mel输出维度异常，期望含有 32 这一维");
  }

  int T = (d[0] == 32) ? (int)d[1] : (int)d[0];
  std::vector<float> mel32T(32 * (size_t)T);

  // 正确重排到 (32, T)（按 C row-major 线性索引）
  if (d[0] == 32) {
    // 内存序等价 (..,32,T,..)-> m*T + t
    for (int m = 0; m < 32; ++m)
      for (int t = 0; t < T; ++t)
        mel32T[m*(size_t)T + t] = buf[m*(size_t)T + t];
  } else {
    // 形如 (T,32) -> 线性索引 t*32 + m
    for (int t = 0; t < T; ++t)
      for (int m = 0; m < 32; ++m)
        mel32T[m*(size_t)T + t] = buf[t*32 + m];
  }

  // 调试（归一化前）
  {
    double mean=0, mn=1e30, mx=-1e30;
    size_t N = mel32T.size();
    for (size_t i=0;i<N;++i){ mean+=mel32T[i]; mn=std::min<double>(mn,mel32T[i]); mx=std::max<double>(mx,mel32T[i]); }
    mean/=N;
    fprintf(stderr, "🔍 mel原始功率: shape=(32,%d) mean=%.6g min=%.6g max=%.6g\n", T, mean, mn, mx);
  }

  // 与 Notebook 一致的 dB01
  power_to_db01(mel32T.data(), mel32T.size());

  // 调试（归一化后）
  {
    double mean=0, stdv=0; size_t N = mel32T.size();
    for (size_t i=0;i<N;++i) mean += mel32T[i];
    mean /= N;
    for (size_t i=0;i<N;++i) stdv += (mel32T[i]-mean)*(mel32T[i]-mean);
    stdv = std::sqrt(stdv/N);
    fprintf(stderr, "🔍 mel dB01: T=%d mean=%.4f std=%.4f first6=[%.3f %.3f %.3f %.3f %.3f %.3f]\n",
            T, mean, stdv, mel32T[0],mel32T[1],mel32T[2],mel32T[3],mel32T[4],mel32T[5]);
  }

  A()->ReleaseValue(out);
  return mel32T;
}

// 运行emb模型，输入NHWC(1,76,32,1)，输出(1,96)
static std::vector<float> run_emb_window(oww_handle* h, const float* mel_window) {
  OrtMemoryInfo* mi=nullptr; 
  oww_handle::ORTCHK(A()->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &mi));
  
  // 输入: NHWC [1, 76, 32, 1]
  OrtValue* in=nullptr;
  int64_t in_shape[4] = {1, 76, 32, 1};
  oww_handle::ORTCHK(A()->CreateTensorWithDataAsOrtValue(mi, (void*)mel_window, 76*32*sizeof(float),
                                                         in_shape, 4, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &in));
  A()->ReleaseMemoryInfo(mi);
  
  // 推理
  const char* in_names[]={h->ort.emb_in0.c_str()}; 
  const char* out_names[]={h->ort.emb_out0.c_str()};
  OrtValue* out=nullptr; 
  oww_handle::ORTCHK(A()->Run(h->ort.emb, nullptr, in_names, (const OrtValue* const*)&in, 1, out_names, 1, &out));
  A()->ReleaseValue(in);
  
  // 读取输出
  float* emb_data=nullptr; 
  oww_handle::ORTCHK(A()->GetTensorMutableData(out, (void**)&emb_data));
  
  // 复制结果
  std::vector<float> result(96);
  memcpy(result.data(), emb_data, 96 * sizeof(float));
  
  A()->ReleaseValue(out);
  return result;
}

// 三链检测：mel -> emb -> cls
static int try_detect_three_chain(oww_handle* h){
  if (h->pcm_buf.size() < oww_handle::NEED_SAMPLES) {
    return 0; // PCM数据不足
  }
  
  // 1. 运行mel模型 - 使用完整的NEED_SAMPLES
  std::vector<float> pcm_data(oww_handle::NEED_SAMPLES);
  std::copy(h->pcm_buf.end() - oww_handle::NEED_SAMPLES, h->pcm_buf.end(), pcm_data.begin());
  std::vector<float> mel_data = run_mel(h, pcm_data.data(), pcm_data.size());
  
  // 调试：打印mel数据统计
  if (!mel_data.empty()) {
    float mel_mean = 0.0f, mel_std = 0.0f;
    for (float v : mel_data) mel_mean += v;
    mel_mean /= mel_data.size();
    for (float v : mel_data) mel_std += (v - mel_mean) * (v - mel_mean);
    mel_std = sqrtf(mel_std / mel_data.size());
    fprintf(stderr, "🔍 DEBUG mel统计: size=%zu, mean=%.6f, std=%.6f\n", 
           mel_data.size(), mel_mean, mel_std);
    fflush(stderr);
  }
  
  if (mel_data.empty()) {
    return 0;
  }
  
  int T = mel_data.size() / 32;
  int need_frames = h->mel_win * h->nwin;  // 76 * 16 = 1216
  
  fprintf(stderr, "🔍 DEBUG mel帧数: T=%d, need=%d, 音频样本=%zu\n", T, need_frames, oww_handle::NEED_SAMPLES);
  fflush(stderr);
  
  // 2. 裁剪/补齐到固定大小（改为左补零、右对齐）
  std::vector<float> aligned_mel(32 * need_frames, 0.0f);
  if (T < need_frames) {
    int off = need_frames - T;  // 在右侧对齐
    for (int m=0;m<32;++m)
      memcpy(aligned_mel.data() + m*need_frames + off,
             mel_data.data() + m*T,
             T*sizeof(float));
  } else if (T > need_frames) {
    int start = (T - need_frames) / 2;  // 仍然居中裁剪
    for (int m=0;m<32;++m)
      memcpy(aligned_mel.data() + m*need_frames,
             mel_data.data() + m*T + start,
             need_frames*sizeof(float));
  } else {
    aligned_mel = mel_data;
  }
  
  // 3. 逐窗运行emb模型
  std::vector<float> emb_features(h->nwin * 96);
  for (int i = 0; i < h->nwin; i++) {
    // 提取窗口 (32, 76) -> 转置为 (76, 32)
    std::vector<float> window(76 * 32);
    for (int t = 0; t < h->mel_win; t++) {
      for (int m = 0; m < 32; m++) {
        window[t * 32 + m] = aligned_mel[m * need_frames + i * h->mel_win + t];
      }
    }
    
    // 运行emb模型
    std::vector<float> emb_out = run_emb_window(h, window.data());
    memcpy(emb_features.data() + i * 96, emb_out.data(), 96 * sizeof(float));
  }
  
  // 调试：打印emb特征统计
  float emb_mean = 0.0f, emb_std = 0.0f;
  for (float v : emb_features) emb_mean += v;
  emb_mean /= emb_features.size();
  for (float v : emb_features) emb_std += (v - emb_mean) * (v - emb_mean);
  emb_std = sqrtf(emb_std / emb_features.size());
  fprintf(stderr, "🔍 DEBUG emb统计: size=%zu, mean=%.6f, std=%.6f, 前6值=[%.3f,%.3f,%.3f,%.3f,%.3f,%.3f]\n", 
         emb_features.size(), emb_mean, emb_std,
         emb_features[0], emb_features[1], emb_features[2], 
         emb_features[3], emb_features[4], emb_features[5]);
  fflush(stderr);
  
  // 4. 运行cls模型
  OrtMemoryInfo* mi=nullptr; 
  oww_handle::ORTCHK(A()->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &mi));
  
  OrtValue* in=nullptr;
  int64_t shape[3] = {1, h->nwin, 96};
  oww_handle::ORTCHK(A()->CreateTensorWithDataAsOrtValue(mi, emb_features.data(), emb_features.size()*sizeof(float),
                                                         shape, 3, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &in));
  A()->ReleaseMemoryInfo(mi);
  
  const char* in_names[]={h->ort.cls_in0.c_str()}; 
  const char* out_names[]={h->ort.cls_out0.c_str()};
  OrtValue* out=nullptr; 
  oww_handle::ORTCHK(A()->Run(h->ort.cls, nullptr, in_names, (const OrtValue* const*)&in, 1, out_names, 1, &out));
  A()->ReleaseValue(in);
  
  // 读取结果
  float* logit_ptr=nullptr; 
  oww_handle::ORTCHK(A()->GetTensorMutableData(out, (void**)&logit_ptr));
  float logit = logit_ptr[0];
  
  // 计算概率
  float clamped_logit = fmaxf(-40.0f, fminf(40.0f, logit));
  float exp_val = expf(-clamped_logit);
  h->last = 1.0f / (1.0f + exp_val);
  
  fprintf(stderr, "🔍 DEBUG 概率计算: 原始logit=%.6f, clamp后=%.6f, exp(-clamp)=%.6e, prob=%.12f\n", 
         logit, clamped_logit, exp_val, h->last);
  fflush(stderr);
  
  A()->ReleaseValue(out);
  
  fprintf(stderr, "🔍 三链唤醒检测: logit=%.6f, prob=%.12f, 阈值=%.6f, 结果=%s\n", 
         logit, h->last, h->threshold, (h->last >= h->threshold) ? "触发" : "未触发");
  fflush(stderr);
  
  return (h->last >= h->threshold) ? 1 : 0;
}


// 三链模式的oww_process_i16函数实现
int oww_process_i16(oww_handle* h, const short* pcm, size_t samples) {
  if (!h || !pcm || samples == 0) return 0;
  
  // 将int16 PCM转换为float并添加到缓冲区
  for (size_t i = 0; i < samples; i++) {
    h->pcm_buf.push_back(pcm[i] / 32768.0f);
  }
  
  // 保持缓冲区大小 - 合理的滑动窗口
  while (h->pcm_buf.size() > oww_handle::NEED_SAMPLES + 3200) {  // 额外0.2秒缓冲
    h->pcm_buf.pop_front();
  }
  
  // 调试：每10次打印一次缓冲区状态
  static int debug_counter = 0;
  if (++debug_counter % 10 == 0) {
    fprintf(stderr, "🔍 三链缓冲区状态: %zu/%d 样本 (%.1f%%)\n", 
           h->pcm_buf.size(), oww_handle::NEED_SAMPLES, 
           100.0f * h->pcm_buf.size() / oww_handle::NEED_SAMPLES);
    fflush(stderr);
  }
  
  // 如果缓冲区足够大，尝试检测
  if (h->pcm_buf.size() >= oww_handle::NEED_SAMPLES) {
    return try_detect_three_chain(h);
  }
  
  return 0;
}