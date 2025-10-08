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
#include <chrono>

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
  int min_samples = 8000;     // 最小0.5秒@16kHz，快速说话检测
  int max_samples = 32000;    // 最大2.0秒@16kHz，保证完整性
  
  float threshold=0.5f;
  float last=0.0f;
  int consec_hits=0;
  int consec_required=2;
  
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

static inline int minimal_frames_gate(const oww_handle* h) {
  return std::max(16, h->mel_win / 2);
}

static inline size_t mel_window_samples(const oww_handle* h) {
  return static_cast<size_t>(h->mel_win) * oww_handle::HOP;
}

static inline size_t preferred_keep_samples(const oww_handle* h) {
  return std::max(mel_window_samples(h), static_cast<size_t>(h->min_samples));
}

static void trim_pcm_buffer(oww_handle* h) {
  const size_t keep = preferred_keep_samples(h);
  if (h->pcm_buf.size() <= keep) {
    return;
  }
  size_t drop = h->pcm_buf.size() - keep;
  while (drop-- > 0) {
    h->pcm_buf.pop_front();
  }
}

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
  h->consec_hits=0;
}

float oww_last_score(const oww_handle* h){ 
  return h->last; 
}

size_t oww_recommended_chunk(){ 
  return 1280; // ~80ms@16k
}

void oww_set_buffer_size(oww_handle* h, size_t min_samples, size_t max_samples) {
  if (h && min_samples > 0 && max_samples >= min_samples) {
    int new_min = static_cast<int>(min_samples);
    int new_max = static_cast<int>(max_samples);
    const int floor_samples = minimal_frames_gate(h) * oww_handle::HOP;
    if (new_min < floor_samples) {
      fprintf(stderr, "⚠️ 提供的最小缓冲不足以覆盖模型窗体，自动提升到%d样本\n", floor_samples);
      new_min = floor_samples;
    }
    if (new_max < new_min) {
      new_max = new_min;
    }
    h->min_samples = new_min;
    h->max_samples = new_max;
    fprintf(stderr, "🔧 设置oww缓冲区: 最小%d样本(%.3fs) 最大%d样本(%.3fs)\n", 
            h->min_samples, h->min_samples / 16000.0, h->max_samples, h->max_samples / 16000.0);
    fflush(stderr);
  }
}

void oww_destroy(oww_handle* h){
  delete h;
}

// 与ipynb完全一致的power→dB→[0,1]转换
static inline void power_to_db01(float* x, size_t n) {
  const float eps = 1e-10f;
  for (size_t i = 0; i < n; ++i) {
    float p  = fmaxf(x[i], eps);
    float db = 10.f * log10f(p);
    float y  = (db + 80.f) / 80.f;
    x[i] = y < 0.f ? 0.f : (y > 1.f ? 1.f : y);
  }
}

static inline void db_to_01(float* x, size_t n) {
  const float inv80 = 1.0f / 80.0f;
  for (size_t i = 0; i < n; ++i) {
    float y = (x[i] + 80.0f) * inv80;
    if (y < 0.0f) y = 0.0f;
    if (y > 1.0f) y = 1.0f;
    x[i] = y;
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
  fprintf(stderr, "🔍 DEBUG 进入run_mel函数: samples=%zu\n", samples);
  fflush(stderr);
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
  double mean=0, mn=1e30, mx=-1e30;
  {
    size_t N = mel32T.size();
    for (size_t i=0;i<N;++i){ mean+=mel32T[i]; mn=std::min<double>(mn,mel32T[i]); mx=std::max<double>(mx,mel32T[i]); }
    mean/=N;
    fprintf(stderr, "🔍 mel原始功率: shape=(32,%d) mean=%.6g min=%.6g max=%.6g\n", T, mean, mn, mx);
    fflush(stderr);
  }

  // 调试（归一化前）
  {
    double mean=0, stdv=0; size_t N = mel32T.size();
    for (size_t i=0;i<N;++i) mean += mel32T[i];
    mean /= N;
    for (size_t i=0;i<N;++i) stdv += (mel32T[i]-mean)*(mel32T[i]-mean);
    stdv = std::sqrt(stdv/N);
    fprintf(stderr, "🔍 mel原始输出: T=%d mean=%.6f std=%.6f first6=[%.3f %.3f %.3f %.3f %.3f %.3f]\n",
            T, mean, stdv, mel32T[0],mel32T[1],mel32T[2],mel32T[3],mel32T[4],mel32T[5]);
    fflush(stderr);
  }

  // ★ 修复：根据colab训练规格，总是执行power→dB→[0,1]归一化
  fprintf(stderr, "🔍 mel统一执行power→dB→[0,1]归一化（匹配训练规格）\n");
    fflush(stderr);
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
  // 动态调整数据量：使用当前缓冲区数据，最少接受min_samples
  size_t actual_samples = h->pcm_buf.size();
  if (actual_samples < h->min_samples) {
    return 0; // PCM数据不足
  }
  
  // 1. 运行mel模型 - 使用实际可用的数据量
  std::vector<float> pcm_data(actual_samples);
  std::copy(h->pcm_buf.end() - actual_samples, h->pcm_buf.end(), pcm_data.begin());
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
  
  const int mel_bins = h->mel_bins;
  int T = mel_data.size() / mel_bins;

  const int min_frames_gate = minimal_frames_gate(h);
  if (T < min_frames_gate) {
    fprintf(stderr, "🛑 DEBUG mel帧不足: T=%d < gate=%d\n", T, min_frames_gate);
    fflush(stderr);
    return 0;
  }

  if (T < h->mel_win) {
    fprintf(stderr, "ℹ️ DEBUG mel帧小于完整窗口: T=%d < mel_win=%d，将补零继续\n", T, h->mel_win);
    fflush(stderr);
  }

  // ★ 滑动窗口策略：扫描整个mel，取最大概率（不裁剪音频）
  const int hop = h->mel_win; // hop = 76
  const int window_frames = h->nwin * hop; // 16 × 76 = 1216帧
  
  // 计算滑动窗口数量
  int num_slides = 1;
  if (T >= window_frames) {
    // 每次滑动hop帧，确保扫描整个音频
    num_slides = (T - window_frames) / hop + 1;
  }
  
  fprintf(stderr,
          "🔍 DEBUG 滑动窗口: T=%d, window=%d, hop=%d, slides=%d, audio=%zu\n",
          T, window_frames, hop, num_slides, actual_samples);
  fflush(stderr);

  float max_prob = 0.0f;
  int best_slide = 0;

  // 3. 对每个滑动位置进行完整推理
  for (int slide = 0; slide < num_slides; slide++) {
    int slide_start = slide * hop;
    
    // 准备当前滑动窗口的mel数据（window_frames帧）
    std::vector<float> processed_mel_slide(mel_bins * window_frames, 0.0f);
    
    for (int m = 0; m < mel_bins; m++) {
      for (int t = 0; t < window_frames; t++) {
        int src_t = slide_start + t;
        if (src_t < T) {
          processed_mel_slide[m * window_frames + t] = mel_data[m * T + src_t];
        }
        // 否则保持0（自动补零）
      }
    }
    
    // 4. 逐窗运行emb模型（固定hop=76，无重叠连续窗口）
    std::vector<float> emb_features_slide(h->nwin * 96);
    std::vector<float> window(h->mel_win * mel_bins);

    for (int i = 0; i < h->nwin; i++) {
      const int win_start = i * hop;

      for (int t = 0; t < h->mel_win; t++) {
        const int src_t = win_start + t;
        const size_t dst_row = t * (size_t)mel_bins;
        for (int m = 0; m < mel_bins; m++) {
          window[dst_row + m] = processed_mel_slide[m * (size_t)window_frames + src_t];
        }
      }

      std::vector<float> emb_out = run_emb_window(h, window.data());
      memcpy(emb_features_slide.data() + i * 96, emb_out.data(), 96 * sizeof(float));
    }
    
    // 5. 运行cls模型（使用flatten输入 [1, 1536]）
    OrtMemoryInfo* mi_cls=nullptr; 
    oww_handle::ORTCHK(A()->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &mi_cls));
    
    OrtValue* in_cls=nullptr;
    int64_t shape_cls[2] = {1, (int64_t)(h->nwin * 96)};  // Flatten to [1, 1536]
    oww_handle::ORTCHK(A()->CreateTensorWithDataAsOrtValue(mi_cls, emb_features_slide.data(), emb_features_slide.size()*sizeof(float),
                                                           shape_cls, 2, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &in_cls));
    A()->ReleaseMemoryInfo(mi_cls);
    
    const char* in_names_cls[]={h->ort.cls_in0.c_str()}; 
    const char* out_names_cls[]={h->ort.cls_out0.c_str()};
    OrtValue* out_cls=nullptr; 
    oww_handle::ORTCHK(A()->Run(h->ort.cls, nullptr, in_names_cls, (const OrtValue* const*)&in_cls, 1, out_names_cls, 1, &out_cls));
    A()->ReleaseValue(in_cls);
    
    // 读取结果 - 模型输出已经是Sigmoid后的概率值
    float* prob_ptr_cls=nullptr; 
    oww_handle::ORTCHK(A()->GetTensorMutableData(out_cls, (void**)&prob_ptr_cls));
    float current_prob = fmaxf(0.0f, fminf(1.0f, prob_ptr_cls[0]));
    
    if (current_prob > max_prob) {
      max_prob = current_prob;
      best_slide = slide;
    }
    
    A()->ReleaseValue(out_cls);
  } // end for slide

  h->last = max_prob; // 使用最大概率
  fprintf(stderr, "🔍 DEBUG 最终概率 (滑动窗口): max_prob=%.12f (位置=%d)\n", max_prob, best_slide);
  fflush(stderr);
  
  bool triggered = false;
  if (h->last >= h->threshold) {
    h->consec_hits++;
  } else if (h->consec_hits > 0) {
    h->consec_hits = 0;
  }

  // ✅ 段检测模式优化：大批量输入时降低consec要求
  // 如果传入的PCM数据量超过1秒（16000样本），认为是段检测模式
  int effective_consec_required = h->consec_required;
  if (actual_samples >= 16000) {
    effective_consec_required = 1;  // 段检测模式只需要1次命中
    fprintf(stderr, "🎯 段检测模式：自动降低consec要求 %d→1 (输入%zu样本=%.2fs)\n", 
           h->consec_required, actual_samples, (double)actual_samples/16000.0);
    fflush(stderr);
  }

  const bool ready_to_trigger = h->consec_hits >= effective_consec_required;
  fprintf(stderr,
          "🔍 三链唤醒检测: prob=%.12f, 阈值=%.6f, consec=%d/%d, 结果=%s\n",
          h->last, h->threshold, h->consec_hits, effective_consec_required,
          ready_to_trigger ? "触发" : "未触发");
  fflush(stderr);

  if (ready_to_trigger) {
    h->consec_hits = 0;
    fprintf(stderr, "🔄 连续命中阈值，清空缓冲区\n");
    h->pcm_buf.clear();
    fflush(stderr);
    triggered = true;
  } else {
    trim_pcm_buffer(h);
  }

  return triggered ? 1 : 0;
}


// 三链模式的oww_process_i16函数实现
int oww_process_i16(oww_handle* h, const short* pcm, size_t samples) {
  static int call_count = 0;
  call_count++;
  
  // 获取当前时间戳（毫秒）
  auto now = std::chrono::duration_cast<std::chrono::milliseconds>(
    std::chrono::system_clock::now().time_since_epoch()).count();
  fprintf(stderr, "🔍 DEBUG oww_process_i16被调用#%d: samples=%zu, 时间=%ld\n", call_count, samples, now);
  fflush(stderr);
  
  if (!h || !pcm || samples == 0) return 0;
  
  // 将int16 PCM转换为float并添加到缓冲区
  for (size_t i = 0; i < samples; i++) {
    h->pcm_buf.push_back(pcm[i] / 32768.0f);
  }
  
  // 保持缓冲区大小 - 动态缓冲区策略
  while (h->pcm_buf.size() > h->max_samples) {  // 不超过最大缓冲区
    h->pcm_buf.pop_front();
  }
  
  // 调试：每10次打印一次缓冲区状态
  static int debug_counter = 0;
  if (++debug_counter % 10 == 0) {
    fprintf(stderr, "🔍 三链缓冲区状态: %zu/%d 样本 (%.1f%%)\n", 
           h->pcm_buf.size(), h->min_samples, 
           100.0f * h->pcm_buf.size() / h->min_samples);
  fflush(stderr);
  }
  
  // 调试缓冲区状态
  double buffer_seconds = (double)h->pcm_buf.size() / 16000.0;
  double min_seconds = (double)h->min_samples / 16000.0;
  fprintf(stderr, "🔍 缓冲区状态: %zu/%d样本 = %.3fs/%.3fs (最小检测条件)\n", 
          h->pcm_buf.size(), h->min_samples, buffer_seconds, min_seconds);
  fflush(stderr);
  
  // 动态检测：达到最小条件就可以检测
  if (h->pcm_buf.size() >= h->min_samples) {
    fprintf(stderr, "🔍 开始检测: 缓冲区=%zu样本 (最小%d)\n", h->pcm_buf.size(), h->min_samples);
    fflush(stderr);
    
    int result = try_detect_three_chain(h);
    fprintf(stderr, "🔍 检测结果: %d\n", result);
    fflush(stderr);
    
    return result;
  } else {
    fprintf(stderr, "🔍 缓冲区不足，需要最小%d，当前%zu\n", h->min_samples, h->pcm_buf.size());
    fflush(stderr);
  }
  
  return 0;
}
