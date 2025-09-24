#include "oww.h"
#include <onnxruntime/core/session/onnxruntime_c_api.h>
#include <vector>
#include <deque>
#include <string>
#include <cstring>
#include <algorithm>
#include <stdexcept>
#include <cstdio>
#include <unistd.h>
#include <syslog.h>

// 调试宏 - 已禁用频繁日志
#define DEBUG_PRINTF(fmt, ...) do { \
    /* 调试日志已禁用，仅在必要时启用 */ \
} while(0)

static const OrtApi* A() { return OrtGetApiBase()->GetApi(ORT_API_VERSION); }

struct OwwOrt {
  OrtEnv* env=nullptr;
  OrtSessionOptions* so=nullptr;
  OrtSession* mels=nullptr;
  OrtSession* embed=nullptr;
  OrtSession* det=nullptr;
  OrtAllocator* alloc=nullptr;
  std::string mels_in0;
  std::string embed_in0;
  std::string det_in0;
  std::string mels_out0;
  std::string embed_out0;
  std::string det_out0;
  ~OwwOrt(){
    if(det)   A()->ReleaseSession(det);
    if(embed) A()->ReleaseSession(embed);
    if(mels)  A()->ReleaseSession(mels);
    if(so)    A()->ReleaseSessionOptions(so);
    if(env)   A()->ReleaseEnv(env);
  }
};

struct oww_handle {
  OwwOrt ort;
  // 形状参数（固定值）
  int mel_win=97, mel_bins=32;     // embed 输入 [1, mel_win, mel_bins, 1]
  int det_T=41, det_D=96;          // detector 输入 [1, det_T, det_D]

  bool two_chain_mode=false;       // 是否使用两链模式（mel+detector）
  float threshold=0.5f;
  float last=0.0f;

  // 缓冲
  std::deque<float> pcm_buf;       // 原始 PCM float
  std::deque<float> mel_buf;       // 按帧 push 的 mel；连续存储为 (frame, mel_bins)
  std::deque<float> emb_buf;       // 每次 96 维

  // 工具
  static void ORTCHK(OrtStatus* st){ if(st){ const char* m=A()->GetErrorMessage(st); std::string s=m?m:"ORT error"; A()->ReleaseStatus(st); throw std::runtime_error(s);} }
};

static void get_embed_shape(oww_handle* h){
  // 直接使用固定值，避免内存管理问题
  h->mel_win = 97;
  h->mel_bins = 32;
  printf("🔍 使用固定embed形状: mel_win=%d, mel_bins=%d\n", h->mel_win, h->mel_bins);
}

static void get_det_shape(oww_handle* h){
  // 直接使用固定值，避免内存管理问题
  h->det_T = 41;
  h->det_D = 96;
  printf("🔍 使用固定detector形状: det_T=%d, det_D=%d\n", h->det_T, h->det_D);
}

static OrtSession* load_session(OrtEnv* env, OrtSessionOptions* so, const char* path){
  if (!path || strlen(path) == 0) {
    throw std::runtime_error("Model path is empty");
  }
  printf("🔍 正在加载ONNX模型: %s\n", path);
  OrtSession* s=nullptr; 
  OrtStatus* status = A()->CreateSession(env, path, so, &s);
  if (status != nullptr) {
    const char* error_msg = A()->GetErrorMessage(status);
    printf("❌ CreateSession失败: %s\n", error_msg);
    A()->ReleaseStatus(status);
    throw std::runtime_error("CreateSession failed");
  }
  printf("✅ 模型加载成功: %s\n", path);
  return s;
}

static std::string ort_get_input_name(oww_handle* h, OrtSession* sess, size_t index){
  if(!h->ort.alloc){
    throw std::runtime_error("ORT allocator not initialized");
  }

  size_t count=0;
  oww_handle::ORTCHK(A()->SessionGetInputCount(sess, &count));
  if(index >= count){
    throw std::out_of_range("ORT input index out of range");
  }

  char* tmp=nullptr;
  oww_handle::ORTCHK(A()->SessionGetInputName(sess, index, h->ort.alloc, &tmp));

  if(!tmp || tmp[0] == '\0'){
    if(tmp) h->ort.alloc->Free(h->ort.alloc, tmp);
    throw std::runtime_error("DEBUG: ORT input name cannot be empty - this is from our modified oww library");
  }

  std::string name(tmp);
  h->ort.alloc->Free(h->ort.alloc, tmp);
  return name;
}

static std::string ort_get_output_name(oww_handle* h, OrtSession* sess, size_t index){
  if(!h->ort.alloc){
    throw std::runtime_error("ORT allocator not initialized");
  }

  size_t count=0;
  oww_handle::ORTCHK(A()->SessionGetOutputCount(sess, &count));
  if(index >= count){
    throw std::out_of_range("ORT output index out of range");
  }

  char* tmp=nullptr;
  oww_handle::ORTCHK(A()->SessionGetOutputName(sess, index, h->ort.alloc, &tmp));

  if(!tmp || tmp[0] == '\0'){
    if(tmp) h->ort.alloc->Free(h->ort.alloc, tmp);
    throw std::runtime_error("DEBUG: ORT output name cannot be empty - this is from our modified oww library");
  }

  std::string name(tmp);
  h->ort.alloc->Free(h->ort.alloc, tmp);
  return name;
}

oww_handle* oww_create(const char* melspec_onnx,
                       const char* embed_onnx,
                       const char* detector_onnx,
                       int threads,
                       float threshold){
  printf("⏰ [2025-09-24 22:15:00] OWW库版本确认：修复为两链架构！\n");
  printf("🔍 开始创建oww_handle...\n");
  
  auto h = new oww_handle();
  printf("✅ oww_handle创建成功\n");
  
  // ORT init
  printf("🔍 初始化ONNX Runtime环境...\n");
  oww_handle::ORTCHK(A()->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "oww", &h->ort.env));
  printf("✅ ONNX Runtime环境创建成功\n");
  
  printf("🔍 创建Session选项...\n");
  oww_handle::ORTCHK(A()->CreateSessionOptions(&h->ort.so));
  printf("✅ Session选项创建成功\n");
  
  printf("🔍 设置线程数: %d\n", threads);
  oww_handle::ORTCHK(A()->SetIntraOpNumThreads(h->ort.so, threads));
  printf("✅ 线程数设置成功\n");
  
#if ORT_API_VERSION >= 12
  printf("🔍 设置图优化级别...\n");
  oww_handle::ORTCHK(A()->SetSessionGraphOptimizationLevel(h->ort.so, ORT_ENABLE_BASIC));
  printf("✅ 图优化级别设置成功\n");
#endif
  
  printf("🔍 获取默认分配器...\n");
  oww_handle::ORTCHK(A()->GetAllocatorWithDefaultOptions(&h->ort.alloc));
  printf("✅ 默认分配器获取成功\n");

  // 检查embed_onnx是否为空，决定是两链还是三链
  bool use_two_chain = (!embed_onnx || strlen(embed_onnx) == 0);
  printf("🔍 检测到架构模式: %s\n", use_two_chain ? "两链(MEL+DETECTOR)" : "三链(MEL+EMBED+DETECTOR)");

  printf("🔍 准备加载MEL模型...\n");
  h->ort.mels = load_session(h->ort.env, h->ort.so, melspec_onnx);
  
  printf("🔍 准备加载DETECTOR模型...\n");
  h->ort.det = load_session(h->ort.env, h->ort.so, detector_onnx);

  if (use_two_chain) {
    printf("🔍 使用两链架构，跳过EMBED模型\n");
    h->ort.embed = nullptr;
    h->two_chain_mode = true;
  } else {
    printf("🔍 准备加载EMBED模型...\n");
    h->ort.embed = load_session(h->ort.env, h->ort.so, embed_onnx);
    h->two_chain_mode = false;
  }

  printf("🔍 获取输入输出名称...\n");
  h->ort.mels_in0 = ort_get_input_name(h, h->ort.mels, 0);
  h->ort.mels_out0 = ort_get_output_name(h, h->ort.mels, 0);
  h->ort.det_in0 = ort_get_input_name(h, h->ort.det, 0);
  h->ort.det_out0 = ort_get_output_name(h, h->ort.det, 0);

  if (!use_two_chain) {
    h->ort.embed_in0 = ort_get_input_name(h, h->ort.embed, 0);
    h->ort.embed_out0 = ort_get_output_name(h, h->ort.embed, 0);
  }

  printf("✅ 输入输出名称获取完成\n");
  printf("   MEL: %s -> %s\n", h->ort.mels_in0.c_str(), h->ort.mels_out0.c_str());
  if (!use_two_chain) {
    printf("   EMBED: %s -> %s\n", h->ort.embed_in0.c_str(), h->ort.embed_out0.c_str());
  }
  printf("   DETECTOR: %s -> %s\n", h->ort.det_in0.c_str(), h->ort.det_out0.c_str());

  h->threshold = threshold;
  printf("✅ oww_create完成，阈值: %.3f，模式: %s\n", threshold, use_two_chain ? "两链" : "三链");
  return h;
}

void oww_reset(oww_handle* h){
  h->pcm_buf.clear(); h->mel_buf.clear(); h->emb_buf.clear(); h->last=0.0f;
}

float oww_last_score(const oww_handle* h){ return h->last; }
size_t oww_recommended_chunk(){ return 1280; } // ~80ms@16k

// 运行一个 session，输入名统一用 "input"
static OrtValue* make_tensor_f32(const float* data, size_t count){
  OrtMemoryInfo* mi=nullptr; oww_handle::ORTCHK(A()->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &mi));
  OrtValue* v=nullptr;
  int64_t shape[2] = { 1, (int64_t)count };
  oww_handle::ORTCHK(A()->CreateTensorWithDataAsOrtValue(mi, (void*)data, count*sizeof(float),
                                                         shape, 2, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &v));
  A()->ReleaseMemoryInfo(mi);
  return v;
}

// 调 melspectrogram.onnx -> 输出形状推断并做 (x/10+2) 归一化
static void run_mels(oww_handle* h, const float* chunk, size_t n){
  printf("🔍 MEL处理开始: 输入样本数=%zu\n", n);
  
  OrtValue* in = make_tensor_f32(chunk, n);
  const char* in_names[]  = {h->ort.mels_in0.c_str()};
  const char* out_names[] = {h->ort.mels_out0.c_str()};
  OrtValue* out=nullptr;
  oww_handle::ORTCHK(A()->Run(h->ort.mels, nullptr, in_names, (const OrtValue* const*)&in, 1, out_names, 1, &out));
  A()->ReleaseValue(in);

  // 读输出
  OrtTensorTypeAndShapeInfo* tsh=nullptr;
  oww_handle::ORTCHK(A()->GetTensorTypeAndShape(out, &tsh));
  size_t dimN=0; oww_handle::ORTCHK(A()->GetDimensionsCount(tsh, &dimN));
  std::vector<int64_t> dims(dimN); oww_handle::ORTCHK(A()->GetDimensions(tsh, dims.data(), dimN));
  A()->ReleaseTensorTypeAndShapeInfo(tsh);

  printf("🔍 MEL输出维度: [");
  for(size_t i=0; i<dimN; ++i) {
    printf("%lld", dims[i]);
    if(i<dimN-1) printf(", ");
  }
  printf("]\n");

  float* p=nullptr; oww_handle::ORTCHK(A()->GetTensorMutableData(out, (void**)&p));
  size_t total = 1; for(auto d:dims) total *= (size_t)(d>0?d:1);

  // 约定输出形如 [1, frames, mel_bins] 或 [frames, mel_bins]
  int frames = (int)(total / std::max(1, h->mel_bins));
  printf("🔍 MEL计算: total=%zu, mel_bins=%d, frames=%d\n", total, h->mel_bins, frames);
  
  // 检查前几个值
  printf("🔍 MEL原始值前5个: ");
  for(int i=0; i<std::min(5, (int)total); ++i) {
    printf("%.3f ", p[i]);
  }
  printf("\n");
  
  for(int f=0; f<frames; ++f){
    for(int b=0; b<h->mel_bins; ++b){
      float v = p[f*h->mel_bins + b];
      // 按 OWW 预处理缩放：value/10 + 2
      h->mel_buf.push_back(v/10.0f + 2.0f);
    }
  }
  
  printf("🔍 MEL处理完成: 新增%d帧, mel_buf总大小=%zu\n", frames, h->mel_buf.size());
  A()->ReleaseValue(out);
}

// 从 mel_buf 尽可能提取嵌入（滑窗步长=按新进帧数）
static void try_make_embeddings(oww_handle* h, int newly_added_frames){
  if(newly_added_frames<=0) return;
  // 能否形成 >= mel_win 帧的窗口
  int frames = (int)h->mel_buf.size() / h->mel_bins;
  int can_emit = std::max(0, frames - h->mel_win + 1);
  int emit = std::min(can_emit, newly_added_frames); // 每进多少帧就前进多少步

  printf("🔍 Embedding处理: newly_added_frames=%d, total_frames=%d, can_emit=%d, emit=%d\n", 
         newly_added_frames, frames, can_emit, emit);
  printf("🔍 Embedding参数: mel_win=%d, mel_bins=%d, mel_buf_size=%zu\n", 
         h->mel_win, h->mel_bins, h->mel_buf.size());

  for(int e=0; e<emit; ++e){
    // 取最后 mel_win 帧里倒数第(emit-e)个窗口
    int start_frame = frames - h->mel_win - (emit-1-e);
    if(start_frame < 0) continue;
    
    printf("🔍 Embedding窗口%d: start_frame=%d, mel_win=%d\n", e, start_frame, h->mel_win);
    
    // 组装 [1, mel_win, mel_bins, 1]
    std::vector<float> win; win.reserve(h->mel_win*h->mel_bins);
    for(int f=0; f<h->mel_win; ++f){
      int idx = (start_frame+f)*h->mel_bins;
      for(int b=0; b<h->mel_bins; ++b) win.push_back(h->mel_buf[idx+b]);
    }
    
    printf("🔍 Embedding输入: 窗口大小=%zu, 预期大小=%d\n", win.size(), h->mel_win*h->mel_bins);
    
    // 构造 OrtValue（直接用数据拷贝到新 tensor）
    OrtMemoryInfo* mi=nullptr; oww_handle::ORTCHK(A()->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &mi));
    OrtValue* in=nullptr;
    int64_t shape[4] = {1, h->mel_win, h->mel_bins, 1};
    oww_handle::ORTCHK(A()->CreateTensorWithDataAsOrtValue(mi, win.data(), win.size()*sizeof(float),
                                                           shape, 4, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &in));
    A()->ReleaseMemoryInfo(mi);

    const char* in_names[]={h->ort.embed_in0.c_str()}; const char* out_names[]={h->ort.embed_out0.c_str()};
    OrtValue* out=nullptr; oww_handle::ORTCHK(A()->Run(h->ort.embed, nullptr, in_names, (const OrtValue* const*)&in, 1, out_names, 1, &out));
    A()->ReleaseValue(in);

    // 读取embedding输出维度
    OrtTensorTypeAndShapeInfo* tsh=nullptr;
    oww_handle::ORTCHK(A()->GetTensorTypeAndShape(out, &tsh));
    size_t dimN=0; oww_handle::ORTCHK(A()->GetDimensionsCount(tsh, &dimN));
    std::vector<int64_t> dims(dimN); oww_handle::ORTCHK(A()->GetDimensions(tsh, dims.data(), dimN));
    A()->ReleaseTensorTypeAndShapeInfo(tsh);

    printf("🔍 Embedding输出维度: [");
    for(size_t i=0; i<dimN; ++i) {
      printf("%lld", dims[i]);
      if(i<dimN-1) printf(", ");
    }
    printf("]\n");

    float* p=nullptr; oww_handle::ORTCHK(A()->GetTensorMutableData(out, (void**)&p));
    size_t total_out = 1; for(auto d:dims) total_out *= (size_t)(d>0?d:1);
    
    printf("🔍 Embedding输出: total_size=%zu, det_D=%d\n", total_out, h->det_D);
    printf("🔍 Embedding前5个值: ");
    for(int i=0; i<std::min(5, (int)total_out); ++i) {
      printf("%.6f ", p[i]);
    }
    printf("\n");
    
    // 期望输出 96 维
    for(int i=0;i<h->det_D;i++) h->emb_buf.push_back(p[i]);
    A()->ReleaseValue(out);
  }
  // 控制 mel_buf 大小：只保留最近 mel_win+64 帧，避免无限增长
  int keep_frames = h->mel_win + 64;
  int cur_frames  = (int)h->mel_buf.size() / h->mel_bins;
  if(cur_frames > keep_frames){
    int drop = (cur_frames - keep_frames)*h->mel_bins;
    h->mel_buf.erase(h->mel_buf.begin(), h->mel_buf.begin()+drop);
  }
  // 控制 emb_buf 大小
  int keep_emb = h->det_T + 64;
  int cur_emb  = (int)h->emb_buf.size() / h->det_D;
  if(cur_emb > keep_emb){
    int drop = (cur_emb - keep_emb)*h->det_D;
    h->emb_buf.erase(h->emb_buf.begin(), h->emb_buf.begin()+drop);
  }
}

// 两链模式：直接从mel特征到detector
static int try_detect_two_chain(oww_handle* h){
  int mel_frames = (int)h->mel_buf.size() / h->mel_bins;
  printf("🔍 TwoChain Detector处理: mel_frames=%d, 需要frames>=36\n", mel_frames);
  
  if(mel_frames < 36) {
    printf("🔍 TwoChain: mel帧数不足(%d < 36)，跳过检测\n", mel_frames);
    return 0;
  }

  // 参考oww_simple.py: mel4: [1,1,96,32] -> [1,36,96]
  // 我们的mel_buf是按(frame, mel_bins)存储，即每帧32维
  
  // 取最后36帧: mel_buf[(mel_frames-36)*32 : mel_frames*32]
  std::vector<float> mel_36_frames;
  mel_36_frames.reserve(36 * 32);
  
  int start_frame = mel_frames - 36;
  for(int f = 0; f < 36; f++) {
    int frame_idx = start_frame + f;
    int base = frame_idx * h->mel_bins;
    for(int b = 0; b < h->mel_bins; b++) {
      mel_36_frames.push_back(h->mel_buf[base + b]);
    }
  }
  
  // 转换为[1,36,96]: 每帧32维重复3次变成96维
  std::vector<float> detector_input;
  detector_input.reserve(36 * 96);
  
  for(int f = 0; f < 36; f++) {
    for(int repeat = 0; repeat < 3; repeat++) {
      for(int b = 0; b < 32; b++) {
        detector_input.push_back(mel_36_frames[f * 32 + b]);
      }
    }
  }
  
  printf("🔍 TwoChain输入: 形状[1,36,96], 数据大小=%zu\n", detector_input.size());
  printf("🔍 TwoChain输入前5个值: ");
  for(int i=0; i<std::min(5, (int)detector_input.size()); ++i) {
    printf("%.6f ", detector_input[i]);
  }
  printf("\n");

  // 执行推理
  OrtMemoryInfo* mi=nullptr; oww_handle::ORTCHK(A()->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &mi));
  OrtValue* in=nullptr;
  int64_t shape[3] = {1, 36, 96};
  oww_handle::ORTCHK(A()->CreateTensorWithDataAsOrtValue(mi, detector_input.data(), detector_input.size()*sizeof(float),
                                                         shape, 3, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &in));
  A()->ReleaseMemoryInfo(mi);
  
  const char* in_names[]={h->ort.det_in0.c_str()}; 
  const char* out_names[]={h->ort.det_out0.c_str()};
  OrtValue* out=nullptr; 
  oww_handle::ORTCHK(A()->Run(h->ort.det, nullptr, in_names, (const OrtValue* const*)&in, 1, out_names, 1, &out));
  A()->ReleaseValue(in);

  // 读取输出
  float* p=nullptr; oww_handle::ORTCHK(A()->GetTensorMutableData(out, (void**)&p));
  h->last = p[0];
  
  printf("🔍 TwoChain最终输出: score=%.6f, 阈值=%.3f, 结果=%s\n", 
         h->last, h->threshold, (h->last >= h->threshold) ? "触发" : "未触发");
  
  A()->ReleaseValue(out);
  return (h->last >= h->threshold) ? 1 : 0;
}

static int try_detect(oww_handle* h){
  int emb_n = (int)h->emb_buf.size() / h->det_D;
  printf("🔍 Detector处理: emb_n=%d, det_T=%d, det_D=%d, emb_buf_size=%zu\n", 
         emb_n, h->det_T, h->det_D, h->emb_buf.size());
  
  if(emb_n < h->det_T) {
    printf("🔍 Detector: 嵌入数量不足(%d < %d)，跳过检测\n", emb_n, h->det_T);
    return 0;
  }

  // 取最后 det_T 个嵌入 => [1, det_T, det_D]
  std::vector<float> x; x.reserve(h->det_T*h->det_D);
  for(int t=emb_n - h->det_T; t<emb_n; ++t){
    int base = t*h->det_D;
    for(int d=0; d<h->det_D; ++d) x.push_back(h->emb_buf[base+d]);
  }
  
  printf("🔍 Detector输入: 形状[1,%d,%d], 数据大小=%zu\n", h->det_T, h->det_D, x.size());
  printf("🔍 Detector输入前5个值: ");
  for(int i=0; i<std::min(5, (int)x.size()); ++i) {
    printf("%.6f ", x[i]);
  }
  printf("\n");

  OrtMemoryInfo* mi=nullptr; oww_handle::ORTCHK(A()->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &mi));
  OrtValue* in=nullptr;
  int64_t shape[3] = {1, h->det_T, h->det_D};
  oww_handle::ORTCHK(A()->CreateTensorWithDataAsOrtValue(mi, x.data(), x.size()*sizeof(float),
                                                         shape, 3, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &in));
  A()->ReleaseMemoryInfo(mi);
  const char* in_names[]={h->ort.det_in0.c_str()}; const char* out_names[]={h->ort.det_out0.c_str()};
  OrtValue* out=nullptr; oww_handle::ORTCHK(A()->Run(h->ort.det, nullptr, in_names, (const OrtValue* const*)&in, 1, out_names, 1, &out));
  A()->ReleaseValue(in);

  // 读取detector输出维度
  OrtTensorTypeAndShapeInfo* tsh=nullptr;
  oww_handle::ORTCHK(A()->GetTensorTypeAndShape(out, &tsh));
  size_t dimN=0; oww_handle::ORTCHK(A()->GetDimensionsCount(tsh, &dimN));
  std::vector<int64_t> dims(dimN); oww_handle::ORTCHK(A()->GetDimensions(tsh, dims.data(), dimN));
  A()->ReleaseTensorTypeAndShapeInfo(tsh);

  printf("🔍 Detector输出维度: [");
  for(size_t i=0; i<dimN; ++i) {
    printf("%lld", dims[i]);
    if(i<dimN-1) printf(", ");
  }
  printf("]\n");

  float* p=nullptr; oww_handle::ORTCHK(A()->GetTensorMutableData(out, (void**)&p));
  h->last = p[0];
  
  printf("🔍 Detector最终输出: score=%.6f, 阈值=%.3f, 结果=%s\n", 
         h->last, h->threshold, (h->last >= h->threshold) ? "触发" : "未触发");
  
  A()->ReleaseValue(out);
  return (h->last >= h->threshold) ? 1 : 0;
}

static int feed_pcm(oww_handle* h, const float* pcm, size_t samples){
  // 以 1280 样本为一块喂入 melspec；其输出有多少帧我们就推进多少
  const size_t step = oww_recommended_chunk();
  size_t off=0, fired=0;
  while(off < samples){
    size_t n = std::min(step, samples-off);
    run_mels(h, pcm+off, n);
    
    if (h->two_chain_mode) {
      // 两链模式：mel → detector
      fired |= try_detect_two_chain(h);
    } else {
      // 三链模式：mel → embedding → detector
      int guess_new_frames = (int)(n / 256); if(guess_new_frames < 1) guess_new_frames = 1;
      try_make_embeddings(h, guess_new_frames);
      fired |= try_detect(h);
    }
    off += n;
  }
  return fired ? 1 : 0;
}

int oww_process_f32(oww_handle* h, const float* pcm, size_t samples){
  return feed_pcm(h, pcm, samples);
}

int oww_process_i16(oww_handle* h, const short* x, size_t samples){
  std::vector<float> f(samples);
  for(size_t i=0;i<samples;i++) f[i] = (float)x[i] / 32768.0f;
  int result = feed_pcm(h, f.data(), samples);
  return result;
}

void oww_destroy(oww_handle* h){ delete h; }
