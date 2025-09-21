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

// 强制刷新输出的调试宏 - 多重输出确保可见
#define DEBUG_PRINTF(fmt, ...) do { \
    printf("[OWW_DEBUG] " fmt "\n", ##__VA_ARGS__); \
    fprintf(stderr, "[OWW_DEBUG] " fmt "\n", ##__VA_ARGS__); \
    fflush(stdout); \
    fflush(stderr); \
    fsync(STDOUT_FILENO); \
    fsync(STDERR_FILENO); \
    /* 写入系统日志 */ \
    syslog(LOG_INFO, "[OWW_DEBUG] " fmt, ##__VA_ARGS__); \
    /* 写入文件 */ \
    FILE* f = fopen("/tmp/oww_debug.log", "a"); \
    if(f) { \
        fprintf(f, "[OWW_DEBUG] " fmt "\n", ##__VA_ARGS__); \
        fclose(f); \
    } \
} while(0)

static const OrtApi* A() { return OrtGetApiBase()->GetApi(ORT_API_VERSION); }

struct OwwOrt {
  OrtEnv* env=nullptr;
  OrtSessionOptions* so=nullptr;
  OrtSession* mels=nullptr;
  OrtSession* embed=nullptr;
  OrtSession* det=nullptr;
  OrtAllocator* alloc=nullptr;
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
  // 形状参数（运行时读出）
  int mel_win=76, mel_bins=32;     // embed 输入 [1, mel_win, mel_bins, 1]
  int det_T=16, det_D=96;          // detector 输入 [1, det_T, det_D]

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
  OrtTypeInfo* ti=nullptr; oww_handle::ORTCHK(A()->SessionGetInputTypeInfo(h->ort.embed, 0, &ti));
  const OrtTensorTypeAndShapeInfo* tsh=nullptr; oww_handle::ORTCHK(A()->CastTypeInfoToTensorInfo(ti, &tsh));
  size_t n=0; oww_handle::ORTCHK(A()->GetDimensionsCount(tsh, &n));
  std::vector<int64_t> d(n); oww_handle::ORTCHK(A()->GetDimensions(tsh, d.data(), n));
  A()->ReleaseTypeInfo(ti);
  // 期望 [1, mel_win, mel_bins, 1]
  h->mel_win  = (n>=2 && d[1]>0) ? (int)d[1] : 76;
  h->mel_bins = (n>=3 && d[2]>0) ? (int)d[2] : 32;
}

static void get_det_shape(oww_handle* h){
  OrtTypeInfo* ti=nullptr; oww_handle::ORTCHK(A()->SessionGetInputTypeInfo(h->ort.det, 0, &ti));
  const OrtTensorTypeAndShapeInfo* tsh=nullptr; oww_handle::ORTCHK(A()->CastTypeInfoToTensorInfo(ti, &tsh));
  size_t n=0; oww_handle::ORTCHK(A()->GetDimensionsCount(tsh, &n));
  std::vector<int64_t> d(n); oww_handle::ORTCHK(A()->GetDimensions(tsh, d.data(), n));
  A()->ReleaseTypeInfo(ti);
  // 期望 [1, det_T, det_D]
  h->det_T = (n>=2 && d[1]>0) ? (int)d[1] : 16;
  h->det_D = (n>=3 && d[2]>0) ? (int)d[2] : 96;
}

static OrtSession* load_session(OrtEnv* env, OrtSessionOptions* so, const char* path){
  OrtSession* s=nullptr; oww_handle::ORTCHK(A()->CreateSession(env, path, so, &s)); return s;
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
  // 故意崩溃测试 - 移动到ort_get_output_name调用处
  
  // 立即输出，确保函数被调用
  write(STDOUT_FILENO, "[OWW_CREATE_CALLED]\n", 20);
  write(STDERR_FILENO, "[OWW_CREATE_CALLED]\n", 20);
  
  DEBUG_PRINTF("=== oww_create START ===");
  DEBUG_PRINTF("MEL model: %s", melspec_onnx);
  DEBUG_PRINTF("EMBED model: %s", embed_onnx);
  DEBUG_PRINTF("DET model: %s", detector_onnx);
  DEBUG_PRINTF("Threads: %d, Threshold: %.3f", threads, threshold);
  
  auto h = new oww_handle();
  DEBUG_PRINTF("Handle created");
  
  // ORT init
  DEBUG_PRINTF("Creating ORT environment...");
  oww_handle::ORTCHK(A()->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "oww", &h->ort.env));
  DEBUG_PRINTF("Creating session options...");
  oww_handle::ORTCHK(A()->CreateSessionOptions(&h->ort.so));
  DEBUG_PRINTF("Setting intra-op threads to %d...", threads);
  oww_handle::ORTCHK(A()->SetIntraOpNumThreads(h->ort.so, threads));
#if ORT_API_VERSION >= 12
  DEBUG_PRINTF("Setting graph optimization level...");
  oww_handle::ORTCHK(A()->SetSessionGraphOptimizationLevel(h->ort.so, ORT_ENABLE_BASIC));
#endif
  DEBUG_PRINTF("Getting allocator...");
  oww_handle::ORTCHK(A()->GetAllocatorWithDefaultOptions(&h->ort.alloc));
  DEBUG_PRINTF("ORT initialization complete");

  // load three sessions
  DEBUG_PRINTF("Loading MEL session: %s", melspec_onnx);
  h->ort.mels  = load_session(h->ort.env, h->ort.so, melspec_onnx);
  DEBUG_PRINTF("MEL session loaded successfully");
  
  DEBUG_PRINTF("Loading EMBED session: %s", embed_onnx);
  h->ort.embed = load_session(h->ort.env, h->ort.so, embed_onnx);
  DEBUG_PRINTF("EMBED session loaded successfully");
  
  DEBUG_PRINTF("Loading DET session: %s", detector_onnx);
  h->ort.det   = load_session(h->ort.env, h->ort.so, detector_onnx);
  DEBUG_PRINTF("DET session loaded successfully");

  // 使用硬编码的输出名称避免 ONNX Runtime 版本兼容性问题
  DEBUG_PRINTF("Using hardcoded output names...");
  h->ort.mels_out0 = "output";
  h->ort.embed_out0 = "output";
  h->ort.det_out0 = "output";
  DEBUG_PRINTF("Output names set: MEL=%s, EMBED=%s, DET=%s", 
               h->ort.mels_out0.c_str(), h->ort.embed_out0.c_str(), h->ort.det_out0.c_str());

  DEBUG_PRINTF("Getting model shapes...");
  get_embed_shape(h);
  DEBUG_PRINTF("Embed shape: mel_win=%d, mel_bins=%d", h->mel_win, h->mel_bins);
  get_det_shape(h);
  DEBUG_PRINTF("Det shape: det_T=%d, det_D=%d", h->det_T, h->det_D);

  h->threshold = threshold;
  DEBUG_PRINTF("=== oww_create SUCCESS ===");
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
  DEBUG_PRINTF("run_mels: processing %zu samples", n);
  OrtValue* in = make_tensor_f32(chunk, n);
  const char* in_names[]  = {"input"};
  const char* out_names[] = {h->ort.mels_out0.c_str()};
  DEBUG_PRINTF("run_mels: calling MEL session with output name: %s", h->ort.mels_out0.c_str());
  OrtValue* out=nullptr;
  oww_handle::ORTCHK(A()->Run(h->ort.mels, nullptr, in_names, (const OrtValue* const*)&in, 1, out_names, 1, &out));
  A()->ReleaseValue(in);
  DEBUG_PRINTF("run_mels: MEL session run completed");

  // 读输出
  OrtTensorTypeAndShapeInfo* tsh=nullptr;
  oww_handle::ORTCHK(A()->GetTensorTypeAndShape(out, &tsh));
  size_t dimN=0; oww_handle::ORTCHK(A()->GetDimensionsCount(tsh, &dimN));
  std::vector<int64_t> dims(dimN); oww_handle::ORTCHK(A()->GetDimensions(tsh, dims.data(), dimN));
  A()->ReleaseTensorTypeAndShapeInfo(tsh);

  float* p=nullptr; oww_handle::ORTCHK(A()->GetTensorMutableData(out, (void**)&p));
  size_t total = 1; for(auto d:dims) total *= (size_t)(d>0?d:1);

  // 约定输出形如 [1, frames, mel_bins] 或 [frames, mel_bins]
  int frames = (int)(total / std::max(1, h->mel_bins));
  DEBUG_PRINTF("run_mels: output shape total=%zu, frames=%d, mel_bins=%d", total, frames, h->mel_bins);
  for(int f=0; f<frames; ++f){
    for(int b=0; b<h->mel_bins; ++b){
      float v = p[f*h->mel_bins + b];
      // 按 OWW 预处理缩放：value/10 + 2
      h->mel_buf.push_back(v/10.0f + 2.0f);
    }
  }
  A()->ReleaseValue(out);
  DEBUG_PRINTF("run_mels: completed, added %d frames", frames);
}

// 从 mel_buf 尽可能提取嵌入（滑窗步长=按新进帧数）
static void try_make_embeddings(oww_handle* h, int newly_added_frames){
  if(newly_added_frames<=0) return;
  // 能否形成 >= mel_win 帧的窗口
  int frames = (int)h->mel_buf.size() / h->mel_bins;
  int can_emit = std::max(0, frames - h->mel_win + 1);
  int emit = std::min(can_emit, newly_added_frames); // 每进多少帧就前进多少步

  for(int e=0; e<emit; ++e){
    // 取最后 mel_win 帧里倒数第(emit-e)个窗口
    int start_frame = frames - h->mel_win - (emit-1-e);
    if(start_frame < 0) continue;
    // 组装 [1, mel_win, mel_bins, 1]
    std::vector<float> win; win.reserve(h->mel_win*h->mel_bins);
    for(int f=0; f<h->mel_win; ++f){
      int idx = (start_frame+f)*h->mel_bins;
      for(int b=0; b<h->mel_bins; ++b) win.push_back(h->mel_buf[idx+b]);
    }
    // 构造 OrtValue（直接用数据拷贝到新 tensor）
    OrtMemoryInfo* mi=nullptr; oww_handle::ORTCHK(A()->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &mi));
    OrtValue* in=nullptr;
    int64_t shape[4] = {1, h->mel_win, h->mel_bins, 1};
    oww_handle::ORTCHK(A()->CreateTensorWithDataAsOrtValue(mi, win.data(), win.size()*sizeof(float),
                                                           shape, 4, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &in));
    A()->ReleaseMemoryInfo(mi);

    const char* in_names[]={"input"}; const char* out_names[]={h->ort.embed_out0.c_str()};
    OrtValue* out=nullptr; oww_handle::ORTCHK(A()->Run(h->ort.embed, nullptr, in_names, (const OrtValue* const*)&in, 1, out_names, 1, &out));
    A()->ReleaseValue(in);

    float* p=nullptr; oww_handle::ORTCHK(A()->GetTensorMutableData(out, (void**)&p));
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

static int try_detect(oww_handle* h){
  int emb_n = (int)h->emb_buf.size() / h->det_D;
  DEBUG_PRINTF("try_detect: emb_n=%d, det_T=%d, det_D=%d", emb_n, h->det_T, h->det_D);
  if(emb_n < h->det_T) return 0;

  // 取最后 det_T 个嵌入 => [1, det_T, det_D]
  std::vector<float> x; x.reserve(h->det_T*h->det_D);
  for(int t=emb_n - h->det_T; t<emb_n; ++t){
    int base = t*h->det_D;
    for(int d=0; d<h->det_D; ++d) x.push_back(h->emb_buf[base+d]);
  }
  DEBUG_PRINTF("try_detect: prepared input tensor with %zu elements", x.size());

  OrtMemoryInfo* mi=nullptr; oww_handle::ORTCHK(A()->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &mi));
  OrtValue* in=nullptr;
  int64_t shape[3] = {1, h->det_T, h->det_D};
  oww_handle::ORTCHK(A()->CreateTensorWithDataAsOrtValue(mi, x.data(), x.size()*sizeof(float),
                                                         shape, 3, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &in));
  A()->ReleaseMemoryInfo(mi);
  const char* in_names[]={"input"}; const char* out_names[]={h->ort.det_out0.c_str()};
  DEBUG_PRINTF("try_detect: calling DET session with output name: %s", h->ort.det_out0.c_str());
  OrtValue* out=nullptr; oww_handle::ORTCHK(A()->Run(h->ort.det, nullptr, in_names, (const OrtValue* const*)&in, 1, out_names, 1, &out));
  A()->ReleaseValue(in);
  DEBUG_PRINTF("try_detect: DET session run completed");

  float* p=nullptr; oww_handle::ORTCHK(A()->GetTensorMutableData(out, (void**)&p));
  h->last = p[0];
  A()->ReleaseValue(out);
  DEBUG_PRINTF("try_detect: score=%.4f, threshold=%.3f", h->last, h->threshold);
  return (h->last >= h->threshold) ? 1 : 0;
}

static int feed_pcm(oww_handle* h, const float* pcm, size_t samples){
  // 以 1280 样本为一块喂入 melspec；其输出有多少帧我们就推进多少
  const size_t step = oww_recommended_chunk();
  size_t off=0, fired=0;
  while(off < samples){
    size_t n = std::min(step, samples-off);
    run_mels(h, pcm+off, n);
    // 估算新加入了多少帧：按输出总 mel size 变化估计更复杂，这里近似为 n/256*? ——直接按 embed 生成结果来推进
    // 简化：每次按 mels 输出后尝试最大化生成嵌入
    // 由于我们不知道确切帧数，这里用一个保守上限：假定本次至少产生了 (int)(n/256) 帧
    int guess_new_frames = (int)(n / 256); if(guess_new_frames < 1) guess_new_frames = 1;
    try_make_embeddings(h, guess_new_frames);
    fired |= try_detect(h);
    off += n;
  }
  return fired ? 1 : 0;
}

int oww_process_f32(oww_handle* h, const float* pcm, size_t samples){
  return feed_pcm(h, pcm, samples);
}

int oww_process_i16(oww_handle* h, const short* x, size_t samples){
  DEBUG_PRINTF("oww_process_i16: processing %zu samples", samples);
  std::vector<float> f(samples);
  for(size_t i=0;i<samples;i++) f[i] = (float)x[i] / 32768.0f;
  DEBUG_PRINTF("oww_process_i16: converted to float, calling feed_pcm");
  int result = feed_pcm(h, f.data(), samples);
  DEBUG_PRINTF("oww_process_i16: feed_pcm returned %d", result);
  return result;
}

void oww_destroy(oww_handle* h){ delete h; }
