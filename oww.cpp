#include "oww.hpp"
#include <vector>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <onnxruntime_c_api.h>

struct oww_handle {
  OrtEnv* env = nullptr;
  OrtSession* sess = nullptr;
  OrtMemoryInfo* mem = nullptr;
  std::vector<float> ring;   // 滑窗缓冲
  size_t ring_wpos = 0;
  bool filled = false;
  float threshold = 0.6f;
  int sr = 16000;
  size_t need_samples = 16000; // 将从模型输入维度推断
  std::mutex mu;
};

static void check(OrtStatus* st, const OrtApi* api){
  if (st){ const char* m = api->GetErrorMessage(st); throw std::runtime_error(m?m:"ORT error"); }
}

static size_t infer_required_samples(OrtSession* sess, const OrtApi* api){
  OrtAllocator* alloc = nullptr;
  check(api->GetAllocatorWithDefaultOptions(&alloc), api);
  size_t n_inputs = 0;
  check(api->SessionGetInputCount(sess, &n_inputs), api);
  if (n_inputs == 0) throw std::runtime_error("no inputs");
  OrtTypeInfo* ti = nullptr;
  check(api->SessionGetInputTypeInfo(sess, 0, &ti), api);
  const OrtTensorTypeAndShapeInfo* tsh = api->CastTypeInfoToTensorInfo(ti);
  if (!tsh) { api->ReleaseTypeInfo(ti); throw std::runtime_error("input is not tensor"); }
  ONNXTensorElementDataType dt = api->GetTensorElementType(tsh);
  if (dt != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT){
    api->ReleaseTypeInfo(ti);
    throw std::runtime_error("input dtype must be float32");
  }
  size_t dim_count = api->GetDimensionsCount(tsh);
  std::vector<int64_t> dims(dim_count);
  check(api->GetDimensions(tsh, dims.data(), dim_count), api);
  api->ReleaseTypeInfo(ti);

  // 常见几种：
  // 1) [1, N] 或 [N] 直接是样本数
  // 2) [1, T, 1] 视为T样本（部分模型内置前端）
  // 3) 如果是梅尔谱(形如 [1, M, T]) 则此封装不支持（需要先做特征）
  if (dim_count == 1) return (size_t)std::max<int64_t>(dims[0], 16000);
  if (dim_count == 2) return (size_t)std::max<int64_t>(dims[1], 16000);
  if (dim_count == 3 && dims[2] == 1) return (size_t)std::max<int64_t>(dims[1], 16000);

  throw std::runtime_error("model expects features (mel) — use a model with built-in frontend");
}

oww_handle* oww_create(const char* model_path, int sample_rate,
                       float window_seconds, float threshold){
  const OrtApi* api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
  auto* h = new (std::nothrow) oww_handle();
  if(!h) return nullptr;
  try{
    h->threshold = threshold;
    h->sr = sample_rate;

    check(api->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "oww", &h->env), api);
    OrtSessionOptions* so = nullptr;
    check(api->CreateSessionOptions(&so), api);
    api->SetIntraOpNumThreads(so, 1);
    api->SetSessionGraphOptimizationLevel(so, ORT_ENABLE_BASIC);

    check(api->CreateSession(h->env, model_path, so, &h->sess), api);
    api->ReleaseSessionOptions(so);
    check(api->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &h->mem), api);

    size_t want = infer_required_samples(h->sess, api);
    size_t win = (size_t)(window_seconds * (float)h->sr);
    h->need_samples = std::max(want, win > 0 ? win : (size_t)h->sr);
    h->ring.assign(h->need_samples, 0.f);
  } catch (...) {
    oww_destroy(h);
    return nullptr;
  }
  return h;
}

static float run_infer(oww_handle* h, const float* window){
  const OrtApi* api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
  // names
  size_t in_cnt=0,out_cnt=0;
  check(api->SessionGetInputCount(h->sess, &in_cnt), api);
  check(api->SessionGetOutputCount(h->sess, &out_cnt), api);
  if(in_cnt==0 || out_cnt==0) throw std::runtime_error("bad io");

  OrtAllocator* alloc=nullptr;
  check(api->GetAllocatorWithDefaultOptions(&alloc), api);
  char* in_name = nullptr; check(api->SessionGetInputName(h->sess, 0, alloc, &in_name), api);
  char* out_name = nullptr; check(api->SessionGetOutputName(h->sess, 0, alloc, &out_name), api);

  // shape: try [1,N] by default
  int64_t dims[2] = {1, (int64_t)h->need_samples};
  OrtValue* in_t = nullptr;
  check(api->CreateTensorWithDataAsOrtValue(
        h->mem, (void*)window, sizeof(float)*h->need_samples,
        dims, 2, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &in_t), api);

  OrtValue* out_t = nullptr;
  const char* in_names[] = { in_name };
  const char* out_names[] = { out_name };

  check(api->Run(h->sess, nullptr,
                 in_names, (const OrtValue* const*)&in_t, 1,
                 out_names, 1, &out_t), api);

  float score = 0.f;
  if (out_t){
    float* p = nullptr;
    check(api->GetTensorMutableData(out_t, (void**)&p), api);
    // 取第一个标量/最大值
    if (p) score = *p;
  }

  api->ReleaseValue(out_t);
  api->ReleaseValue(in_t);
  alloc->Free(alloc, in_name);
  alloc->Free(alloc, out_name);
  return score;
}

int oww_process(oww_handle* h, const float* pcm, size_t n_samples, float* out_score){
  if (!h || !pcm || n_samples==0) return -1;
  std::lock_guard<std::mutex> lk(h->mu);

  // 写入环形缓冲
  size_t N = h->need_samples;
  size_t pos = h->ring_wpos;
  size_t left = N - pos;
  if (n_samples <= left){
    std::memcpy(&h->ring[pos], pcm, n_samples*sizeof(float));
    h->ring_wpos += n_samples;
    if (h->ring_wpos == N){ h->ring_wpos = 0; h->filled = true; }
  } else {
    std::memcpy(&h->ring[pos], pcm, left*sizeof(float));
    std::memcpy(&h->ring[0], pcm+left, (n_samples-left)*sizeof(float));
    h->ring_wpos = (n_samples-left) % N;
    h->filled = true;
  }

  if (!h->filled) { if(out_score) *out_score = 0.f; return 0; }

  // 组装连续窗口（避免跨界）
  std::vector<float> win(N);
  size_t tail = N - h->ring_wpos;
  std::memcpy(win.data(), &h->ring[h->ring_wpos], tail*sizeof(float));
  if (h->ring_wpos) std::memcpy(win.data()+tail, &h->ring[0], h->ring_wpos*sizeof(float));

  float s = 0.f;
  try { s = run_infer(h, win.data()); }
  catch (...) { return -2; }

  if (out_score) *out_score = s;
  return (s >= h->threshold) ? 1 : 0;
}

void oww_reset(oww_handle* h){
  if(!h) return;
  std::lock_guard<std::mutex> lk(h->mu);
  std::fill(h->ring.begin(), h->ring.end(), 0.f);
  h->ring_wpos = 0; h->filled = false;
}

void oww_destroy(oww_handle* h){
  if(!h) return;
  const OrtApi* api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
  if (h->mem) { api->ReleaseMemoryInfo(h->mem); h->mem=nullptr; }
  if (h->sess){ api->ReleaseSession(h->sess); h->sess=nullptr; }
  if (h->env) { api->ReleaseEnv(h->env); h->env=nullptr; }
  delete h;
}
