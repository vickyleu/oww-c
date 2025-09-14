#include "oww.h"
#include <onnxruntime_c_api.h>
#include <vector>
#include <string>
#include <memory>
#include <cstring>

struct oww_handle {
    const OrtApi* api;
    OrtEnv* env;
    OrtSession* session;
    OrtSessionOptions* so;
    OrtAllocator* allocator;
    size_t input_samples;
    std::vector<float> buffer;
};

static size_t infer_required_samples(OrtSession* session, const OrtApi* api, OrtAllocator* allocator) {
    OrtTypeInfo* ti = nullptr;
    api->SessionGetInputTypeInfo(session, 0, &ti);

    const OrtTensorTypeAndShapeInfo* tsh = nullptr;
    api->CastTypeInfoToTensorInfo(ti, &tsh);

    size_t dim_count = 0;
    api->GetDimensionsCount(tsh, &dim_count);

    std::vector<int64_t> dims(dim_count);
    api->GetDimensions(tsh, dims.data(), dim_count);

    api->ReleaseTypeInfo(ti);

    return (size_t)dims[1]; // 假设输入是 [1, N]
}

oww_handle* oww_create(const char* model_path, int threads, float sensitivity, float threshold) {
    auto h = new oww_handle;
    h->api = OrtGetApiBase()->GetApi(ORT_API_VERSION);

    h->api->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "oww", &h->env);
    h->api->CreateSessionOptions(&h->so);
    h->api->SetIntraOpNumThreads(h->so, threads);
    h->api->SetSessionGraphOptimizationLevel(h->so, ORT_ENABLE_BASIC);

    h->api->CreateSession(h->env, model_path, h->so, &h->session);
    h->api->GetAllocatorWithDefaultOptions(&h->allocator);

    h->input_samples = infer_required_samples(h->session, h->api, h->allocator);
    h->buffer.resize(h->input_samples);

    return h;
}

int oww_process(oww_handle* h, const float* pcm, size_t samples) {
    if (samples < h->input_samples) return 0;

    memcpy(h->buffer.data(), pcm, h->input_samples * sizeof(float));

    OrtMemoryInfo* mem_info;
    h->api->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &mem_info);

    int64_t shape[2] = {1, (int64_t)h->input_samples};
    OrtValue* input_tensor = nullptr;
    h->api->CreateTensorWithDataAsOrtValue(mem_info,
        h->buffer.data(),
        h->input_samples * sizeof(float),
        shape, 2, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &input_tensor);

    OrtValue* output_tensor = nullptr;
    const char* input_names[] = {"input"};
    const char* output_names[] = {"output"};
    h->api->Run(h->session, nullptr,
        input_names, (const OrtValue* const*)&input_tensor, 1,
        output_names, 1, &output_tensor);

    float* out_data;
    h->api->GetTensorMutableData(output_tensor, (void**)&out_data);

    int result = out_data[0] > 0.5f ? 1 : 0;

    h->api->ReleaseValue(input_tensor);
    h->api->ReleaseValue(output_tensor);
    h->api->ReleaseMemoryInfo(mem_info);

    return result;
}

void oww_destroy(oww_handle* h) {
    h->api->ReleaseSession(h->session);
    h->api->ReleaseSessionOptions(h->so);
    h->api->ReleaseEnv(h->env);
    delete h;
}
