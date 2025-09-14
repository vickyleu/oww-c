#pragma once
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct oww_handle oww_handle;

oww_handle* oww_create(const char* model_path, int threads, float sensitivity, float threshold);
int oww_process(oww_handle* h, const float* pcm, size_t samples);
void oww_destroy(oww_handle* h);

#ifdef __cplusplus
}
#endif
