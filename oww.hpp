#pragma once
#include <cstddef>
#ifdef __cplusplus
extern "C" {
#endif

typedef struct oww_handle oww_handle;

/** 创建检测器
 *  model_path: .onnx 模型路径（建议用自带前端的openWakeWord模型）
 *  sample_rate: 16000（推荐）
 *  window_seconds: 滑窗长度，常用1.0~2.0
 *  threshold: 触发阈值(0..1)
 *  return: NULL=失败
 */
oww_handle* oww_create(const char* model_path, int sample_rate,
                       float window_seconds, float threshold);

/** 输入一帧音频（float32, [-1,1], 单声道） */
int oww_process(oww_handle* h, const float* pcm, size_t n_samples,
                float* out_score /*可为NULL*/);

/** 重置内部缓冲 */
void oww_reset(oww_handle* h);

/** 释放 */
void oww_destroy(oww_handle* h);

#ifdef __cplusplus
}
#endif
