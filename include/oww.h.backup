#pragma once
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif

typedef struct oww_handle oww_handle;

/** 两段式：melspectrogram.onnx + wakeword_detector.onnx（embed_onnx参数保留兼容性但不使用） */
oww_handle* oww_create(const char* melspec_onnx,
                       const char* embed_onnx,
                       const char* detector_onnx,
                       int threads,
                       float threshold);

/** 流式喂入 PCM（16kHz，单声道） */
int  oww_process_f32(oww_handle* h, const float* pcm, size_t samples);
int  oww_process_i16(oww_handle* h, const short* pcm, size_t samples);

/** 重置内部缓冲（切换音源/场景时可用） */
void oww_reset(oww_handle* h);

/** 最近一次检测分数（0..1） */
float oww_last_score(const oww_handle* h);

/** 每个输入块建议大小（样本数，约等于80ms=1280） */
size_t oww_recommended_chunk();

/** 销毁 */
void oww_destroy(oww_handle* h);

// ==================== 新的KWS单模型接口 ====================

typedef struct kws_handle kws_handle;

/** 单模型KWS：直接使用ONNX模型进行唤醒词检测 */
kws_handle* kws_create(const char* model_path,
                       int threads,
                       float threshold);

/** 流式喂入 PCM（16kHz，单声道，int16） */
int kws_process_i16(kws_handle* h, const short* pcm, size_t samples);

/** 重置内部缓冲 */
void kws_reset(kws_handle* h);

/** 最近一次检测分数（0..1） */
float kws_last_score(const kws_handle* h);

/** 每个输入块建议大小（样本数，10ms=160） */
size_t kws_recommended_chunk();

/** 销毁 */
void kws_destroy(kws_handle* h);

#ifdef __cplusplus
}
#endif
