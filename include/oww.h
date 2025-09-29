#pragma once
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif

// ==================== OpenWakeWord 三链接口 ====================

typedef struct oww_handle oww_handle;

/** 三链模式：melspectrogram.onnx → embedding_model.onnx → mlp_wakeword_(f32/int16).onnx */
oww_handle* oww_create(const char* mel_onnx,
                       const char* emb_onnx, 
                       const char* cls_onnx,
                       int threads,
                       float threshold);

/** 流式喂入 PCM（16kHz，单声道，int16） */
int oww_process_i16(oww_handle* h, const short* pcm, size_t samples);

/** 重置内部缓冲 */
void oww_reset(oww_handle* h);

/** 最近一次检测分数（0..1） */
float oww_last_score(const oww_handle* h);

/** 每个输入块建议大小（样本数，约等于80ms=1280） */
size_t oww_recommended_chunk();

/** 设置检测所需的最小缓冲区大小（样本数） */
void oww_set_buffer_size(oww_handle* h, size_t samples);

/** 销毁 */
void oww_destroy(oww_handle* h);

#ifdef __cplusplus
}
#endif
