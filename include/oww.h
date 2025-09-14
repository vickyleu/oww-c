#pragma once
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif

typedef struct oww_handle oww_handle;

/** 三段式：melspectrogram.onnx + embedding_model.onnx + 某个 wakeword_xxx.onnx */
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

#ifdef __cplusplus
}
#endif
