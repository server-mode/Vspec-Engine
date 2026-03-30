#ifndef VSPEC_PYTHON_CAPI_H
#define VSPEC_PYTHON_CAPI_H

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32)
  #if defined(VSPEC_PYTHON_CAPI_BUILD)
    #define VSPEC_PY_API __declspec(dllexport)
  #else
    #define VSPEC_PY_API __declspec(dllimport)
  #endif
#else
  #define VSPEC_PY_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

VSPEC_PY_API const char* vspec_py_version(void);
VSPEC_PY_API int vspec_py_load_manifest_count(const char* path);
VSPEC_PY_API int vspec_py_parse_safetensors_count(const char* path);
VSPEC_PY_API int vspec_py_rewrite_demo_final_nodes(void);
VSPEC_PY_API int vspec_py_generate(const char* prompt, char* out, size_t out_size);
VSPEC_PY_API int vspec_py_weight_canonical_name(const char* raw_name, char* out_name, size_t out_name_size);
VSPEC_PY_API int vspec_py_safetensors_tensor_descriptor(
  const char* path,
  const char* tensor_name,
  char* out_dtype,
  size_t out_dtype_size,
  size_t* out_shape,
  size_t shape_cap,
  size_t* out_ndim,
  uint64_t* out_data_start,
  uint64_t* out_data_end
);
VSPEC_PY_API int vspec_py_sample_candidate(
  const int* token_ids,
  const float* scores,
  size_t count,
  int greedy,
  uint64_t random_bits,
  int* out_token_id
);
VSPEC_PY_API int vspec_py_kv_cache_create(size_t page_tokens, size_t max_pages, size_t num_heads, size_t head_dim);
VSPEC_PY_API void vspec_py_kv_cache_destroy(int handle_id);
VSPEC_PY_API int vspec_py_kv_cache_reset(int handle_id);
VSPEC_PY_API int vspec_py_kv_cache_append(int handle_id, uint64_t session_id, const float* key_token, const float* value_token);
VSPEC_PY_API size_t vspec_py_kv_cache_session_tokens(int handle_id, uint64_t session_id);
VSPEC_PY_API size_t vspec_py_kv_cache_read(
  int handle_id,
  uint64_t session_id,
  float* out_keys,
  float* out_values,
  size_t max_tokens
);
VSPEC_PY_API int vspec_py_decode_session_create(
  size_t total_vram_bytes,
  size_t max_active,
  size_t max_batch_tokens,
  size_t token_quantum
);
VSPEC_PY_API void vspec_py_decode_session_destroy(int handle_id);
VSPEC_PY_API int vspec_py_decode_session_begin(
  int handle_id,
  size_t reserve_bytes,
  size_t prompt_tokens,
  size_t max_new_tokens,
  uint16_t priority
);
VSPEC_PY_API size_t vspec_py_decode_session_next_quota(int handle_id);
VSPEC_PY_API int vspec_py_decode_session_commit(int handle_id, size_t generated_tokens, int reached_eos);
VSPEC_PY_API int vspec_py_decode_session_cancel(int handle_id);
VSPEC_PY_API int vspec_py_decode_session_is_active(int handle_id);
VSPEC_PY_API size_t vspec_py_decode_session_remaining_tokens(int handle_id);

VSPEC_PY_API int vspec_py_native_decode_loop_create(
  size_t total_vram_bytes,
  size_t max_active,
  size_t max_batch_tokens,
  size_t token_quantum
);
VSPEC_PY_API void vspec_py_native_decode_loop_destroy(int handle_id);
VSPEC_PY_API int vspec_py_native_decode_loop_begin(
  int handle_id,
  size_t reserve_bytes,
  size_t prompt_tokens,
  size_t max_new_tokens,
  uint16_t priority,
  uint64_t graph_signature
);
VSPEC_PY_API size_t vspec_py_native_decode_loop_next_quota(int handle_id);
VSPEC_PY_API int vspec_py_native_decode_loop_commit(int handle_id, size_t generated_tokens, int reached_eos);
VSPEC_PY_API int vspec_py_native_decode_loop_cancel(int handle_id);
VSPEC_PY_API int vspec_py_native_decode_loop_stats(
  int handle_id,
  uint64_t* out_graph_signature,
  uint64_t* out_graph_reuse_hits,
  uint64_t* out_graph_reuse_misses,
  uint64_t* out_steps
);
VSPEC_PY_API int vspec_py_native_decode_loop_graph_cache_stats(
  int handle_id,
  uint64_t* out_graph_captures,
  uint64_t* out_graph_replays,
  uint64_t* out_cached_signatures,
  int* out_graph_capture_enabled
);
VSPEC_PY_API int vspec_py_continuous_batch_create(
  size_t total_vram_bytes,
  size_t max_active,
  size_t max_batch_items,
  size_t max_batch_tokens,
  size_t prefill_quantum,
  size_t decode_quantum
);
VSPEC_PY_API void vspec_py_continuous_batch_destroy(int handle_id);
VSPEC_PY_API int vspec_py_continuous_batch_submit(
  int handle_id,
  size_t reserve_bytes,
  size_t prompt_tokens,
  size_t max_new_tokens,
  uint16_t priority,
  uint64_t* out_request_id
);
VSPEC_PY_API size_t vspec_py_continuous_batch_next(
  int handle_id,
  uint64_t* out_request_ids,
  uint32_t* out_phases,
  size_t* out_quotas,
  size_t* out_prompt_cursors,
  size_t cap
);
VSPEC_PY_API int vspec_py_continuous_batch_commit_prefill(int handle_id, uint64_t request_id, size_t consumed_tokens);
VSPEC_PY_API int vspec_py_continuous_batch_commit_decode(int handle_id, uint64_t request_id, size_t generated_tokens, int reached_eos);
VSPEC_PY_API int vspec_py_continuous_batch_cancel(int handle_id, uint64_t request_id);
VSPEC_PY_API int vspec_py_continuous_batch_stats(
  int handle_id,
  uint64_t* out_prefill_steps,
  uint64_t* out_decode_steps,
  uint64_t* out_prefill_tokens,
  uint64_t* out_decode_tokens,
  size_t* out_active_prefill,
  size_t* out_active_decode,
  size_t* out_reserved_vram
);

VSPEC_PY_API int vspec_py_runtime_adaptive_step(
  const char* token_text,
  float token_entropy,
  float attention_entropy_collapse,
  float latency_ms,
  float vram_pressure,
  float quality_drift,
  uint32_t layer_type,
  uint8_t* out_target_bits,
  uint8_t* out_skip_compute,
  uint8_t* out_reduce_attention_depth,
  uint8_t* out_enable_kv_compression,
  uint8_t* out_routed_bits,
  uint32_t* out_attention_depth_hint,
  uint32_t* out_token_tier,
  float* out_token_importance,
  uint32_t* out_kv_action
);

VSPEC_PY_API int vspec_py_runtime_output_guard_init(float strictness);
VSPEC_PY_API int vspec_py_runtime_output_guard_allow(const char* text_fragment);
VSPEC_PY_API float vspec_py_runtime_output_guard_score_adjustment(const char* text_fragment);
VSPEC_PY_API int vspec_py_runtime_output_guard_observe(const char* text_fragment);

VSPEC_PY_API int vspec_py_runtime_anf_available(void);
VSPEC_PY_API int vspec_py_runtime_anf_observe_activations(const float* activations, size_t count);
VSPEC_PY_API int vspec_py_runtime_anf_observe_quality(float residual_rms, float attention_entropy_collapse, float activation_norm_drift);
VSPEC_PY_API int vspec_py_runtime_anf_report(
  int* out_anf_available,
  int* out_anf_mode,
  float* out_hot_ratio,
  uint32_t* out_hot_neurons,
  uint32_t* out_tokens_observed,
  float* out_hot_ratio_avg,
  float* out_skip_ratio_avg,
  uint32_t* out_cache_updates,
  float* out_error_wave_avg,
  float* out_contamination_avg,
  uint32_t* out_cascade_depth,
  uint32_t* out_cascade_depth_max,
  uint32_t* out_forced_fallback_count,
  uint32_t* out_silent_stop_count
);

VSPEC_PY_API int vspec_py_native_forward_create(const char* model_path, uint64_t seed);
VSPEC_PY_API void vspec_py_native_forward_destroy(int handle_id);
VSPEC_PY_API int vspec_py_native_forward_step(
  int handle_id,
  const char* prompt,
  size_t produced_tokens,
  const int* candidate_ids,
  const float* base_scores,
  size_t candidate_count,
  float blend,
  float* out_scores
);

VSPEC_PY_API int vspec_py_plugin_load_dynamic(const char* path, const char* symbol_name, char* out_msg, size_t out_msg_size);
VSPEC_PY_API int vspec_py_plugin_unload_dynamic(const char* name, char* out_msg, size_t out_msg_size);

#ifdef __cplusplus
}
#endif

#endif
