#define _GNU_SOURCE

#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <unistd.h>

typedef struct minicpmo_gpu_stats {
    uint64_t backend_init_us;
    uint64_t host_to_device_bytes;
    uint64_t device_to_host_bytes;
    uint64_t weight_upload_bytes;
    uint64_t scratch_x_bytes;
    uint64_t scratch_y_bytes;
    uint64_t uploaded_weight_count;
    uint64_t native_quant_matvec_count;
    uint64_t staged_quant_matvec_count;
    uint64_t q8_1_x_bytes;
    uint64_t attention_kernel_calls;
    uint64_t kv_device_bytes;
    uint64_t kv_host_to_device_bytes;
} minicpmo_gpu_stats;

typedef int (*minicpmo_gpu_context_create_fn)(unsigned int, int, void **, char *, size_t);
typedef void (*minicpmo_gpu_context_destroy_fn)(void *);
typedef int (*minicpmo_gpu_context_smoke_fn)(void *, char *, size_t);
typedef int (*minicpmo_gpu_context_upload_tensor_fn)(void *, const char *, const void *, size_t, int, size_t, size_t, char *, size_t);
typedef int (*minicpmo_gpu_context_matvec_fn)(void *, const char *, const void *, int, const float *, size_t, size_t, float *, char *, size_t);
typedef int (*minicpmo_gpu_context_prepare_attention_fn)(void *, size_t, size_t, size_t, char *, size_t);
typedef int (*minicpmo_gpu_context_attention_fn)(void *, const float *, const float *, const float *, size_t, size_t, size_t, size_t, size_t, size_t, float *, char *, size_t);
typedef int (*minicpmo_gpu_context_slot_upload_f32_fn)(void *, unsigned int, const float *, size_t, char *, size_t);
typedef int (*minicpmo_gpu_context_slot_download_f32_fn)(void *, unsigned int, float *, size_t, char *, size_t);
typedef int (*minicpmo_gpu_context_slot_matvec_fn)(void *, unsigned int, const char *, const void *, int, unsigned int, size_t, size_t, char *, size_t);
typedef int (*minicpmo_gpu_context_slot_rms_norm_fn)(void *, unsigned int, unsigned int, const void *, size_t, float, char *, size_t);
typedef int (*minicpmo_gpu_context_slot_head_norm_fn)(void *, unsigned int, const void *, size_t, size_t, char *, size_t);
typedef int (*minicpmo_gpu_context_slot_rope_fn)(void *, unsigned int, size_t, size_t, size_t, float, float, char *, size_t);
typedef int (*minicpmo_gpu_context_slot_add_inplace_fn)(void *, unsigned int, unsigned int, size_t, char *, size_t);
typedef int (*minicpmo_gpu_context_slot_silu_mul_fn)(void *, unsigned int, unsigned int, unsigned int, size_t, char *, size_t);
typedef int (*minicpmo_gpu_context_attention_slots_fn)(void *, unsigned int, unsigned int, unsigned int, unsigned int, size_t, size_t, size_t, size_t, size_t, size_t, char *, size_t);
typedef void (*minicpmo_gpu_context_snapshot_fn)(void *, minicpmo_gpu_stats *);

typedef struct minicpmo_gpu_helper_api {
    void *lib;
    minicpmo_gpu_context_create_fn create;
    minicpmo_gpu_context_destroy_fn destroy;
    minicpmo_gpu_context_smoke_fn smoke;
    minicpmo_gpu_context_upload_tensor_fn upload_tensor;
    minicpmo_gpu_context_matvec_fn matvec;
    minicpmo_gpu_context_prepare_attention_fn prepare_attention;
    minicpmo_gpu_context_attention_fn attention;
    minicpmo_gpu_context_slot_upload_f32_fn slot_upload_f32;
    minicpmo_gpu_context_slot_download_f32_fn slot_download_f32;
    minicpmo_gpu_context_slot_matvec_fn slot_matvec;
    minicpmo_gpu_context_slot_rms_norm_fn slot_rms_norm;
    minicpmo_gpu_context_slot_head_norm_fn slot_head_norm;
    minicpmo_gpu_context_slot_rope_fn slot_rope;
    minicpmo_gpu_context_slot_add_inplace_fn slot_add_inplace;
    minicpmo_gpu_context_slot_silu_mul_fn slot_silu_mul;
    minicpmo_gpu_context_attention_slots_fn attention_slots;
    minicpmo_gpu_context_snapshot_fn snapshot;
} minicpmo_gpu_helper_api;

static minicpmo_gpu_helper_api minicpmo_gpu_helper = {0};

int minicpmo_gpu_context_create(unsigned int device_index, int debug, void **out_handle, char *error, size_t error_cap);
void minicpmo_gpu_context_destroy(void *handle);
int minicpmo_gpu_context_smoke(void *handle, char *error, size_t error_cap);
int minicpmo_gpu_context_upload_tensor(void *handle, const char *name, const void *host_ptr, size_t bytes, int dtype, size_t rows, size_t cols, char *error, size_t error_cap);
int minicpmo_gpu_context_matvec(void *handle, const char *tensor_name, const void *host_ptr, int dtype, const float *x, size_t in_dim, size_t out_dim, float *dst, char *error, size_t error_cap);
int minicpmo_gpu_context_prepare_attention(void *handle, size_t layer_count, size_t context_capacity, size_t kv_dim, char *error, size_t error_cap);
int minicpmo_gpu_context_attention(void *handle, const float *q, const float *k, const float *v, size_t layer, size_t pos, size_t head_count, size_t kv_head_count, size_t head_dim, size_t context_capacity, float *dst, char *error, size_t error_cap);
int minicpmo_gpu_context_slot_upload_f32(void *handle, unsigned int slot, const float *src, size_t len, char *error, size_t error_cap);
int minicpmo_gpu_context_slot_download_f32(void *handle, unsigned int slot, float *dst, size_t len, char *error, size_t error_cap);
int minicpmo_gpu_context_slot_matvec(void *handle, unsigned int dst_slot, const char *tensor_name, const void *host_ptr, int dtype, unsigned int src_slot, size_t in_dim, size_t out_dim, char *error, size_t error_cap);
int minicpmo_gpu_context_slot_rms_norm(void *handle, unsigned int dst_slot, unsigned int src_slot, const void *weight_host_ptr, size_t n, float eps, char *error, size_t error_cap);
int minicpmo_gpu_context_slot_head_norm(void *handle, unsigned int slot, const void *weight_host_ptr, size_t head_count, size_t head_dim, char *error, size_t error_cap);
int minicpmo_gpu_context_slot_rope(void *handle, unsigned int slot, size_t head_count, size_t head_dim, size_t pos, float freq_base, float freq_scale, char *error, size_t error_cap);
int minicpmo_gpu_context_slot_add_inplace(void *handle, unsigned int dst_slot, unsigned int src_slot, size_t len, char *error, size_t error_cap);
int minicpmo_gpu_context_slot_silu_mul(void *handle, unsigned int dst_slot, unsigned int gate_slot, unsigned int up_slot, size_t len, char *error, size_t error_cap);
int minicpmo_gpu_context_attention_slots(void *handle, unsigned int dst_slot, unsigned int q_slot, unsigned int k_slot, unsigned int v_slot, size_t layer, size_t pos, size_t head_count, size_t kv_head_count, size_t head_dim, size_t context_capacity, char *error, size_t error_cap);
void minicpmo_gpu_context_snapshot(void *handle, minicpmo_gpu_stats *stats);

static void minicpmo_gpu_bridge_copy_error(char *dst, size_t dst_cap, const char *src) {
    size_t i;
    if (dst == NULL || dst_cap == 0) {
        return;
    }
    if (src == NULL) {
        dst[0] = '\0';
        return;
    }
    for (i = 0; i + 1 < dst_cap && src[i] != '\0'; ++i) {
        dst[i] = src[i];
    }
    dst[i] = '\0';
}

static int minicpmo_gpu_bridge_build_helper_path(char *dst, size_t dst_cap) {
    ssize_t len;
    char exe_path[4096];
    size_t i;
    if (dst == NULL || dst_cap < 64) {
        return 0;
    }
    len = readlink("/proc/self/exe", exe_path, sizeof(exe_path) - 1);
    if (len <= 0) {
        return 0;
    }
    exe_path[len] = '\0';
    i = (size_t) len;
    while (i > 0 && exe_path[i - 1] != '/') {
        --i;
    }
    exe_path[i] = '\0';
    snprintf(dst, dst_cap, "%slibminicpmo_gpu_helper.so", exe_path);
    return 1;
}

static int minicpmo_gpu_bridge_open(char *error, size_t error_cap) {
    char helper_path[4096];
    void *lib;
    if (minicpmo_gpu_helper.lib != NULL) {
        return 1;
    }
    if (!minicpmo_gpu_bridge_build_helper_path(helper_path, sizeof(helper_path))) {
        minicpmo_gpu_bridge_copy_error(error, error_cap, "error: unable to resolve gpu helper path from /proc/self/exe");
        return 0;
    }
    lib = dlmopen(LM_ID_NEWLM, helper_path, RTLD_NOW | RTLD_LOCAL);
    if (lib == NULL) {
        char message[512];
        snprintf(message, sizeof(message), "error: unable to load gpu helper %s (%s)", helper_path, dlerror());
        minicpmo_gpu_bridge_copy_error(error, error_cap, message);
        return 0;
    }
    minicpmo_gpu_helper.lib = lib;
    minicpmo_gpu_helper.create = (minicpmo_gpu_context_create_fn) dlsym(lib, "minicpmo_gpu_context_create");
    minicpmo_gpu_helper.destroy = (minicpmo_gpu_context_destroy_fn) dlsym(lib, "minicpmo_gpu_context_destroy");
    minicpmo_gpu_helper.smoke = (minicpmo_gpu_context_smoke_fn) dlsym(lib, "minicpmo_gpu_context_smoke");
    minicpmo_gpu_helper.upload_tensor = (minicpmo_gpu_context_upload_tensor_fn) dlsym(lib, "minicpmo_gpu_context_upload_tensor");
    minicpmo_gpu_helper.matvec = (minicpmo_gpu_context_matvec_fn) dlsym(lib, "minicpmo_gpu_context_matvec");
    minicpmo_gpu_helper.prepare_attention = (minicpmo_gpu_context_prepare_attention_fn) dlsym(lib, "minicpmo_gpu_context_prepare_attention");
    minicpmo_gpu_helper.attention = (minicpmo_gpu_context_attention_fn) dlsym(lib, "minicpmo_gpu_context_attention");
    minicpmo_gpu_helper.slot_upload_f32 = (minicpmo_gpu_context_slot_upload_f32_fn) dlsym(lib, "minicpmo_gpu_context_slot_upload_f32");
    minicpmo_gpu_helper.slot_download_f32 = (minicpmo_gpu_context_slot_download_f32_fn) dlsym(lib, "minicpmo_gpu_context_slot_download_f32");
    minicpmo_gpu_helper.slot_matvec = (minicpmo_gpu_context_slot_matvec_fn) dlsym(lib, "minicpmo_gpu_context_slot_matvec");
    minicpmo_gpu_helper.slot_rms_norm = (minicpmo_gpu_context_slot_rms_norm_fn) dlsym(lib, "minicpmo_gpu_context_slot_rms_norm");
    minicpmo_gpu_helper.slot_head_norm = (minicpmo_gpu_context_slot_head_norm_fn) dlsym(lib, "minicpmo_gpu_context_slot_head_norm");
    minicpmo_gpu_helper.slot_rope = (minicpmo_gpu_context_slot_rope_fn) dlsym(lib, "minicpmo_gpu_context_slot_rope");
    minicpmo_gpu_helper.slot_add_inplace = (minicpmo_gpu_context_slot_add_inplace_fn) dlsym(lib, "minicpmo_gpu_context_slot_add_inplace");
    minicpmo_gpu_helper.slot_silu_mul = (minicpmo_gpu_context_slot_silu_mul_fn) dlsym(lib, "minicpmo_gpu_context_slot_silu_mul");
    minicpmo_gpu_helper.attention_slots = (minicpmo_gpu_context_attention_slots_fn) dlsym(lib, "minicpmo_gpu_context_attention_slots");
    minicpmo_gpu_helper.snapshot = (minicpmo_gpu_context_snapshot_fn) dlsym(lib, "minicpmo_gpu_context_snapshot");
    if (minicpmo_gpu_helper.create == NULL || minicpmo_gpu_helper.destroy == NULL || minicpmo_gpu_helper.smoke == NULL ||
        minicpmo_gpu_helper.upload_tensor == NULL || minicpmo_gpu_helper.matvec == NULL || minicpmo_gpu_helper.prepare_attention == NULL ||
        minicpmo_gpu_helper.attention == NULL || minicpmo_gpu_helper.slot_upload_f32 == NULL || minicpmo_gpu_helper.slot_download_f32 == NULL ||
        minicpmo_gpu_helper.slot_matvec == NULL || minicpmo_gpu_helper.slot_rms_norm == NULL || minicpmo_gpu_helper.slot_head_norm == NULL ||
        minicpmo_gpu_helper.slot_rope == NULL || minicpmo_gpu_helper.slot_add_inplace == NULL || minicpmo_gpu_helper.slot_silu_mul == NULL ||
        minicpmo_gpu_helper.attention_slots == NULL || minicpmo_gpu_helper.snapshot == NULL) {
        minicpmo_gpu_bridge_copy_error(error, error_cap, "error: gpu helper missing exported symbols");
        dlclose(lib);
        memset(&minicpmo_gpu_helper, 0, sizeof(minicpmo_gpu_helper));
        return 0;
    }
    return 1;
}

int minicpmo_gpu_context_create(unsigned int device_index, int debug, void **out_handle, char *error, size_t error_cap) {
    if (!minicpmo_gpu_bridge_open(error, error_cap)) {
        return 0;
    }
    return minicpmo_gpu_helper.create(device_index, debug, out_handle, error, error_cap);
}

void minicpmo_gpu_context_destroy(void *handle) {
    if (minicpmo_gpu_helper.lib == NULL) {
        return;
    }
    minicpmo_gpu_helper.destroy(handle);
}

int minicpmo_gpu_context_smoke(void *handle, char *error, size_t error_cap) {
    if (!minicpmo_gpu_bridge_open(error, error_cap)) {
        return 0;
    }
    return minicpmo_gpu_helper.smoke(handle, error, error_cap);
}

int minicpmo_gpu_context_upload_tensor(void *handle, const char *name, const void *host_ptr, size_t bytes, int dtype, size_t rows, size_t cols, char *error, size_t error_cap) {
    if (!minicpmo_gpu_bridge_open(error, error_cap)) {
        return 0;
    }
    return minicpmo_gpu_helper.upload_tensor(handle, name, host_ptr, bytes, dtype, rows, cols, error, error_cap);
}

int minicpmo_gpu_context_matvec(void *handle, const char *tensor_name, const void *host_ptr, int dtype, const float *x, size_t in_dim, size_t out_dim, float *dst, char *error, size_t error_cap) {
    if (!minicpmo_gpu_bridge_open(error, error_cap)) {
        return 0;
    }
    return minicpmo_gpu_helper.matvec(handle, tensor_name, host_ptr, dtype, x, in_dim, out_dim, dst, error, error_cap);
}

int minicpmo_gpu_context_prepare_attention(void *handle, size_t layer_count, size_t context_capacity, size_t kv_dim, char *error, size_t error_cap) {
    if (!minicpmo_gpu_bridge_open(error, error_cap)) {
        return 0;
    }
    return minicpmo_gpu_helper.prepare_attention(handle, layer_count, context_capacity, kv_dim, error, error_cap);
}

int minicpmo_gpu_context_attention(void *handle, const float *q, const float *k, const float *v, size_t layer, size_t pos, size_t head_count, size_t kv_head_count, size_t head_dim, size_t context_capacity, float *dst, char *error, size_t error_cap) {
    if (!minicpmo_gpu_bridge_open(error, error_cap)) {
        return 0;
    }
    return minicpmo_gpu_helper.attention(handle, q, k, v, layer, pos, head_count, kv_head_count, head_dim, context_capacity, dst, error, error_cap);
}

int minicpmo_gpu_context_slot_upload_f32(void *handle, unsigned int slot, const float *src, size_t len, char *error, size_t error_cap) {
    if (!minicpmo_gpu_bridge_open(error, error_cap)) {
        return 0;
    }
    return minicpmo_gpu_helper.slot_upload_f32(handle, slot, src, len, error, error_cap);
}

int minicpmo_gpu_context_slot_download_f32(void *handle, unsigned int slot, float *dst, size_t len, char *error, size_t error_cap) {
    if (!minicpmo_gpu_bridge_open(error, error_cap)) {
        return 0;
    }
    return minicpmo_gpu_helper.slot_download_f32(handle, slot, dst, len, error, error_cap);
}

int minicpmo_gpu_context_slot_matvec(void *handle, unsigned int dst_slot, const char *tensor_name, const void *host_ptr, int dtype, unsigned int src_slot, size_t in_dim, size_t out_dim, char *error, size_t error_cap) {
    if (!minicpmo_gpu_bridge_open(error, error_cap)) {
        return 0;
    }
    return minicpmo_gpu_helper.slot_matvec(handle, dst_slot, tensor_name, host_ptr, dtype, src_slot, in_dim, out_dim, error, error_cap);
}

int minicpmo_gpu_context_slot_rms_norm(void *handle, unsigned int dst_slot, unsigned int src_slot, const void *weight_host_ptr, size_t n, float eps, char *error, size_t error_cap) {
    if (!minicpmo_gpu_bridge_open(error, error_cap)) {
        return 0;
    }
    return minicpmo_gpu_helper.slot_rms_norm(handle, dst_slot, src_slot, weight_host_ptr, n, eps, error, error_cap);
}

int minicpmo_gpu_context_slot_head_norm(void *handle, unsigned int slot, const void *weight_host_ptr, size_t head_count, size_t head_dim, char *error, size_t error_cap) {
    if (!minicpmo_gpu_bridge_open(error, error_cap)) {
        return 0;
    }
    return minicpmo_gpu_helper.slot_head_norm(handle, slot, weight_host_ptr, head_count, head_dim, error, error_cap);
}

int minicpmo_gpu_context_slot_rope(void *handle, unsigned int slot, size_t head_count, size_t head_dim, size_t pos, float freq_base, float freq_scale, char *error, size_t error_cap) {
    if (!minicpmo_gpu_bridge_open(error, error_cap)) {
        return 0;
    }
    return minicpmo_gpu_helper.slot_rope(handle, slot, head_count, head_dim, pos, freq_base, freq_scale, error, error_cap);
}

int minicpmo_gpu_context_slot_add_inplace(void *handle, unsigned int dst_slot, unsigned int src_slot, size_t len, char *error, size_t error_cap) {
    if (!minicpmo_gpu_bridge_open(error, error_cap)) {
        return 0;
    }
    return minicpmo_gpu_helper.slot_add_inplace(handle, dst_slot, src_slot, len, error, error_cap);
}

int minicpmo_gpu_context_slot_silu_mul(void *handle, unsigned int dst_slot, unsigned int gate_slot, unsigned int up_slot, size_t len, char *error, size_t error_cap) {
    if (!minicpmo_gpu_bridge_open(error, error_cap)) {
        return 0;
    }
    return minicpmo_gpu_helper.slot_silu_mul(handle, dst_slot, gate_slot, up_slot, len, error, error_cap);
}

int minicpmo_gpu_context_attention_slots(void *handle, unsigned int dst_slot, unsigned int q_slot, unsigned int k_slot, unsigned int v_slot, size_t layer, size_t pos, size_t head_count, size_t kv_head_count, size_t head_dim, size_t context_capacity, char *error, size_t error_cap) {
    if (!minicpmo_gpu_bridge_open(error, error_cap)) {
        return 0;
    }
    return minicpmo_gpu_helper.attention_slots(handle, dst_slot, q_slot, k_slot, v_slot, layer, pos, head_count, kv_head_count, head_dim, context_capacity, error, error_cap);
}

void minicpmo_gpu_context_snapshot(void *handle, minicpmo_gpu_stats *stats) {
    char ignored_error[8];
    if (!minicpmo_gpu_bridge_open(ignored_error, sizeof(ignored_error))) {
        if (stats != NULL) {
            memset(stats, 0, sizeof(*stats));
        }
        return;
    }
    minicpmo_gpu_helper.snapshot(handle, stats);
}
