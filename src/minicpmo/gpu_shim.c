#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

typedef int CUresult;
typedef int CUdevice;
typedef unsigned long long CUdeviceptr;
typedef void *CUcontext;
typedef void *CUmodule;
typedef void *CUfunction;
typedef int cublasStatus_t;
typedef int cublasOperation_t;
typedef int cublasComputeType_t;
typedef int cublasGemmAlgo_t;
typedef int cudaDataType_t;
typedef void *cublasHandle_t;
typedef int nvrtcResult;
typedef void *nvrtcProgram;

enum {
    CUDA_SUCCESS = 0,
    CUBLAS_STATUS_SUCCESS = 0,
    CUBLAS_OP_N = 0,
    CUBLAS_OP_T = 1,
    CUDA_R_32F = 0,
    CUDA_R_16F = 2,
    CUDA_R_16BF = 14,
    CUBLAS_GEMM_DEFAULT = -1,
    CUBLAS_COMPUTE_32F = 68,
    MINICPMO_CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = 75,
    MINICPMO_CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR = 76,
    MINICPMO_GPU_MAX_UPLOADS = 512,
    NVRTC_SUCCESS = 0,
    MINICPMO_GPU_QK_K = 256,
    MINICPMO_GPU_QK8_1 = 32,
    MINICPMO_GPU_Q8_1_BLOCKS_PER_SUPER = MINICPMO_GPU_QK_K / MINICPMO_GPU_QK8_1,
    MINICPMO_GPU_Q8_1_BLOCK_BYTES = (2 * (int) sizeof(uint16_t)) + (MINICPMO_GPU_QK8_1 * (int) sizeof(int8_t)),
    MINICPMO_GPU_Q8_1_SUPER_BYTES = MINICPMO_GPU_Q8_1_BLOCKS_PER_SUPER * MINICPMO_GPU_Q8_1_BLOCK_BYTES,
    MINICPMO_GPU_QUANT_THREADS = 128,
    MINICPMO_GPU_ATTENTION_THREADS = 256,
    MINICPMO_GPU_VECTOR_THREADS = 256,
    MINICPMO_GPU_SLOT_COUNT = 16,
};

typedef CUresult (*minicpmo_cuda_init_fn)(unsigned int flags);
typedef CUresult (*minicpmo_cuda_device_get_count_fn)(int *count);
typedef CUresult (*minicpmo_cuda_device_get_fn)(CUdevice *device, int ordinal);
typedef CUresult (*minicpmo_cuda_device_get_attribute_fn)(int *pi, int attrib, CUdevice dev);
typedef CUresult (*minicpmo_cuda_ctx_create_fn)(CUcontext *pctx, unsigned int flags, CUdevice dev);
typedef CUresult (*minicpmo_cuda_ctx_destroy_fn)(CUcontext ctx);
typedef CUresult (*minicpmo_cuda_mem_alloc_fn)(CUdeviceptr *dptr, size_t bytesize);
typedef CUresult (*minicpmo_cuda_mem_free_fn)(CUdeviceptr dptr);
typedef CUresult (*minicpmo_cuda_memcpy_hto_d_fn)(CUdeviceptr dst, const void *src, size_t count);
typedef CUresult (*minicpmo_cuda_memcpy_dto_d_fn)(CUdeviceptr dst, CUdeviceptr src, size_t count);
typedef CUresult (*minicpmo_cuda_memcpy_dto_h_fn)(void *dst, CUdeviceptr src, size_t count);
typedef CUresult (*minicpmo_cuda_mem_get_info_fn)(size_t *free_bytes, size_t *total_bytes);
typedef CUresult (*minicpmo_cuda_get_error_string_fn)(CUresult error, const char **pstr);
typedef CUresult (*minicpmo_cuda_module_load_data_ex_fn)(CUmodule *module, const void *image, unsigned int num_options, void *options, void *option_values);
typedef CUresult (*minicpmo_cuda_module_get_function_fn)(CUfunction *hfunc, CUmodule hmod, const char *name);
typedef CUresult (*minicpmo_cuda_module_unload_fn)(CUmodule hmod);
typedef CUresult (*minicpmo_cuda_launch_kernel_fn)(
    CUfunction f,
    unsigned int grid_dim_x,
    unsigned int grid_dim_y,
    unsigned int grid_dim_z,
    unsigned int block_dim_x,
    unsigned int block_dim_y,
    unsigned int block_dim_z,
    unsigned int shared_mem_bytes,
    void *hstream,
    void **kernel_params,
    void **extra);
typedef CUresult (*minicpmo_cuda_ctx_synchronize_fn)(void);

typedef cublasStatus_t (*minicpmo_cublas_create_fn)(cublasHandle_t *handle);
typedef cublasStatus_t (*minicpmo_cublas_destroy_fn)(cublasHandle_t handle);
typedef cublasStatus_t (*minicpmo_cublas_sgemv_fn)(
    cublasHandle_t handle,
    cublasOperation_t trans,
    int m,
    int n,
    const float *alpha,
    const float *a,
    int lda,
    const float *x,
    int incx,
    const float *beta,
    float *y,
    int incy);
typedef cublasStatus_t (*minicpmo_cublas_gemm_ex_fn)(
    cublasHandle_t handle,
    cublasOperation_t transa,
    cublasOperation_t transb,
    int m,
    int n,
    int k,
    const void *alpha,
    const void *a,
    cudaDataType_t atype,
    int lda,
    const void *b,
    cudaDataType_t btype,
    int ldb,
    const void *beta,
    void *c,
    cudaDataType_t ctype,
    int ldc,
    cublasComputeType_t compute_type,
    cublasGemmAlgo_t algo);

typedef nvrtcResult (*minicpmo_nvrtc_create_program_fn)(
    nvrtcProgram *prog,
    const char *src,
    const char *name,
    int num_headers,
    const char *const *headers,
    const char *const *include_names);
typedef nvrtcResult (*minicpmo_nvrtc_compile_program_fn)(nvrtcProgram prog, int num_options, const char *const *options);
typedef nvrtcResult (*minicpmo_nvrtc_get_ptx_size_fn)(nvrtcProgram prog, size_t *ptx_size_ret);
typedef nvrtcResult (*minicpmo_nvrtc_get_ptx_fn)(nvrtcProgram prog, char *ptx);
typedef nvrtcResult (*minicpmo_nvrtc_destroy_program_fn)(nvrtcProgram *prog);
typedef nvrtcResult (*minicpmo_nvrtc_get_program_log_size_fn)(nvrtcProgram prog, size_t *log_size_ret);
typedef nvrtcResult (*minicpmo_nvrtc_get_program_log_fn)(nvrtcProgram prog, char *log);
typedef const char *(*minicpmo_nvrtc_get_error_string_fn)(nvrtcResult result);

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

typedef struct minicpmo_gpu_upload {
    const void *host_ptr;
    const void *staged_host_ptr;
    CUdeviceptr device_ptr;
    size_t bytes;
    size_t rows;
    size_t cols;
    size_t staged_bytes;
    size_t estimated_device_bytes;
    int dtype;
    int upload_dtype;
    int staged_owned;
    int ever_uploaded;
    int persistent_decided;
    int persistent;
    int resident;
    uint64_t last_use_tick;
    char name[128];
} minicpmo_gpu_upload;

typedef struct minicpmo_gpu_slot {
    CUdeviceptr device_ptr;
    size_t bytes;
} minicpmo_gpu_slot;

typedef struct minicpmo_gpu_context {
    void *cuda_lib;
    void *cublas_lib;
    void *nvrtc_lib;
    minicpmo_cuda_init_fn cuda_init;
    minicpmo_cuda_device_get_count_fn cuda_device_get_count;
    minicpmo_cuda_device_get_fn cuda_device_get;
    minicpmo_cuda_device_get_attribute_fn cuda_device_get_attribute;
    minicpmo_cuda_ctx_create_fn cuda_ctx_create;
    minicpmo_cuda_ctx_destroy_fn cuda_ctx_destroy;
    minicpmo_cuda_mem_alloc_fn cuda_mem_alloc;
    minicpmo_cuda_mem_free_fn cuda_mem_free;
    minicpmo_cuda_memcpy_hto_d_fn cuda_memcpy_hto_d;
    minicpmo_cuda_memcpy_dto_d_fn cuda_memcpy_dto_d;
    minicpmo_cuda_memcpy_dto_h_fn cuda_memcpy_dto_h;
    minicpmo_cuda_mem_get_info_fn cuda_mem_get_info;
    minicpmo_cuda_get_error_string_fn cuda_get_error_string;
    minicpmo_cuda_module_load_data_ex_fn cuda_module_load_data_ex;
    minicpmo_cuda_module_get_function_fn cuda_module_get_function;
    minicpmo_cuda_module_unload_fn cuda_module_unload;
    minicpmo_cuda_launch_kernel_fn cuda_launch_kernel;
    minicpmo_cuda_ctx_synchronize_fn cuda_ctx_synchronize;
    minicpmo_cublas_create_fn cublas_create;
    minicpmo_cublas_destroy_fn cublas_destroy;
    minicpmo_cublas_sgemv_fn cublas_sgemv;
    minicpmo_cublas_gemm_ex_fn cublas_gemm_ex;
    minicpmo_nvrtc_create_program_fn nvrtc_create_program;
    minicpmo_nvrtc_compile_program_fn nvrtc_compile_program;
    minicpmo_nvrtc_get_ptx_size_fn nvrtc_get_ptx_size;
    minicpmo_nvrtc_get_ptx_fn nvrtc_get_ptx;
    minicpmo_nvrtc_destroy_program_fn nvrtc_destroy_program;
    minicpmo_nvrtc_get_program_log_size_fn nvrtc_get_program_log_size;
    minicpmo_nvrtc_get_program_log_fn nvrtc_get_program_log;
    minicpmo_nvrtc_get_error_string_fn nvrtc_get_error_string;
    cublasHandle_t cublas_handle;
    CUmodule quant_module;
    CUfunction q4_k_matvec_fn;
    CUfunction q6_k_matvec_fn;
    CUfunction q4_k_matvec_multi_fn;
    CUfunction q6_k_matvec_multi_fn;
    CUfunction q8_1_quantize_fn;
    CUfunction attention_decode_fn;
    CUfunction f32_to_f16_fn;
    CUfunction add_inplace_fn;
    CUfunction silu_mul_fn;
    CUfunction rope_fn;
    CUfunction rms_norm_fn;
    CUfunction head_norm_fn;
    unsigned int device_index;
    CUdevice cuda_device;
    CUcontext cuda_context;
    int compute_capability_major;
    int compute_capability_minor;
    int debug;
    int quant_kernels_attempted;
    int quant_kernels_ready;
    CUdeviceptr scratch_x;
    CUdeviceptr scratch_y;
    size_t scratch_x_bytes;
    size_t scratch_y_bytes;
    CUdeviceptr device_k_cache;
    CUdeviceptr device_v_cache;
    size_t device_kv_bytes;
    size_t kv_layer_count;
    size_t kv_context_capacity;
    size_t kv_dim;
    CUdeviceptr device_rope_cos;
    CUdeviceptr device_rope_sin;
    size_t device_rope_bytes;
    const float *rope_cos_host;
    const float *rope_sin_host;
    size_t rope_cache_context;
    size_t rope_cache_half_dim;
    int rope_cache_valid;
    minicpmo_gpu_upload uploads[MINICPMO_GPU_MAX_UPLOADS];
    size_t upload_count;
    size_t current_device_bytes;
    size_t cache_limit_bytes;
    size_t persistent_limit_bytes;
    size_t persistent_reserved_bytes;
    minicpmo_gpu_slot slots[MINICPMO_GPU_SLOT_COUNT];
    int q8_1_cache_valid;
    unsigned int q8_1_cache_slot;
    size_t q8_1_cache_cols;
    uint64_t use_tick;
    minicpmo_gpu_stats stats;
    char last_error[512];
} minicpmo_gpu_context;

void minicpmo_gpu_context_destroy(void *handle);
static float minicpmo_gpu_f16_to_f32(uint16_t bits);
static float minicpmo_gpu_bf16_to_f32(uint16_t bits);
static uint16_t minicpmo_gpu_f32_to_f16(float value);
static float minicpmo_gpu_sqrtf_positive(float value);
static int minicpmo_gpu_ensure_scratch(
    minicpmo_gpu_context *ctx,
    CUdeviceptr *ptr,
    size_t *capacity,
    size_t required,
    const char *label,
    char *error,
    size_t error_cap);
static int minicpmo_gpu_ensure_slot(
    minicpmo_gpu_context *ctx,
    unsigned int slot,
    size_t required,
    const char *label,
    char *error,
    size_t error_cap);

static uint64_t minicpmo_gpu_now_us(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return ((uint64_t) tv.tv_sec * 1000000ULL) + (uint64_t) tv.tv_usec;
}

static size_t minicpmo_gpu_parse_env_size_mb(const char *name, size_t fallback) {
    const char *value = getenv(name);
    char *end = NULL;
    unsigned long long mb;
    if (value == NULL || value[0] == '\0') {
        return fallback;
    }
    mb = strtoull(value, &end, 10);
    if (end == value || (end != NULL && *end != '\0')) {
        return fallback;
    }
    return (size_t) (mb * 1024ULL * 1024ULL);
}

static int minicpmo_gpu_env_flag_enabled(const char *name, int fallback) {
    const char *value = getenv(name);
    if (value == NULL || value[0] == '\0') {
        return fallback;
    }
    if ((value[0] == '0' && value[1] == '\0') ||
        strcmp(value, "false") == 0 ||
        strcmp(value, "FALSE") == 0 ||
        strcmp(value, "no") == 0 ||
        strcmp(value, "NO") == 0) {
        return 0;
    }
    return 1;
}

static int minicpmo_gpu_device_supports_dp4a(const minicpmo_gpu_context *ctx) {
    if (ctx == NULL) {
        return 0;
    }
    return ctx->compute_capability_major > 6 || (ctx->compute_capability_major == 6 && ctx->compute_capability_minor >= 1);
}

static int minicpmo_gpu_native_kmatvec_requested(const minicpmo_gpu_context *ctx, int dtype) {
    if (dtype != 10 && dtype != 12) {
        return 0;
    }
    if (!minicpmo_gpu_env_flag_enabled("MINICPMO_GPU_EXPERIMENTAL_NATIVE_KMATVEC", 1)) {
        return 0;
    }
    return minicpmo_gpu_device_supports_dp4a(ctx);
}

static size_t minicpmo_gpu_q8_1_super_bytes_for_cols(size_t cols) {
    return (cols / MINICPMO_GPU_QK_K) * MINICPMO_GPU_Q8_1_SUPER_BYTES;
}

static int minicpmo_gpu_f32_to_i8_round(float value) {
    if (value >= 0.0f) {
        return (int) (value + 0.5f);
    }
    return (int) (value - 0.5f);
}

static void minicpmo_gpu_q8_1_quantize_block(uint16_t *d_bits, uint16_t *s_bits, int8_t *qs, const float *src) {
    float amax = 0.0f;
    float sum = 0.0f;
    size_t i;
    for (i = 0; i < MINICPMO_GPU_QK8_1; ++i) {
        float xi = src[i];
        float ax = xi < 0.0f ? -xi : xi;
        if (ax > amax) {
            amax = ax;
        }
        sum += xi;
    }
    if (amax == 0.0f) {
        *d_bits = 0;
        *s_bits = 0;
        for (i = 0; i < MINICPMO_GPU_QK8_1; ++i) {
            qs[i] = 0;
        }
        return;
    }
    {
        float d = amax / 127.0f;
        *d_bits = minicpmo_gpu_f32_to_f16(d);
        *s_bits = minicpmo_gpu_f32_to_f16(sum);
        for (i = 0; i < MINICPMO_GPU_QK8_1; ++i) {
            int q = minicpmo_gpu_f32_to_i8_round(src[i] / d);
            if (q > 127) {
                q = 127;
            }
            if (q < -128) {
                q = -128;
            }
            qs[i] = (int8_t) q;
        }
    }
}

static int minicpmo_gpu_q8_1_quantize_row(unsigned char *dst, const float *src, size_t n) {
    size_t super;
    if (dst == NULL || src == NULL || n == 0 || (n % MINICPMO_GPU_QK_K) != 0) {
        return 0;
    }
    for (super = 0; super < n / MINICPMO_GPU_QK_K; ++super) {
        unsigned char *super_dst = dst + super * MINICPMO_GPU_Q8_1_SUPER_BYTES;
        size_t block;
        for (block = 0; block < MINICPMO_GPU_Q8_1_BLOCKS_PER_SUPER; ++block) {
            unsigned char *block_dst = super_dst + block * MINICPMO_GPU_Q8_1_BLOCK_BYTES;
            uint16_t *d_bits = (uint16_t *) block_dst;
            uint16_t *s_bits = (uint16_t *) (block_dst + sizeof(uint16_t));
            int8_t *qs = (int8_t *) (block_dst + 2u * sizeof(uint16_t));
            minicpmo_gpu_q8_1_quantize_block(
                d_bits,
                s_bits,
                qs,
                &src[super * MINICPMO_GPU_QK_K + block * MINICPMO_GPU_QK8_1]);
        }
    }
    return 1;
}

static void minicpmo_gpu_copy_error(char *dst, size_t dst_cap, const char *src) {
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

static void minicpmo_gpu_set_error(minicpmo_gpu_context *ctx, char *error, size_t error_cap, const char *text) {
    if (ctx != NULL) {
        minicpmo_gpu_copy_error(ctx->last_error, sizeof(ctx->last_error), text);
    }
    minicpmo_gpu_copy_error(error, error_cap, text);
}

static void *minicpmo_gpu_try_dlopen(const char *const *names) {
    size_t i = 0;
    int flags = RTLD_LAZY | RTLD_LOCAL;
#ifdef RTLD_DEEPBIND
    flags |= RTLD_DEEPBIND;
#endif
    while (names[i] != NULL) {
        void *handle = dlopen(names[i], flags);
        if (handle != NULL) {
            return handle;
        }
        ++i;
    }
    return NULL;
}

static int minicpmo_gpu_resolve_symbol(
    minicpmo_gpu_context *ctx,
    void *lib,
    const char *name,
    void **out,
    char *error,
    size_t error_cap) {
    void *fn = NULL;
    if (lib == NULL) {
        minicpmo_gpu_set_error(ctx, error, error_cap, "internal: null dlopen handle");
        return 0;
    }
    dlerror();
    fn = dlsym(lib, name);
    if (fn == NULL) {
        char message[512];
        const char *dlsym_error = dlerror();
        snprintf(message, sizeof(message), "error: missing CUDA symbol %s (%s)", name, dlsym_error != NULL ? dlsym_error : "unknown");
        minicpmo_gpu_set_error(ctx, error, error_cap, message);
        return 0;
    }
    *out = fn;
    return 1;
}

static int minicpmo_gpu_cuda_check(minicpmo_gpu_context *ctx, CUresult code, const char *label, char *error, size_t error_cap) {
    if (code == CUDA_SUCCESS) {
        return 1;
    }
    {
        char message[512];
        const char *detail = "unknown";
        if (ctx != NULL && ctx->cuda_get_error_string != NULL) {
            ctx->cuda_get_error_string(code, &detail);
        }
        snprintf(message, sizeof(message), "error: cuda %s failed code=%d detail=%s", label, code, detail != NULL ? detail : "unknown");
        minicpmo_gpu_set_error(ctx, error, error_cap, message);
    }
    return 0;
}

static int minicpmo_gpu_cublas_check(minicpmo_gpu_context *ctx, cublasStatus_t code, const char *label, char *error, size_t error_cap) {
    if (code == CUBLAS_STATUS_SUCCESS) {
        return 1;
    }
    {
        char message[256];
        snprintf(message, sizeof(message), "error: cublas %s failed status=%d", label, code);
        minicpmo_gpu_set_error(ctx, error, error_cap, message);
    }
    return 0;
}

/*
 * Portions of the native quantized CUDA matvec kernels below are adapted from
 * llama.cpp ggml-cuda (MIT License, Copyright (c) 2023-2026 The ggml authors),
 * primarily ggml/src/ggml-cuda/mmvq.cu and vecdotq.cuh. The surrounding
 * driver/NVRTC bridge remains project-local glue.
 */
static const char *minicpmo_gpu_quant_kernel_source =
    "extern \"C\" __device__ __forceinline__ float minicpmo_half_to_float(unsigned short bits) {\n"
    "    unsigned int sign = ((unsigned int) bits >> 15) & 1u;\n"
    "    unsigned int exp = ((unsigned int) bits >> 10) & 31u;\n"
    "    unsigned int frac = (unsigned int) bits & 1023u;\n"
    "    unsigned int out_bits = 0u;\n"
    "    union { unsigned int u; float f; } out;\n"
    "    if (exp == 0u) {\n"
    "        if (frac == 0u) {\n"
    "            out_bits = sign << 31;\n"
    "        } else {\n"
    "            int sub_exp = -14;\n"
    "            while ((frac & 1024u) == 0u) { frac <<= 1u; sub_exp -= 1; }\n"
    "            frac &= 1023u;\n"
    "            out_bits = (sign << 31) | (((unsigned int) (sub_exp + 127)) << 23) | (frac << 13);\n"
    "        }\n"
    "    } else if (exp == 31u) {\n"
    "        out_bits = (sign << 31) | 0x7f800000u | (frac << 13);\n"
    "    } else {\n"
    "        out_bits = (sign << 31) | ((exp + (127u - 15u)) << 23) | (frac << 13);\n"
    "    }\n"
    "    out.u = out_bits;\n"
    "    return out.f;\n"
    "}\n"
    "extern \"C\" __device__ __forceinline__ unsigned short minicpmo_u16_at(const unsigned char *ptr) {\n"
    "    return (unsigned short) ((unsigned short) ptr[0] | ((unsigned short) ptr[1] << 8));\n"
    "}\n"
    "extern \"C\" __device__ __forceinline__ int minicpmo_i32_at(const void *ptr, int i32) {\n"
    "    const unsigned char *p = ((const unsigned char *) ptr) + 4 * i32;\n"
    "    unsigned int v = ((unsigned int) p[0]) | (((unsigned int) p[1]) << 8) | (((unsigned int) p[2]) << 16) | (((unsigned int) p[3]) << 24);\n"
    "    return (int) v;\n"
    "}\n"
    "extern \"C\" __device__ __forceinline__ int minicpmo_i32_at_b2(const void *ptr, int i32) {\n"
    "    const unsigned char *p = ((const unsigned char *) ptr) + 4 * i32;\n"
    "    unsigned int lo = ((unsigned int) p[0]) | (((unsigned int) p[1]) << 8);\n"
    "    unsigned int hi = ((unsigned int) p[2]) | (((unsigned int) p[3]) << 8);\n"
    "    return (int) (lo | (hi << 16));\n"
    "}\n"
    "extern \"C\" __device__ __forceinline__ int minicpmo_i8x4_dot(int a, int b, int c) {\n"
    "#if __CUDA_ARCH__ >= 610\n"
    "    int r;\n"
    "    asm(\"dp4a.s32.s32 %0, %1, %2, %3;\" : \"=r\"(r) : \"r\"(a), \"r\"(b), \"r\"(c));\n"
    "    return r;\n"
    "#else\n"
    "    const signed char *aa = (const signed char *) &a;\n"
    "    const signed char *bb = (const signed char *) &b;\n"
    "    return c + (int) aa[0] * (int) bb[0] + (int) aa[1] * (int) bb[1] + (int) aa[2] * (int) bb[2] + (int) aa[3] * (int) bb[3];\n"
    "#endif\n"
    "}\n"
    "extern \"C\" __device__ __forceinline__ int minicpmo_vsubss4_i32(int a, int b) {\n"
    "    unsigned int out = 0u;\n"
    "    for (int i = 0; i < 4; ++i) {\n"
    "        int av = (int) ((a >> (8 * i)) & 255);\n"
    "        int bv = (int) ((b >> (8 * i)) & 255);\n"
    "        int sv = av - bv;\n"
    "        signed char sc = (signed char) sv;\n"
    "        out |= ((unsigned int) ((unsigned char) sc)) << (8 * i);\n"
    "    }\n"
    "    return (int) out;\n"
    "}\n"
    "extern \"C\" __device__ __forceinline__ unsigned int minicpmo_q4_k_scale_at(int j, const unsigned char *scales) {\n"
    "    if (j < 4) { return (unsigned int) (scales[j] & 63u); }\n"
    "    return (unsigned int) ((scales[j + 4] & 15u) | ((scales[j - 4] >> 6) << 4));\n"
    "}\n"
    "extern \"C\" __device__ __forceinline__ unsigned int minicpmo_q4_k_min_at(int j, const unsigned char *scales) {\n"
    "    if (j < 4) { return (unsigned int) (scales[j + 4] & 63u); }\n"
    "    return (unsigned int) ((scales[j + 4] >> 4) | ((scales[j] >> 6) << 4));\n"
    "}\n"
    "extern \"C\" __device__ __forceinline__ float minicpmo_q8_1_d_at(const unsigned char *xq_super, int block) {\n"
    "    const unsigned char *xq_block = xq_super + block * 36;\n"
    "    return minicpmo_half_to_float(minicpmo_u16_at(xq_block));\n"
    "}\n"
    "extern \"C\" __device__ __forceinline__ unsigned short minicpmo_float_to_half(float value) {\n"
    "    union { float f; unsigned int u; } in;\n"
    "    unsigned int sign;\n"
    "    unsigned int exp;\n"
    "    unsigned int frac;\n"
    "    unsigned short out;\n"
    "    in.f = value;\n"
    "    sign = (in.u >> 16) & 0x8000u;\n"
    "    exp = (in.u >> 23) & 0xffu;\n"
    "    frac = in.u & 0x7fffffu;\n"
    "    if (exp == 255u) {\n"
    "        if (frac != 0u) { return (unsigned short) (sign | 0x7e00u); }\n"
    "        return (unsigned short) (sign | 0x7c00u);\n"
    "    }\n"
    "    if (exp > 142u) { return (unsigned short) (sign | 0x7c00u); }\n"
    "    if (exp < 113u) {\n"
    "        if (exp < 103u) { return (unsigned short) sign; }\n"
    "        frac |= 0x800000u;\n"
    "        out = (unsigned short) (sign | (frac >> (126u - exp)));\n"
    "        if ((frac >> (125u - exp)) & 1u) { out = (unsigned short) (out + 1u); }\n"
    "        return out;\n"
    "    }\n"
    "    out = (unsigned short) (sign | ((exp - 112u) << 10) | (frac >> 13));\n"
    "    if (frac & 0x1000u) { out = (unsigned short) (out + 1u); }\n"
    "    return out;\n"
    "}\n"
    "extern \"C\" __device__ __forceinline__ float minicpmo_vec_weight_at(const unsigned char *weight, int dtype, int idx) {\n"
    "    if (dtype == 0) {\n"
    "        return ((const float *) weight)[idx];\n"
    "    }\n"
    "    return minicpmo_half_to_float(((const unsigned short *) weight)[idx]);\n"
    "}\n"
    "extern \"C\" __global__ void minicpmo_q8_1_quantize(const float *src, unsigned char *dst, int cols) {\n"
    "    int super = (int) blockIdx.x;\n"
    "    int tid = (int) threadIdx.x;\n"
    "    int idx = super * 256 + tid;\n"
    "    int lane = tid & 31;\n"
    "    int group = tid >> 5;\n"
    "    float x = 0.0f;\n"
    "    unsigned char *dst_block = dst + (unsigned long long) super * 288ull + (unsigned long long) group * 36ull;\n"
    "    __shared__ float xv[256];\n"
    "    __shared__ float av[256];\n"
    "    __shared__ float sv[256];\n"
    "    __shared__ float d_shared[8];\n"
    "    if (idx >= cols) { return; }\n"
    "    x = src[idx];\n"
    "    xv[tid] = x;\n"
    "    av[tid] = x < 0.0f ? -x : x;\n"
    "    sv[tid] = x;\n"
    "    __syncthreads();\n"
    "    for (int stride = 16; stride > 0; stride >>= 1) {\n"
    "        if (lane < stride) {\n"
    "            float other_abs = av[tid + stride];\n"
    "            av[tid] = av[tid] > other_abs ? av[tid] : other_abs;\n"
    "            sv[tid] += sv[tid + stride];\n"
    "        }\n"
    "        __syncthreads();\n"
    "    }\n"
    "    if (lane == 0) {\n"
    "        float d = av[tid] > 0.0f ? av[tid] / 127.0f : 0.0f;\n"
    "        d_shared[group] = d;\n"
    "        ((unsigned short *) dst_block)[0] = minicpmo_float_to_half(d);\n"
    "        ((unsigned short *) (dst_block + 2))[0] = minicpmo_float_to_half(sv[tid]);\n"
    "    }\n"
    "    __syncthreads();\n"
    "    {\n"
    "        float d = d_shared[group];\n"
    "        int q = 0;\n"
    "        if (d > 0.0f) {\n"
    "            float scaled = xv[tid] / d;\n"
    "            q = scaled >= 0.0f ? (int) (scaled + 0.5f) : (int) (scaled - 0.5f);\n"
    "            if (q > 127) { q = 127; }\n"
    "            if (q < -128) { q = -128; }\n"
    "        }\n"
    "        ((signed char *) (dst_block + 4))[lane] = (signed char) q;\n"
    "    }\n"
    "}\n"
    "extern \"C\" __global__ void minicpmo_f32_to_f16(const float *src, unsigned short *dst, int n) {\n"
    "    int idx = (int) blockIdx.x * (int) blockDim.x + (int) threadIdx.x;\n"
    "    if (idx >= n) { return; }\n"
    "    dst[idx] = minicpmo_float_to_half(src[idx]);\n"
    "}\n"
    "extern \"C\" __global__ void minicpmo_add_inplace(float *dst, const float *src, int n) {\n"
    "    int idx = (int) blockIdx.x * (int) blockDim.x + (int) threadIdx.x;\n"
    "    if (idx >= n) { return; }\n"
    "    dst[idx] += src[idx];\n"
    "}\n"
    "extern \"C\" __global__ void minicpmo_silu_mul(float *dst, const float *gate, const float *up, int n) {\n"
    "    int idx = (int) blockIdx.x * (int) blockDim.x + (int) threadIdx.x;\n"
    "    if (idx >= n) { return; }\n"
    "    float gv = gate[idx];\n"
    "    dst[idx] = (gv / (1.0f + expf(-gv))) * up[idx];\n"
    "}\n"
    "extern \"C\" __global__ void minicpmo_apply_rope(float *vec, int head_count, int head_dim, int pos, float freq_base, float freq_scale, const float *rope_cos, const float *rope_sin) {\n"
    "    int h = (int) blockIdx.x;\n"
    "    int d = (int) threadIdx.x;\n"
    "    int half_dim = head_dim / 2;\n"
    "    int base = h * head_dim;\n"
    "    if (h >= head_count || d >= half_dim) { return; }\n"
    "    {\n"
    "        float c;\n"
    "        float s;\n"
    "        if (rope_cos != 0 && rope_sin != 0) {\n"
    "            int rope_index = pos * half_dim + d;\n"
    "            c = rope_cos[rope_index];\n"
    "            s = rope_sin[rope_index];\n"
    "        } else {\n"
    "            float theta = 0.0f;\n"
    "            if (pos != 0) {\n"
    "                float exponent = ((float) (2 * d)) / ((float) head_dim);\n"
    "                float log_base = freq_base > 0.0f ? logf(freq_base) : 0.0f;\n"
    "                theta = ((float) pos) * freq_scale * expf(-log_base * exponent);\n"
    "            }\n"
    "            c = cosf(theta);\n"
    "            s = sinf(theta);\n"
    "        }\n"
    "        float x0 = vec[base + d];\n"
    "        float x1 = vec[base + half_dim + d];\n"
    "        vec[base + d] = x0 * c - x1 * s;\n"
    "        vec[base + half_dim + d] = x1 * c + x0 * s;\n"
    "    }\n"
    "}\n"
    "extern \"C\" __global__ void minicpmo_rms_norm(float *dst, const float *src, const unsigned char *weight, int weight_dtype, int n, float eps) {\n"
    "    int tid = (int) threadIdx.x;\n"
    "    __shared__ float reduce_buf[256];\n"
    "    __shared__ float inv_shared;\n"
    "    float sum = 0.0f;\n"
    "    for (int i = tid; i < n; i += (int) blockDim.x) {\n"
    "        float v = src[i];\n"
    "        sum += v * v;\n"
    "    }\n"
    "    reduce_buf[tid] = sum;\n"
    "    __syncthreads();\n"
    "    for (int stride = (int) blockDim.x / 2; stride > 0; stride >>= 1) {\n"
    "        if (tid < stride) {\n"
    "            reduce_buf[tid] += reduce_buf[tid + stride];\n"
    "        }\n"
    "        __syncthreads();\n"
    "    }\n"
    "    if (tid == 0) {\n"
    "        inv_shared = rsqrtf(reduce_buf[0] / (float) n + eps);\n"
    "    }\n"
    "    __syncthreads();\n"
    "    for (int i = tid; i < n; i += (int) blockDim.x) {\n"
    "        dst[i] = src[i] * inv_shared * minicpmo_vec_weight_at(weight, weight_dtype, i);\n"
    "    }\n"
    "}\n"
    "extern \"C\" __global__ void minicpmo_head_norm(float *vec, const unsigned char *weight, int weight_dtype, int head_dim) {\n"
    "    int h = (int) blockIdx.x;\n"
    "    int tid = (int) threadIdx.x;\n"
    "    int base = h * head_dim;\n"
    "    __shared__ float reduce_buf[256];\n"
    "    __shared__ float inv_shared;\n"
    "    float sum = 0.0f;\n"
    "    if (tid < head_dim) {\n"
    "        float v = vec[base + tid];\n"
    "        sum = v * v;\n"
    "    }\n"
    "    reduce_buf[tid] = sum;\n"
    "    __syncthreads();\n"
    "    for (int stride = (int) blockDim.x / 2; stride > 0; stride >>= 1) {\n"
    "        if (tid < stride) {\n"
    "            reduce_buf[tid] += reduce_buf[tid + stride];\n"
    "        }\n"
    "        __syncthreads();\n"
    "    }\n"
    "    if (tid == 0) {\n"
    "        inv_shared = rsqrtf(reduce_buf[0] / (float) head_dim + 0.000001f);\n"
    "    }\n"
    "    __syncthreads();\n"
    "    if (tid < head_dim) {\n"
    "        vec[base + tid] = vec[base + tid] * inv_shared * minicpmo_vec_weight_at(weight, weight_dtype, tid);\n"
    "    }\n"
    "}\n"
    "extern \"C\" __device__ __forceinline__ float minicpmo_q4_k_vecdot_llama(const unsigned char *row_base, const unsigned char *xq, int kbx, int iqs) {\n"
    "    const unsigned char *bq4 = row_base + (unsigned long long) kbx * 144ull;\n"
    "    const unsigned char *xq_super = xq + (unsigned long long) kbx * 288ull;\n"
    "    const unsigned char *qs = bq4 + 16;\n"
    "    int v0;\n"
    "    int v1;\n"
    "    int u0;\n"
    "    int u1;\n"
    "    int u2;\n"
    "    int u3;\n"
    "    float d8_0;\n"
    "    float d8_1;\n"
    "    unsigned short aux0;\n"
    "    unsigned short aux1;\n"
    "    unsigned char sc0;\n"
    "    unsigned char sc1;\n"
    "    unsigned char m0;\n"
    "    unsigned char m1;\n"
    "    const int bq8_offset = 2 * ((iqs / 2) / 4);\n"
    "    const int q4_offset = 16 * bq8_offset + 4 * ((iqs / 2) % 4);\n"
    "    const int j = bq8_offset / 2;\n"
    "    const unsigned char *scales = bq4 + 4;\n"
    "    if (j < 2) {\n"
    "        aux0 = (unsigned short) (minicpmo_u16_at(scales + 2 * (j + 0)) & 0x3f3fu);\n"
    "        aux1 = (unsigned short) (minicpmo_u16_at(scales + 2 * (j + 2)) & 0x3f3fu);\n"
    "    } else {\n"
    "        aux0 = (unsigned short) (((minicpmo_u16_at(scales + 2 * (j + 2)) >> 0) & 0x0f0fu) | ((minicpmo_u16_at(scales + 2 * (j - 2)) & 0xc0c0u) >> 2));\n"
    "        aux1 = (unsigned short) (((minicpmo_u16_at(scales + 2 * (j + 2)) >> 4) & 0x0f0fu) | ((minicpmo_u16_at(scales + 2 * (j - 0)) & 0xc0c0u) >> 2));\n"
    "    }\n"
    "    sc0 = (unsigned char) (aux0 & 255u);\n"
    "    sc1 = (unsigned char) (aux0 >> 8);\n"
    "    m0 = (unsigned char) (aux1 & 255u);\n"
    "    m1 = (unsigned char) (aux1 >> 8);\n"
    "    v0 = minicpmo_i32_at(qs + q4_offset, 0);\n"
    "    v1 = minicpmo_i32_at(qs + q4_offset + 16, 0);\n"
    "    {\n"
    "        const unsigned char *bq8 = xq_super + bq8_offset * 36;\n"
    "        const unsigned char *q8 = bq8 + 4 + 4 * ((iqs / 2) % 4);\n"
    "        d8_0 = minicpmo_half_to_float(minicpmo_u16_at(bq8));\n"
    "        u0 = minicpmo_i32_at(q8, 0);\n"
    "        u1 = minicpmo_i32_at(q8 + 16, 0);\n"
    "    }\n"
    "    {\n"
    "        const unsigned char *bq8 = xq_super + (bq8_offset + 1) * 36;\n"
    "        const unsigned char *q8 = bq8 + 4 + 4 * ((iqs / 2) % 4);\n"
    "        d8_1 = minicpmo_half_to_float(minicpmo_u16_at(bq8));\n"
    "        u2 = minicpmo_i32_at(q8, 0);\n"
    "        u3 = minicpmo_i32_at(q8 + 16, 0);\n"
    "    }\n"
    "    {\n"
    "        const int v00 = (v0 >> 0) & 0x0f0f0f0f;\n"
    "        const int v10 = (v1 >> 0) & 0x0f0f0f0f;\n"
    "        const int v01 = (v0 >> 4) & 0x0f0f0f0f;\n"
    "        const int v11 = (v1 >> 4) & 0x0f0f0f0f;\n"
    "        const int dot_d0 = minicpmo_i8x4_dot(v10, u1, minicpmo_i8x4_dot(v00, u0, 0));\n"
    "        const int dot_m0 = minicpmo_i8x4_dot(0x01010101, u1, minicpmo_i8x4_dot(0x01010101, u0, 0));\n"
    "        const int dot_d1 = minicpmo_i8x4_dot(v11, u3, minicpmo_i8x4_dot(v01, u2, 0));\n"
    "        const int dot_m1 = minicpmo_i8x4_dot(0x01010101, u3, minicpmo_i8x4_dot(0x01010101, u2, 0));\n"
    "        const float d = minicpmo_half_to_float(minicpmo_u16_at(bq4));\n"
    "        const float dmin = minicpmo_half_to_float(minicpmo_u16_at(bq4 + 2));\n"
    "        return d * (d8_0 * (float) dot_d0 * (float) sc0 + d8_1 * (float) dot_d1 * (float) sc1) - dmin * (d8_0 * (float) dot_m0 * (float) m0 + d8_1 * (float) dot_m1 * (float) m1);\n"
    "    }\n"
    "}\n"
    "extern \"C\" __device__ __forceinline__ float minicpmo_q4_k_dot_row(const unsigned char *weights, const unsigned char *xq, int row, int cols) {\n"
    "    int tid = (int) threadIdx.x;\n"
    "    int blocks_per_row = cols / 256;\n"
    "    int iqs = 2 * (tid % 16);\n"
    "    int kbx = tid / 16;\n"
    "    float sum = 0.0f;\n"
    "    const unsigned char *row_base = weights + (unsigned long long) row * (unsigned long long) blocks_per_row * 144ull;\n"
    "    for (; kbx < blocks_per_row; kbx += 8) {\n"
    "        sum += minicpmo_q4_k_vecdot_llama(row_base, xq, kbx, iqs);\n"
    "    }\n"
    "    return sum;\n"
    "}\n"
    "extern \"C\" __global__ void minicpmo_q4_k_matvec_multi(const unsigned char *w0, const unsigned char *w1, const unsigned char *w2, const unsigned char *xq, float *d0, float *d1, float *d2, int rows0, int rows1, int rows2, int cols, int count) {\n"
    "    int linear = (int) blockIdx.x;\n"
    "    int row = linear;\n"
    "    int tid = (int) threadIdx.x;\n"
    "    const unsigned char *weights = w0;\n"
    "    float *dst = d0;\n"
    "    int limit1 = rows0 + rows1;\n"
    "    __shared__ float reduce_buf[128];\n"
    "    if (linear >= rows0) {\n"
    "        if (count > 1 && linear < limit1) { weights = w1; dst = d1; row = linear - rows0; }\n"
    "        else if (count > 2 && linear < limit1 + rows2) { weights = w2; dst = d2; row = linear - limit1; }\n"
    "        else { return; }\n"
    "    }\n"
    "    if (weights == 0 || dst == 0) { return; }\n"
    "    reduce_buf[tid] = minicpmo_q4_k_dot_row(weights, xq, row, cols);\n"
    "    __syncthreads();\n"
    "    for (int stride = (int) blockDim.x / 2; stride > 0; stride >>= 1) {\n"
    "        if (tid < stride) { reduce_buf[tid] += reduce_buf[tid + stride]; }\n"
    "        __syncthreads();\n"
    "    }\n"
    "    if (tid == 0) { dst[row] = reduce_buf[0]; }\n"
    "}\n"
    "extern \"C\" __global__ void minicpmo_q4_k_matvec(const unsigned char *weights, const unsigned char *xq, float *dst, int rows, int cols) {\n"
    "    int row = (int) blockIdx.x;\n"
    "    int tid = (int) threadIdx.x;\n"
    "    __shared__ float reduce_buf[128];\n"
    "    if (row >= rows) { return; }\n"
    "    reduce_buf[tid] = minicpmo_q4_k_dot_row(weights, xq, row, cols);\n"
    "    __syncthreads();\n"
    "    for (int stride = (int) blockDim.x / 2; stride > 0; stride >>= 1) {\n"
    "        if (tid < stride) { reduce_buf[tid] += reduce_buf[tid + stride]; }\n"
    "        __syncthreads();\n"
    "    }\n"
    "    if (tid == 0) { dst[row] = reduce_buf[0]; }\n"
    "}\n"
    "extern \"C\" __device__ __forceinline__ float minicpmo_q6_k_vecdot_llama(const unsigned char *row_base, const unsigned char *xq, int kbx, int iqs) {\n"
    "    const unsigned char *bq6 = row_base + (unsigned long long) kbx * 210ull;\n"
    "    const unsigned char *xq_super = xq + (unsigned long long) kbx * 288ull;\n"
    "    const int bq8_offset = 4 * (iqs / 16) + ((iqs % 16) / 8);\n"
    "    const int scale_offset = 8 * (iqs / 16) + ((iqs % 16) / 4);\n"
    "    const int vh_shift = 2 * ((iqs % 16) / 8);\n"
    "    const int vl = minicpmo_i32_at_b2(bq6, iqs);\n"
    "    const int vh = minicpmo_i32_at_b2(bq6 + 128, 8 * (iqs / 16) + (iqs % 8)) >> vh_shift;\n"
    "    const signed char *scales = (const signed char *) (bq6 + 192 + scale_offset);\n"
    "    const float d = minicpmo_half_to_float(minicpmo_u16_at(bq6 + 208));\n"
    "    float sumf = 0.0f;\n"
    "    for (int i = 0; i < 2; ++i) {\n"
    "        const unsigned char *bq8 = xq_super + (bq8_offset + 2 * i) * 36;\n"
    "        const int u = minicpmo_i32_at(bq8 + 4, iqs % 8);\n"
    "        const float d8 = minicpmo_half_to_float(minicpmo_u16_at(bq8));\n"
    "        const int sc = (int) scales[4 * i];\n"
    "        const int vil = (vl >> (4 * i)) & 0x0f0f0f0f;\n"
    "        const int vih = ((vh >> (4 * i)) << 4) & 0x30303030;\n"
    "        const int vi = minicpmo_vsubss4_i32(vil | vih, 0x20202020);\n"
    "        sumf += d8 * ((float) minicpmo_i8x4_dot(vi, u, 0) * (float) sc);\n"
    "    }\n"
    "    return d * sumf;\n"
    "}\n"
    "extern \"C\" __device__ __forceinline__ float minicpmo_q6_k_dot_row(const unsigned char *weights, const unsigned char *xq, int row, int cols) {\n"
    "    int tid = (int) threadIdx.x;\n"
    "    int blocks_per_row = cols / 256;\n"
    "    int iqs = tid % 32;\n"
    "    int kbx = tid / 32;\n"
    "    float sum = 0.0f;\n"
    "    const unsigned char *row_base = weights + (unsigned long long) row * (unsigned long long) blocks_per_row * 210ull;\n"
    "    for (; kbx < blocks_per_row; kbx += 4) {\n"
    "        sum += minicpmo_q6_k_vecdot_llama(row_base, xq, kbx, iqs);\n"
    "    }\n"
    "    return sum;\n"
    "}\n"
    "extern \"C\" __global__ void minicpmo_q6_k_matvec_multi(const unsigned char *w0, const unsigned char *w1, const unsigned char *w2, const unsigned char *xq, float *d0, float *d1, float *d2, int rows0, int rows1, int rows2, int cols, int count) {\n"
    "    int linear = (int) blockIdx.x;\n"
    "    int row = linear;\n"
    "    int tid = (int) threadIdx.x;\n"
    "    const unsigned char *weights = w0;\n"
    "    float *dst = d0;\n"
    "    int limit1 = rows0 + rows1;\n"
    "    __shared__ float reduce_buf[128];\n"
    "    if (linear >= rows0) {\n"
    "        if (count > 1 && linear < limit1) { weights = w1; dst = d1; row = linear - rows0; }\n"
    "        else if (count > 2 && linear < limit1 + rows2) { weights = w2; dst = d2; row = linear - limit1; }\n"
    "        else { return; }\n"
    "    }\n"
    "    if (weights == 0 || dst == 0) { return; }\n"
    "    reduce_buf[tid] = minicpmo_q6_k_dot_row(weights, xq, row, cols);\n"
    "    __syncthreads();\n"
    "    for (int stride = (int) blockDim.x / 2; stride > 0; stride >>= 1) {\n"
    "        if (tid < stride) { reduce_buf[tid] += reduce_buf[tid + stride]; }\n"
    "        __syncthreads();\n"
    "    }\n"
    "    if (tid == 0) { dst[row] = reduce_buf[0]; }\n"
    "}\n"
    "extern \"C\" __global__ void minicpmo_q6_k_matvec(const unsigned char *weights, const unsigned char *xq, float *dst, int rows, int cols) {\n"
    "    int row = (int) blockIdx.x;\n"
    "    int tid = (int) threadIdx.x;\n"
    "    __shared__ float reduce_buf[128];\n"
    "    if (row >= rows) { return; }\n"
    "    reduce_buf[tid] = minicpmo_q6_k_dot_row(weights, xq, row, cols);\n"
    "    __syncthreads();\n"
    "    for (int stride = (int) blockDim.x / 2; stride > 0; stride >>= 1) {\n"
    "        if (tid < stride) { reduce_buf[tid] += reduce_buf[tid + stride]; }\n"
    "        __syncthreads();\n"
    "    }\n"
    "    if (tid == 0) { dst[row] = reduce_buf[0]; }\n"
    "}\n"
    "extern \"C\" __global__ void minicpmo_qwen3_attention_decode(const float *q, const float *k_cache, const float *v_cache, float *dst, int layer, int pos, int head_count, int kv_head_count, int head_dim, int context_capacity, float scale) {\n"
    "    int h = (int) blockIdx.x;\n"
    "    int tid = (int) threadIdx.x;\n"
    "    int kv_dim = kv_head_count * head_dim;\n"
    "    int out_base = h * head_dim;\n"
    "    int kvh = (h * kv_head_count) / head_count;\n"
    "    __shared__ float reduce_buf[256];\n"
    "    __shared__ float alpha_shared;\n"
    "    __shared__ float beta_shared;\n"
    "    __shared__ float l_shared;\n"
    "    float acc = 0.0f;\n"
    "    float qv = 0.0f;\n"
    "    float m = -3.402823466e+38f;\n"
    "    float l = 0.0f;\n"
    "    if (h >= head_count) { return; }\n"
    "    if (tid < head_dim) {\n"
    "        qv = q[out_base + tid];\n"
    "    }\n"
    "    for (int t = 0; t <= pos; ++t) {\n"
    "        float partial = 0.0f;\n"
    "        if (tid < head_dim) {\n"
    "            unsigned long long kv_offset = (((unsigned long long) layer * (unsigned long long) context_capacity) + (unsigned long long) t) * (unsigned long long) kv_dim + (unsigned long long) kvh * (unsigned long long) head_dim + (unsigned long long) tid;\n"
    "            partial = qv * k_cache[kv_offset];\n"
    "        }\n"
    "        reduce_buf[tid] = partial;\n"
    "        __syncthreads();\n"
    "        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {\n"
    "            if (tid < stride) {\n"
    "                reduce_buf[tid] += reduce_buf[tid + stride];\n"
    "            }\n"
    "            __syncthreads();\n"
    "        }\n"
    "        if (tid == 0) {\n"
    "            float score = reduce_buf[0] * scale;\n"
    "            float m_new = m > score ? m : score;\n"
    "            float alpha = l > 0.0f ? expf(m - m_new) : 0.0f;\n"
    "            float beta = expf(score - m_new);\n"
    "            l = l * alpha + beta;\n"
    "            m = m_new;\n"
    "            alpha_shared = alpha;\n"
    "            beta_shared = beta;\n"
    "            l_shared = l;\n"
    "        }\n"
    "        __syncthreads();\n"
    "        if (tid < head_dim) {\n"
    "            unsigned long long kv_offset = (((unsigned long long) layer * (unsigned long long) context_capacity) + (unsigned long long) t) * (unsigned long long) kv_dim + (unsigned long long) kvh * (unsigned long long) head_dim + (unsigned long long) tid;\n"
    "            acc = acc * alpha_shared + beta_shared * v_cache[kv_offset];\n"
    "        }\n"
    "        __syncthreads();\n"
    "    }\n"
    "    if (tid < head_dim) {\n"
    "        dst[out_base + tid] = l_shared > 0.0f ? acc / l_shared : 0.0f;\n"
    "    }\n"
    "}\n";

static int minicpmo_gpu_nvrtc_check(minicpmo_gpu_context *ctx, nvrtcResult code, const char *label, char *error, size_t error_cap) {
    if (code == NVRTC_SUCCESS) {
        return 1;
    }
    {
        char message[512];
        const char *detail = "unknown";
        if (ctx != NULL && ctx->nvrtc_get_error_string != NULL) {
            detail = ctx->nvrtc_get_error_string(code);
        }
        snprintf(message, sizeof(message), "error: nvrtc %s failed code=%d detail=%s", label, code, detail != NULL ? detail : "unknown");
        minicpmo_gpu_set_error(ctx, error, error_cap, message);
    }
    return 0;
}

static void minicpmo_gpu_nvrtc_log(
    minicpmo_gpu_context *ctx,
    nvrtcProgram program,
    char *dst,
    size_t dst_cap,
    const char *prefix) {
    size_t log_size = 0;
    char *log = NULL;
    if (ctx == NULL || ctx->nvrtc_get_program_log_size == NULL || ctx->nvrtc_get_program_log == NULL || dst == NULL || dst_cap == 0) {
        return;
    }
    if (ctx->nvrtc_get_program_log_size(program, &log_size) != NVRTC_SUCCESS || log_size <= 1) {
        return;
    }
    log = (char *) malloc(log_size);
    if (log == NULL) {
        return;
    }
    if (ctx->nvrtc_get_program_log(program, log) == NVRTC_SUCCESS) {
        snprintf(dst, dst_cap, "%s%s", prefix, log);
    }
    free(log);
}

static void minicpmo_gpu_nvrtc_arch_option(const minicpmo_gpu_context *ctx, char *dst, size_t dst_cap) {
    int major = ctx != NULL ? ctx->compute_capability_major : 0;
    int minor = ctx != NULL ? ctx->compute_capability_minor : 0;
    if (dst == NULL || dst_cap == 0) {
        return;
    }
    if (major > 6 || (major == 6 && minor >= 1)) {
        snprintf(dst, dst_cap, "--gpu-architecture=compute_61");
        return;
    }
    if (major >= 5) {
        snprintf(dst, dst_cap, "--gpu-architecture=compute_%d%d", major, minor);
        return;
    }
    snprintf(dst, dst_cap, "--gpu-architecture=compute_50");
}

static int minicpmo_gpu_prepare_quant_runtime(minicpmo_gpu_context *ctx, char *error, size_t error_cap) {
    static const char *const nvrtc_names[] = {
        "libnvrtc.so.12",
        "libnvrtc.so",
        NULL,
    };
    if (ctx->nvrtc_lib == NULL) {
        ctx->nvrtc_lib = minicpmo_gpu_try_dlopen(nvrtc_names);
        if (ctx->nvrtc_lib == NULL) {
            minicpmo_gpu_set_error(ctx, error, error_cap, "error: unable to load libnvrtc.so; install NVIDIA NVRTC runtime for native Q4_K/Q6_K matvec");
            return 0;
        }
    }
    if ((ctx->nvrtc_create_program == NULL &&
         !minicpmo_gpu_resolve_symbol(ctx, ctx->nvrtc_lib, "nvrtcCreateProgram", (void **) &ctx->nvrtc_create_program, error, error_cap)) ||
        (ctx->nvrtc_compile_program == NULL &&
         !minicpmo_gpu_resolve_symbol(ctx, ctx->nvrtc_lib, "nvrtcCompileProgram", (void **) &ctx->nvrtc_compile_program, error, error_cap)) ||
        (ctx->nvrtc_get_ptx_size == NULL &&
         !minicpmo_gpu_resolve_symbol(ctx, ctx->nvrtc_lib, "nvrtcGetPTXSize", (void **) &ctx->nvrtc_get_ptx_size, error, error_cap)) ||
        (ctx->nvrtc_get_ptx == NULL &&
         !minicpmo_gpu_resolve_symbol(ctx, ctx->nvrtc_lib, "nvrtcGetPTX", (void **) &ctx->nvrtc_get_ptx, error, error_cap)) ||
        (ctx->nvrtc_destroy_program == NULL &&
         !minicpmo_gpu_resolve_symbol(ctx, ctx->nvrtc_lib, "nvrtcDestroyProgram", (void **) &ctx->nvrtc_destroy_program, error, error_cap)) ||
        (ctx->nvrtc_get_program_log_size == NULL &&
         !minicpmo_gpu_resolve_symbol(ctx, ctx->nvrtc_lib, "nvrtcGetProgramLogSize", (void **) &ctx->nvrtc_get_program_log_size, error, error_cap)) ||
        (ctx->nvrtc_get_program_log == NULL &&
         !minicpmo_gpu_resolve_symbol(ctx, ctx->nvrtc_lib, "nvrtcGetProgramLog", (void **) &ctx->nvrtc_get_program_log, error, error_cap)) ||
        (ctx->nvrtc_get_error_string == NULL &&
         !minicpmo_gpu_resolve_symbol(ctx, ctx->nvrtc_lib, "nvrtcGetErrorString", (void **) &ctx->nvrtc_get_error_string, error, error_cap))) {
        return 0;
    }
    if ((ctx->cuda_module_load_data_ex == NULL &&
         !minicpmo_gpu_resolve_symbol(ctx, ctx->cuda_lib, "cuModuleLoadDataEx", (void **) &ctx->cuda_module_load_data_ex, error, error_cap)) ||
        (ctx->cuda_module_get_function == NULL &&
         !minicpmo_gpu_resolve_symbol(ctx, ctx->cuda_lib, "cuModuleGetFunction", (void **) &ctx->cuda_module_get_function, error, error_cap)) ||
        (ctx->cuda_module_unload == NULL &&
         !minicpmo_gpu_resolve_symbol(ctx, ctx->cuda_lib, "cuModuleUnload", (void **) &ctx->cuda_module_unload, error, error_cap)) ||
        (ctx->cuda_launch_kernel == NULL &&
         !minicpmo_gpu_resolve_symbol(ctx, ctx->cuda_lib, "cuLaunchKernel", (void **) &ctx->cuda_launch_kernel, error, error_cap)) ||
        (ctx->cuda_ctx_synchronize == NULL &&
         !minicpmo_gpu_resolve_symbol(ctx, ctx->cuda_lib, "cuCtxSynchronize", (void **) &ctx->cuda_ctx_synchronize, error, error_cap))) {
        return 0;
    }
    return 1;
}

static int minicpmo_gpu_prepare_quant_kernels(minicpmo_gpu_context *ctx, char *error, size_t error_cap) {
    nvrtcProgram program = NULL;
    char *ptx = NULL;
    size_t ptx_size = 0;
    char arch_option[64];
    const char *options[] = {
        arch_option,
        "--std=c++11",
        "--fmad=false",
    };
    if (ctx == NULL) {
        minicpmo_gpu_copy_error(error, error_cap, "error: null gpu quant kernel context");
        return 0;
    }
    if (ctx->quant_kernels_ready) {
        return 1;
    }
    if (ctx->quant_kernels_attempted) {
        minicpmo_gpu_copy_error(error, error_cap, ctx->last_error);
        return 0;
    }
    ctx->quant_kernels_attempted = 1;
    if (!minicpmo_gpu_prepare_quant_runtime(ctx, error, error_cap)) {
        return 0;
    }
    minicpmo_gpu_nvrtc_arch_option(ctx, arch_option, sizeof(arch_option));
    if (!minicpmo_gpu_nvrtc_check(ctx, ctx->nvrtc_create_program(&program, minicpmo_gpu_quant_kernel_source, "minicpmo_quant_kernels.cu", 0, NULL, NULL), "create_program", error, error_cap)) {
        return 0;
    }
    if (!minicpmo_gpu_nvrtc_check(ctx, ctx->nvrtc_compile_program(program, (int) (sizeof(options) / sizeof(options[0])), options), "compile_program", error, error_cap)) {
        minicpmo_gpu_nvrtc_log(ctx, program, error, error_cap, "error: nvrtc compile log: ");
        ctx->nvrtc_destroy_program(&program);
        minicpmo_gpu_copy_error(ctx->last_error, sizeof(ctx->last_error), error);
        return 0;
    }
    if (!minicpmo_gpu_nvrtc_check(ctx, ctx->nvrtc_get_ptx_size(program, &ptx_size), "get_ptx_size", error, error_cap)) {
        ctx->nvrtc_destroy_program(&program);
        return 0;
    }
    ptx = (char *) malloc(ptx_size);
    if (ptx == NULL) {
        ctx->nvrtc_destroy_program(&program);
        minicpmo_gpu_set_error(ctx, error, error_cap, "error: gpu ptx allocation failed");
        return 0;
    }
    if (!minicpmo_gpu_nvrtc_check(ctx, ctx->nvrtc_get_ptx(program, ptx), "get_ptx", error, error_cap)) {
        free(ptx);
        ctx->nvrtc_destroy_program(&program);
        return 0;
    }
    ctx->nvrtc_destroy_program(&program);
    {
        enum {
            MINICPMO_CU_JIT_INFO_LOG_BUFFER = 3,
            MINICPMO_CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES = 4,
            MINICPMO_CU_JIT_ERROR_LOG_BUFFER = 5,
            MINICPMO_CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES = 6,
            MINICPMO_CU_JIT_LOG_VERBOSE = 12,
        };
        int jit_options[5];
        void *jit_values[5];
        char jit_info[4096];
        char jit_error[4096];
        CUresult load_result;
        jit_info[0] = '\0';
        jit_error[0] = '\0';
        jit_options[0] = MINICPMO_CU_JIT_INFO_LOG_BUFFER;
        jit_options[1] = MINICPMO_CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES;
        jit_options[2] = MINICPMO_CU_JIT_ERROR_LOG_BUFFER;
        jit_options[3] = MINICPMO_CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES;
        jit_options[4] = MINICPMO_CU_JIT_LOG_VERBOSE;
        jit_values[0] = jit_info;
        jit_values[1] = (void *) (uintptr_t) sizeof(jit_info);
        jit_values[2] = jit_error;
        jit_values[3] = (void *) (uintptr_t) sizeof(jit_error);
        jit_values[4] = (void *) (uintptr_t) 1u;
        load_result = ctx->cuda_module_load_data_ex(&ctx->quant_module, ptx, 5, jit_options, jit_values);
        if (load_result != CUDA_SUCCESS) {
            char message[512];
            const char *detail = "unknown";
            if (ctx->cuda_get_error_string != NULL) {
                ctx->cuda_get_error_string(load_result, &detail);
            }
            snprintf(message, sizeof(message), "error: cuda module_load_data_ex failed code=%d detail=%s jit_error=%s", load_result, detail != NULL ? detail : "unknown", jit_error[0] != '\0' ? jit_error : "(empty)");
            minicpmo_gpu_set_error(ctx, error, error_cap, message);
            free(ptx);
            return 0;
        }
        if (ctx->debug && jit_info[0] != '\0') {
            fprintf(stderr, "minicpmo-gpu: jit info: %s\n", jit_info);
        }
    }
    if (!minicpmo_gpu_cuda_check(ctx, ctx->cuda_module_get_function(&ctx->q4_k_matvec_fn, ctx->quant_module, "minicpmo_q4_k_matvec"), "module_get_function_q4_k", error, error_cap) ||
        !minicpmo_gpu_cuda_check(ctx, ctx->cuda_module_get_function(&ctx->q6_k_matvec_fn, ctx->quant_module, "minicpmo_q6_k_matvec"), "module_get_function_q6_k", error, error_cap) ||
        !minicpmo_gpu_cuda_check(ctx, ctx->cuda_module_get_function(&ctx->q4_k_matvec_multi_fn, ctx->quant_module, "minicpmo_q4_k_matvec_multi"), "module_get_function_q4_k_multi", error, error_cap) ||
        !minicpmo_gpu_cuda_check(ctx, ctx->cuda_module_get_function(&ctx->q6_k_matvec_multi_fn, ctx->quant_module, "minicpmo_q6_k_matvec_multi"), "module_get_function_q6_k_multi", error, error_cap) ||
        !minicpmo_gpu_cuda_check(ctx, ctx->cuda_module_get_function(&ctx->q8_1_quantize_fn, ctx->quant_module, "minicpmo_q8_1_quantize"), "module_get_function_q8_1_quantize", error, error_cap) ||
        !minicpmo_gpu_cuda_check(ctx, ctx->cuda_module_get_function(&ctx->attention_decode_fn, ctx->quant_module, "minicpmo_qwen3_attention_decode"), "module_get_function_attention_decode", error, error_cap) ||
        !minicpmo_gpu_cuda_check(ctx, ctx->cuda_module_get_function(&ctx->f32_to_f16_fn, ctx->quant_module, "minicpmo_f32_to_f16"), "module_get_function_f32_to_f16", error, error_cap) ||
        !minicpmo_gpu_cuda_check(ctx, ctx->cuda_module_get_function(&ctx->add_inplace_fn, ctx->quant_module, "minicpmo_add_inplace"), "module_get_function_add_inplace", error, error_cap) ||
        !minicpmo_gpu_cuda_check(ctx, ctx->cuda_module_get_function(&ctx->silu_mul_fn, ctx->quant_module, "minicpmo_silu_mul"), "module_get_function_silu_mul", error, error_cap) ||
        !minicpmo_gpu_cuda_check(ctx, ctx->cuda_module_get_function(&ctx->rope_fn, ctx->quant_module, "minicpmo_apply_rope"), "module_get_function_apply_rope", error, error_cap) ||
        !minicpmo_gpu_cuda_check(ctx, ctx->cuda_module_get_function(&ctx->rms_norm_fn, ctx->quant_module, "minicpmo_rms_norm"), "module_get_function_rms_norm", error, error_cap) ||
        !minicpmo_gpu_cuda_check(ctx, ctx->cuda_module_get_function(&ctx->head_norm_fn, ctx->quant_module, "minicpmo_head_norm"), "module_get_function_head_norm", error, error_cap)) {
        if (ctx->quant_module != NULL && ctx->cuda_module_unload != NULL) {
            ctx->cuda_module_unload(ctx->quant_module);
        }
        ctx->quant_module = NULL;
        ctx->q4_k_matvec_fn = NULL;
        ctx->q6_k_matvec_fn = NULL;
        ctx->q4_k_matvec_multi_fn = NULL;
        ctx->q6_k_matvec_multi_fn = NULL;
        ctx->q8_1_quantize_fn = NULL;
        ctx->attention_decode_fn = NULL;
        ctx->f32_to_f16_fn = NULL;
        ctx->add_inplace_fn = NULL;
        ctx->silu_mul_fn = NULL;
        ctx->rope_fn = NULL;
        ctx->rms_norm_fn = NULL;
        ctx->head_norm_fn = NULL;
        free(ptx);
        return 0;
    }
    free(ptx);
    ctx->quant_kernels_ready = 1;
    if (ctx->debug) {
        fprintf(stderr, "minicpmo-gpu: quant kernels ready device=%u\n", ctx->device_index);
    }
    return 1;
}

static int minicpmo_gpu_launch_quant_matvec_to(
    minicpmo_gpu_context *ctx,
    minicpmo_gpu_upload *upload,
    CUdeviceptr x_ptr,
    CUdeviceptr y_ptr,
    size_t out_dim,
    size_t in_dim,
    char *error,
    size_t error_cap) {
    CUfunction fn = NULL;
    int rows = (int) out_dim;
    int cols = (int) in_dim;
    void *args[5];
    if (!minicpmo_gpu_prepare_quant_kernels(ctx, error, error_cap)) {
        return 0;
    }
    if (upload->upload_dtype == 10) {
        fn = ctx->q4_k_matvec_fn;
    } else if (upload->upload_dtype == 12) {
        fn = ctx->q6_k_matvec_fn;
    } else {
        minicpmo_gpu_set_error(ctx, error, error_cap, "error: unsupported native quant kernel dtype");
        return 0;
    }
    args[0] = &upload->device_ptr;
    args[1] = &x_ptr;
    args[2] = &y_ptr;
    args[3] = &rows;
    args[4] = &cols;
    if (!minicpmo_gpu_cuda_check(
            ctx,
            ctx->cuda_launch_kernel(
                fn,
                (unsigned int) out_dim,
                1,
                1,
                MINICPMO_GPU_QUANT_THREADS,
                1,
                1,
                0,
                NULL,
                args,
                NULL),
            "launch_quant_matvec",
            error,
            error_cap)) {
        return 0;
    }
    return 1;
}

static int minicpmo_gpu_launch_quant_matvec(
    minicpmo_gpu_context *ctx,
    minicpmo_gpu_upload *upload,
    size_t out_dim,
    size_t in_dim,
    char *error,
    size_t error_cap) {
    return minicpmo_gpu_launch_quant_matvec_to(ctx, upload, ctx->scratch_x, ctx->scratch_y, out_dim, in_dim, error, error_cap);
}

static int minicpmo_gpu_launch_quant_matvec_multi_to(
    minicpmo_gpu_context *ctx,
    minicpmo_gpu_upload **uploads,
    size_t count,
    CUdeviceptr x_ptr,
    CUdeviceptr *dst_ptrs,
    size_t *out_dims,
    size_t in_dim,
    char *error,
    size_t error_cap) {
    CUfunction fn = NULL;
    CUdeviceptr w0 = 0;
    CUdeviceptr w1 = 0;
    CUdeviceptr w2 = 0;
    CUdeviceptr d0 = 0;
    CUdeviceptr d1 = 0;
    CUdeviceptr d2 = 0;
    int rows0 = 0;
    int rows1 = 0;
    int rows2 = 0;
    int cols = (int) in_dim;
    int count_i = (int) count;
    size_t total_rows = 0;
    void *args[12];
    if (count < 2u || count > 3u || uploads == NULL || dst_ptrs == NULL || out_dims == NULL) {
        minicpmo_gpu_set_error(ctx, error, error_cap, "error: invalid grouped quant matvec arguments");
        return 0;
    }
    if (!minicpmo_gpu_prepare_quant_kernels(ctx, error, error_cap)) {
        return 0;
    }
    if (uploads[0]->upload_dtype == 10) {
        fn = ctx->q4_k_matvec_multi_fn;
    } else if (uploads[0]->upload_dtype == 12) {
        fn = ctx->q6_k_matvec_multi_fn;
    } else {
        minicpmo_gpu_set_error(ctx, error, error_cap, "error: unsupported grouped native quant kernel dtype");
        return 0;
    }
    w0 = uploads[0]->device_ptr;
    w1 = uploads[1]->device_ptr;
    d0 = dst_ptrs[0];
    d1 = dst_ptrs[1];
    rows0 = (int) out_dims[0];
    rows1 = (int) out_dims[1];
    total_rows = out_dims[0] + out_dims[1];
    if (count == 3u) {
        w2 = uploads[2]->device_ptr;
        d2 = dst_ptrs[2];
        rows2 = (int) out_dims[2];
        total_rows += out_dims[2];
    }
    if (total_rows == 0 || total_rows > 2147483647ull) {
        minicpmo_gpu_set_error(ctx, error, error_cap, "error: grouped native quant matvec rows exceed int32 limit");
        return 0;
    }
    args[0] = &w0;
    args[1] = &w1;
    args[2] = &w2;
    args[3] = &x_ptr;
    args[4] = &d0;
    args[5] = &d1;
    args[6] = &d2;
    args[7] = &rows0;
    args[8] = &rows1;
    args[9] = &rows2;
    args[10] = &cols;
    args[11] = &count_i;
    if (!minicpmo_gpu_cuda_check(
            ctx,
            ctx->cuda_launch_kernel(
                fn,
                (unsigned int) total_rows,
                1,
                1,
                MINICPMO_GPU_QUANT_THREADS,
                1,
                1,
                0,
                NULL,
                args,
                NULL),
            "launch_quant_matvec_multi",
            error,
            error_cap)) {
        return 0;
    }
    return 1;
}

static void minicpmo_gpu_release_attention_cache(minicpmo_gpu_context *ctx) {
    if (ctx == NULL || ctx->cuda_mem_free == NULL) {
        return;
    }
    if (ctx->device_k_cache != 0) {
        ctx->cuda_mem_free(ctx->device_k_cache);
        ctx->device_k_cache = 0;
    }
    if (ctx->device_v_cache != 0) {
        ctx->cuda_mem_free(ctx->device_v_cache);
        ctx->device_v_cache = 0;
    }
    ctx->device_kv_bytes = 0;
    ctx->kv_layer_count = 0;
    ctx->kv_context_capacity = 0;
    ctx->kv_dim = 0;
    ctx->stats.kv_device_bytes = 0;
}

int minicpmo_gpu_context_prepare_attention(
    void *handle,
    size_t layer_count,
    size_t context_capacity,
    size_t kv_dim,
    char *error,
    size_t error_cap) {
    minicpmo_gpu_context *ctx = (minicpmo_gpu_context *) handle;
    size_t one_cache_bytes;
    size_t total_device_bytes;
    if (ctx == NULL) {
        minicpmo_gpu_copy_error(error, error_cap, "error: null gpu attention context");
        return 0;
    }
    if (layer_count == 0 || context_capacity == 0 || kv_dim == 0) {
        minicpmo_gpu_set_error(ctx, error, error_cap, "error: invalid gpu attention dimensions");
        return 0;
    }
    if (!minicpmo_gpu_prepare_quant_kernels(ctx, error, error_cap) || ctx->attention_decode_fn == NULL) {
        return 0;
    }
    if (ctx->device_k_cache != 0 &&
        ctx->device_v_cache != 0 &&
        ctx->kv_layer_count == layer_count &&
        ctx->kv_context_capacity == context_capacity &&
        ctx->kv_dim == kv_dim) {
        return 1;
    }
    if (layer_count > SIZE_MAX / context_capacity || layer_count * context_capacity > SIZE_MAX / kv_dim ||
        layer_count * context_capacity * kv_dim > SIZE_MAX / sizeof(float)) {
        minicpmo_gpu_set_error(ctx, error, error_cap, "error: gpu attention cache size overflow");
        return 0;
    }
    one_cache_bytes = layer_count * context_capacity * kv_dim * sizeof(float);
    total_device_bytes = one_cache_bytes * 2u;
    minicpmo_gpu_release_attention_cache(ctx);
    if (!minicpmo_gpu_cuda_check(ctx, ctx->cuda_mem_alloc(&ctx->device_k_cache, one_cache_bytes), "attention_k_cache_malloc", error, error_cap)) {
        minicpmo_gpu_release_attention_cache(ctx);
        return 0;
    }
    if (!minicpmo_gpu_cuda_check(ctx, ctx->cuda_mem_alloc(&ctx->device_v_cache, one_cache_bytes), "attention_v_cache_malloc", error, error_cap)) {
        minicpmo_gpu_release_attention_cache(ctx);
        return 0;
    }
    ctx->device_kv_bytes = total_device_bytes;
    ctx->kv_layer_count = layer_count;
    ctx->kv_context_capacity = context_capacity;
    ctx->kv_dim = kv_dim;
    ctx->stats.kv_device_bytes = (uint64_t) total_device_bytes;
    if (ctx->cache_limit_bytes > total_device_bytes) {
        ctx->cache_limit_bytes -= total_device_bytes;
    } else {
        ctx->cache_limit_bytes = 0;
    }
    if (ctx->persistent_limit_bytes > total_device_bytes) {
        ctx->persistent_limit_bytes -= total_device_bytes;
    } else {
        ctx->persistent_limit_bytes = 0;
    }
    if (ctx->debug) {
        fprintf(stderr, "minicpmo-gpu: attention cache ready layers=%zu context=%zu kv_dim=%zu device_bytes=%zu cache_limit=%zu persistent_limit=%zu\n",
            layer_count,
            context_capacity,
            kv_dim,
            total_device_bytes,
            ctx->cache_limit_bytes,
            ctx->persistent_limit_bytes);
    }
    return 1;
}

int minicpmo_gpu_context_attention(
    void *handle,
    const float *q,
    const float *k,
    const float *v,
    size_t layer,
    size_t pos,
    size_t head_count,
    size_t kv_head_count,
    size_t head_dim,
    size_t context_capacity,
    float *dst,
    char *error,
    size_t error_cap) {
    minicpmo_gpu_context *ctx = (minicpmo_gpu_context *) handle;
    size_t q_dim;
    size_t q_bytes;
    size_t kv_bytes;
    int layer_i;
    int pos_i;
    int head_count_i;
    int kv_head_count_i;
    int head_dim_i;
    int context_capacity_i;
    float scale;
    void *args[11];
    if (ctx == NULL) {
        minicpmo_gpu_copy_error(error, error_cap, "error: null gpu attention context");
        return 0;
    }
    if (q == NULL || k == NULL || v == NULL || dst == NULL || head_count == 0 || kv_head_count == 0 || head_dim == 0) {
        minicpmo_gpu_set_error(ctx, error, error_cap, "error: invalid gpu attention arguments");
        return 0;
    }
    if (head_dim > MINICPMO_GPU_ATTENTION_THREADS) {
        minicpmo_gpu_set_error(ctx, error, error_cap, "error: gpu attention currently supports head_dim <= 256");
        return 0;
    }
    if (ctx->device_k_cache == 0 || ctx->device_v_cache == 0 || ctx->kv_dim == 0) {
        minicpmo_gpu_set_error(ctx, error, error_cap, "error: gpu attention cache not prepared");
        return 0;
    }
    if (layer >= ctx->kv_layer_count || pos >= ctx->kv_context_capacity || context_capacity != ctx->kv_context_capacity) {
        minicpmo_gpu_set_error(ctx, error, error_cap, "error: gpu attention cache shape mismatch");
        return 0;
    }
    if (kv_head_count > SIZE_MAX / head_dim || head_count > SIZE_MAX / head_dim) {
        minicpmo_gpu_set_error(ctx, error, error_cap, "error: gpu attention shape overflow");
        return 0;
    }
    q_dim = head_count * head_dim;
    if (kv_head_count * head_dim != ctx->kv_dim) {
        minicpmo_gpu_set_error(ctx, error, error_cap, "error: gpu attention kv_dim mismatch");
        return 0;
    }
    if (q_dim > 2147483647u || ctx->kv_dim > 2147483647u || layer > 2147483647u || pos > 2147483647u ||
        head_count > 2147483647u || kv_head_count > 2147483647u || head_dim > 2147483647u || context_capacity > 2147483647u) {
        minicpmo_gpu_set_error(ctx, error, error_cap, "error: gpu attention shape exceeds int32 limit");
        return 0;
    }
    q_bytes = q_dim * sizeof(float);
    kv_bytes = ctx->kv_dim * sizeof(float);
    if (!minicpmo_gpu_ensure_scratch(ctx, &ctx->scratch_x, &ctx->scratch_x_bytes, q_bytes, "attention_q_malloc", error, error_cap) ||
        !minicpmo_gpu_ensure_scratch(ctx, &ctx->scratch_y, &ctx->scratch_y_bytes, q_bytes, "attention_out_malloc", error, error_cap)) {
        return 0;
    }
    ctx->stats.scratch_x_bytes = (uint64_t) ctx->scratch_x_bytes;
    ctx->stats.scratch_y_bytes = (uint64_t) ctx->scratch_y_bytes;
    if (!minicpmo_gpu_cuda_check(ctx, ctx->cuda_memcpy_hto_d(ctx->scratch_x, q, q_bytes), "attention_q_copy_h2d", error, error_cap)) {
        return 0;
    }
    {
        size_t kv_offset = ((layer * ctx->kv_context_capacity) + pos) * ctx->kv_dim * sizeof(float);
        if (!minicpmo_gpu_cuda_check(ctx, ctx->cuda_memcpy_hto_d(ctx->device_k_cache + kv_offset, k, kv_bytes), "attention_k_copy_h2d", error, error_cap) ||
            !minicpmo_gpu_cuda_check(ctx, ctx->cuda_memcpy_hto_d(ctx->device_v_cache + kv_offset, v, kv_bytes), "attention_v_copy_h2d", error, error_cap)) {
            return 0;
        }
    }
    ctx->stats.host_to_device_bytes += (uint64_t) (q_bytes + kv_bytes + kv_bytes);
    ctx->stats.kv_host_to_device_bytes += (uint64_t) (kv_bytes + kv_bytes);
    layer_i = (int) layer;
    pos_i = (int) pos;
    head_count_i = (int) head_count;
    kv_head_count_i = (int) kv_head_count;
    head_dim_i = (int) head_dim;
    context_capacity_i = (int) context_capacity;
    scale = 1.0f / minicpmo_gpu_sqrtf_positive((float) head_dim);
    args[0] = &ctx->scratch_x;
    args[1] = &ctx->device_k_cache;
    args[2] = &ctx->device_v_cache;
    args[3] = &ctx->scratch_y;
    args[4] = &layer_i;
    args[5] = &pos_i;
    args[6] = &head_count_i;
    args[7] = &kv_head_count_i;
    args[8] = &head_dim_i;
    args[9] = &context_capacity_i;
    args[10] = &scale;
    if (!minicpmo_gpu_cuda_check(
            ctx,
            ctx->cuda_launch_kernel(
                ctx->attention_decode_fn,
                (unsigned int) head_count,
                1,
                1,
                MINICPMO_GPU_ATTENTION_THREADS,
                1,
                1,
                0,
                NULL,
                args,
                NULL),
            "launch_attention_decode",
            error,
            error_cap) ||
        !minicpmo_gpu_cuda_check(ctx, ctx->cuda_ctx_synchronize(), "attention_decode_sync", error, error_cap)) {
        return 0;
    }
    if (!minicpmo_gpu_cuda_check(ctx, ctx->cuda_memcpy_dto_h(dst, ctx->scratch_y, q_bytes), "attention_out_copy_d2h", error, error_cap)) {
        return 0;
    }
    ctx->stats.device_to_host_bytes += (uint64_t) q_bytes;
    ctx->stats.attention_kernel_calls += 1u;
    if (ctx->debug) {
        fprintf(stderr, "minicpmo-gpu: attention layer=%zu pos=%zu heads=%zu kv_heads=%zu head_dim=%zu q_bytes=%zu kv_bytes=%zu\n",
            layer,
            pos,
            head_count,
            kv_head_count,
            head_dim,
            q_bytes,
            kv_bytes);
    }
    return 1;
}

static minicpmo_gpu_upload *minicpmo_gpu_find_upload(minicpmo_gpu_context *ctx, const void *host_ptr) {
    size_t i;
    for (i = 0; i < ctx->upload_count; ++i) {
        if (ctx->uploads[i].host_ptr == host_ptr) {
            return &ctx->uploads[i];
        }
    }
    return NULL;
}

static void minicpmo_gpu_release_upload(minicpmo_gpu_context *ctx, minicpmo_gpu_upload *upload) {
    if (ctx == NULL || upload == NULL || upload->device_ptr == 0 || ctx->cuda_mem_free == NULL) {
        return;
    }
    ctx->cuda_mem_free(upload->device_ptr);
    upload->device_ptr = 0;
    if (upload->resident) {
        if (ctx->current_device_bytes >= upload->staged_bytes) {
            ctx->current_device_bytes -= upload->staged_bytes;
        } else {
            ctx->current_device_bytes = 0;
        }
    }
    upload->resident = 0;
    upload->last_use_tick = 0;
}

static void minicpmo_gpu_mark_upload_used(minicpmo_gpu_context *ctx, minicpmo_gpu_upload *upload) {
    if (ctx == NULL || upload == NULL || !upload->resident || upload->device_ptr == 0) {
        return;
    }
    ctx->use_tick += 1u;
    upload->last_use_tick = ctx->use_tick;
}

static void minicpmo_gpu_invalidate_q8_1_cache(minicpmo_gpu_context *ctx) {
    if (ctx == NULL) {
        return;
    }
    ctx->q8_1_cache_valid = 0;
    ctx->q8_1_cache_slot = 0;
    ctx->q8_1_cache_cols = 0;
}

static void minicpmo_gpu_mark_slot_written(minicpmo_gpu_context *ctx, unsigned int slot) {
    if (ctx == NULL) {
        return;
    }
    if (ctx->q8_1_cache_valid && ctx->q8_1_cache_slot == slot) {
        minicpmo_gpu_invalidate_q8_1_cache(ctx);
    }
}

static int minicpmo_gpu_evict_until_fit(
    minicpmo_gpu_context *ctx,
    size_t required,
    minicpmo_gpu_upload *skip_upload,
    char *error,
    size_t error_cap) {
    if (ctx == NULL) {
        minicpmo_gpu_copy_error(error, error_cap, "error: null gpu cache context");
        return 0;
    }
    if (ctx->cache_limit_bytes == 0) {
        return 1;
    }
    while (ctx->current_device_bytes + required > ctx->cache_limit_bytes) {
        size_t victim_index = MINICPMO_GPU_MAX_UPLOADS;
        uint64_t victim_tick = 0;
        size_t i;
        for (i = 0; i < ctx->upload_count; ++i) {
            minicpmo_gpu_upload *candidate = &ctx->uploads[i];
            if (candidate == skip_upload || candidate->device_ptr == 0 || candidate->persistent || !candidate->resident) {
                continue;
            }
            if (victim_index == MINICPMO_GPU_MAX_UPLOADS || candidate->last_use_tick < victim_tick) {
                victim_index = i;
                victim_tick = candidate->last_use_tick;
            }
        }
        if (victim_index == MINICPMO_GPU_MAX_UPLOADS) {
            return 1;
        }
        if (ctx->debug) {
            fprintf(stderr, "minicpmo-gpu: evict tensor=%s bytes=%zu last_use=%llu current_device_bytes=%zu cache_limit=%zu\n",
                ctx->uploads[victim_index].name,
                ctx->uploads[victim_index].staged_bytes,
                (unsigned long long) ctx->uploads[victim_index].last_use_tick,
                ctx->current_device_bytes,
                ctx->cache_limit_bytes);
        }
        minicpmo_gpu_release_upload(ctx, &ctx->uploads[victim_index]);
    }
    return 1;
}

static void minicpmo_gpu_evict_all(minicpmo_gpu_context *ctx) {
    size_t i;
    if (ctx == NULL || ctx->cuda_mem_free == NULL) {
        return;
    }
    for (i = 0; i < ctx->upload_count; ++i) {
        minicpmo_gpu_release_upload(ctx, &ctx->uploads[i]);
    }
    ctx->current_device_bytes = 0;
}

static void minicpmo_gpu_finalize_upload_plan(minicpmo_gpu_context *ctx, minicpmo_gpu_upload *upload) {
    if (ctx == NULL || upload == NULL) {
        return;
    }
    upload->estimated_device_bytes = upload->staged_bytes;
    if (upload->persistent_decided) {
        return;
    }
    upload->persistent = 0;
    if (upload->staged_bytes > 0 &&
        ctx->persistent_reserved_bytes + upload->staged_bytes <= ctx->persistent_limit_bytes) {
        upload->persistent = 1;
        ctx->persistent_reserved_bytes += upload->staged_bytes;
    }
    upload->persistent_decided = 1;
}

static int minicpmo_gpu_prepare_staging(minicpmo_gpu_context *ctx, minicpmo_gpu_upload *upload, char *error, size_t error_cap) {
    if (upload->staged_host_ptr != NULL) {
        minicpmo_gpu_finalize_upload_plan(ctx, upload);
        return 1;
    }
    if (upload->dtype == 0) {
        upload->staged_host_ptr = upload->host_ptr;
        upload->staged_bytes = upload->bytes;
        upload->upload_dtype = 0;
        upload->staged_owned = 0;
        minicpmo_gpu_finalize_upload_plan(ctx, upload);
        return 1;
    }
    if (upload->dtype == 1) {
        upload->staged_host_ptr = upload->host_ptr;
        upload->staged_bytes = upload->rows * upload->cols * sizeof(uint16_t);
        upload->upload_dtype = 1;
        upload->staged_owned = 0;
        minicpmo_gpu_finalize_upload_plan(ctx, upload);
        return 1;
    }
    if (upload->dtype == 10 || upload->dtype == 12) {
        if ((upload->cols % 256u) != 0u) {
            minicpmo_gpu_set_error(ctx, error, error_cap, "error: gpu quant staging requires cols multiple of 256");
            return 0;
        }
        if (minicpmo_gpu_native_kmatvec_requested(ctx, upload->dtype)) {
            char native_error[512];
            if (minicpmo_gpu_prepare_quant_kernels(ctx, native_error, sizeof(native_error))) {
                upload->staged_host_ptr = upload->host_ptr;
                upload->staged_bytes = upload->bytes;
                upload->upload_dtype = upload->dtype;
                upload->staged_owned = 0;
                minicpmo_gpu_finalize_upload_plan(ctx, upload);
                return 1;
            }
            if (ctx->debug) {
                fprintf(stderr, "minicpmo-gpu: native quant unavailable tensor=%s dtype=%d falling_back=host_f16_staging reason=%s\n",
                    upload->name[0] != '\0' ? upload->name : "(unnamed)",
                    upload->dtype,
                    native_error[0] != '\0' ? native_error : "unknown");
            }
        }
        {
            uint16_t *dst = (uint16_t *) malloc(upload->rows * upload->cols * sizeof(uint16_t));
            if (dst == NULL) {
                minicpmo_gpu_set_error(ctx, error, error_cap, "error: gpu host staging allocation failed");
                return 0;
            }
            if (upload->dtype == 10) {
                const uint8_t *data = (const uint8_t *) upload->host_ptr;
                const size_t qk = 256;
                const size_t block_bytes = 144;
                const size_t blocks_per_row = upload->cols / qk;
                size_t row = 0;
                for (row = 0; row < upload->rows; ++row) {
                    size_t row_base = row * blocks_per_row * block_bytes;
                    size_t block = 0;
                    while (block < blocks_per_row) {
                        size_t base = row_base + block * block_bytes;
                        float d = minicpmo_gpu_f16_to_f32((uint16_t) (data[base] | (data[base + 1] << 8)));
                        float dmin = minicpmo_gpu_f16_to_f32((uint16_t) (data[base + 2] | (data[base + 3] << 8)));
                        const uint8_t *scales = &data[base + 4];
                        const uint8_t *qs = &data[base + 16];
                        size_t out_base = row * upload->cols + block * qk;
                        int is = 0;
                        size_t group = 0;
                        while (group < 4) {
                            uint8_t sc = 0;
                            uint8_t mn = 0;
                            size_t l;
                            float d1;
                            float m1;
                            float d2;
                            float m2;
                            size_t qbase = group * 32;
                            if (is < 4) {
                                sc = scales[is] & 63u;
                                mn = scales[is + 4] & 63u;
                            } else {
                                sc = (uint8_t) ((scales[is + 4] & 15u) | ((scales[is - 4] >> 6) << 4));
                                mn = (uint8_t) ((scales[is + 4] >> 4) | ((scales[is] >> 6) << 4));
                            }
                            d1 = d * (float) sc;
                            m1 = dmin * (float) mn;
                            if (is + 1 < 4) {
                                sc = scales[is + 1] & 63u;
                                mn = scales[is + 5] & 63u;
                            } else {
                                sc = (uint8_t) ((scales[is + 5] & 15u) | ((scales[is - 3] >> 6) << 4));
                                mn = (uint8_t) ((scales[is + 5] >> 4) | ((scales[is + 1] >> 6) << 4));
                            }
                            d2 = d * (float) sc;
                            m2 = dmin * (float) mn;
                            for (l = 0; l < 32; ++l) {
                                dst[out_base + group * 64 + l] = minicpmo_gpu_f32_to_f16(d1 * (float) (qs[qbase + l] & 15u) - m1);
                            }
                            for (l = 0; l < 32; ++l) {
                                dst[out_base + group * 64 + 32 + l] = minicpmo_gpu_f32_to_f16(d2 * (float) (qs[qbase + l] >> 4) - m2);
                            }
                            is += 2;
                            ++group;
                        }
                        ++block;
                    }
                }
            } else {
                const uint8_t *data = (const uint8_t *) upload->host_ptr;
                const size_t qk = 256;
                const size_t block_bytes = 210;
                const size_t blocks_per_row = upload->cols / qk;
                size_t row = 0;
                for (row = 0; row < upload->rows; ++row) {
                    size_t row_base = row * blocks_per_row * block_bytes;
                    size_t block = 0;
                    while (block < blocks_per_row) {
                        size_t base = row_base + block * block_bytes;
                        const uint8_t *ql = &data[base];
                        const uint8_t *qh = &data[base + 128];
                        const int8_t *sc = (const int8_t *) &data[base + 192];
                        float d = minicpmo_gpu_f16_to_f32((uint16_t) (data[base + 208] | (data[base + 209] << 8)));
                        size_t out_base = row * upload->cols + block * qk;
                        size_t n = 0;
                        while (n < 256) {
                            size_t l = 0;
                            while (l < 32) {
                                size_t isv = l / 16;
                                int q1 = ((int) (ql[n / 2 + l] & 15u) | (((int) ((qh[n / 4 + l] >> 0) & 3u)) << 4)) - 32;
                                int q2 = ((int) (ql[n / 2 + l + 32] & 15u) | (((int) ((qh[n / 4 + l] >> 2) & 3u)) << 4)) - 32;
                                int q3 = ((int) (ql[n / 2 + l] >> 4) | (((int) ((qh[n / 4 + l] >> 4) & 3u)) << 4)) - 32;
                                int q4 = ((int) (ql[n / 2 + l + 32] >> 4) | (((int) ((qh[n / 4 + l] >> 6) & 3u)) << 4)) - 32;
                                dst[out_base + n + l] = minicpmo_gpu_f32_to_f16(d * (float) sc[n / 16 + isv + 0] * (float) q1);
                                dst[out_base + n + 32 + l] = minicpmo_gpu_f32_to_f16(d * (float) sc[n / 16 + isv + 2] * (float) q2);
                                dst[out_base + n + 64 + l] = minicpmo_gpu_f32_to_f16(d * (float) sc[n / 16 + isv + 4] * (float) q3);
                                dst[out_base + n + 96 + l] = minicpmo_gpu_f32_to_f16(d * (float) sc[n / 16 + isv + 6] * (float) q4);
                                ++l;
                            }
                            n += 128;
                        }
                        ++block;
                    }
                }
            }
            upload->staged_host_ptr = dst;
            upload->staged_bytes = upload->rows * upload->cols * sizeof(uint16_t);
            upload->upload_dtype = 1;
            upload->staged_owned = 1;
            minicpmo_gpu_finalize_upload_plan(ctx, upload);
            return 1;
        }
    }
    if (upload->dtype == 20) {
        uint16_t *dst = (uint16_t *) malloc(upload->rows * upload->cols * sizeof(uint16_t));
        if (dst == NULL) {
            minicpmo_gpu_set_error(ctx, error, error_cap, "error: gpu host staging allocation failed");
            return 0;
        }
        {
            const uint16_t *src = (const uint16_t *) upload->host_ptr;
            size_t count = upload->rows * upload->cols;
            size_t i;
            for (i = 0; i < count; ++i) {
                dst[i] = minicpmo_gpu_f32_to_f16(minicpmo_gpu_bf16_to_f32(src[i]));
            }
        }
        upload->staged_host_ptr = dst;
        upload->staged_bytes = upload->rows * upload->cols * sizeof(uint16_t);
        upload->upload_dtype = 1;
        upload->staged_owned = 1;
        minicpmo_gpu_finalize_upload_plan(ctx, upload);
        return 1;
    }
    minicpmo_gpu_set_error(ctx, error, error_cap, "error: unsupported gpu upload dtype");
    return 0;
}

static size_t minicpmo_gpu_estimate_device_bytes(int dtype, size_t rows, size_t cols) {
    if (dtype == 0) {
        return rows * cols * sizeof(float);
    }
    if (dtype == 1 || dtype == 20) {
        return rows * cols * sizeof(uint16_t);
    }
    if (dtype == 10) {
        return rows * cols * sizeof(uint16_t);
    }
    if (dtype == 12) {
        return rows * cols * sizeof(uint16_t);
    }
    return 0;
}

static int minicpmo_gpu_ensure_upload_device(
    minicpmo_gpu_context *ctx,
    minicpmo_gpu_upload *upload,
    const char *tensor_name,
    int *release_after_use,
    char *error,
    size_t error_cap) {
    const char *name = tensor_name;
    if (ctx == NULL || upload == NULL) {
        minicpmo_gpu_set_error(ctx, error, error_cap, "error: null gpu upload context");
        return 0;
    }
    if (release_after_use != NULL) {
        *release_after_use = 0;
    }
    if (!minicpmo_gpu_prepare_staging(ctx, upload, error, error_cap)) {
        return 0;
    }
    if (upload->device_ptr != 0) {
        minicpmo_gpu_mark_upload_used(ctx, upload);
        return 1;
    }
    if (name == NULL || name[0] == '\0') {
        name = upload->name[0] != '\0' ? upload->name : "(unnamed)";
    }
    if (upload->persistent) {
        if (!minicpmo_gpu_evict_until_fit(ctx, upload->staged_bytes, upload, error, error_cap)) {
            return 0;
        }
        if (ctx->cache_limit_bytes != 0 && ctx->current_device_bytes + upload->staged_bytes > ctx->cache_limit_bytes) {
            char message[256];
            snprintf(message, sizeof(message), "error: persistent gpu cache overcommitted tensor=%s staged_bytes=%zu current_device_bytes=%zu cache_limit_bytes=%zu", name, upload->staged_bytes, ctx->current_device_bytes, ctx->cache_limit_bytes);
            minicpmo_gpu_set_error(ctx, error, error_cap, message);
            return 0;
        }
    } else if (ctx->cache_limit_bytes != 0) {
        if (!minicpmo_gpu_evict_until_fit(ctx, upload->staged_bytes, upload, error, error_cap)) {
            return 0;
        }
        if (ctx->current_device_bytes + upload->staged_bytes > ctx->cache_limit_bytes) {
            if (release_after_use != NULL) {
                *release_after_use = 1;
            }
        }
    }
    if (!minicpmo_gpu_cuda_check(ctx, ctx->cuda_mem_alloc(&upload->device_ptr, upload->staged_bytes), "weight_malloc", error, error_cap)) {
        return 0;
    }
    if (!minicpmo_gpu_cuda_check(ctx, ctx->cuda_memcpy_hto_d(upload->device_ptr, upload->staged_host_ptr, upload->staged_bytes), "weight_copy_h2d", error, error_cap)) {
        ctx->cuda_mem_free(upload->device_ptr);
        upload->device_ptr = 0;
        return 0;
    }
    upload->resident = release_after_use == NULL || *release_after_use == 0;
    if (upload->resident) {
        ctx->current_device_bytes += upload->staged_bytes;
        minicpmo_gpu_mark_upload_used(ctx, upload);
    }
    ctx->stats.host_to_device_bytes += (uint64_t) upload->staged_bytes;
    ctx->stats.weight_upload_bytes += (uint64_t) upload->staged_bytes;
    if (!upload->ever_uploaded) {
        upload->ever_uploaded = 1;
        ctx->stats.uploaded_weight_count += 1;
    }
    if (ctx->debug) {
        fprintf(stderr, "minicpmo-gpu: device upload tensor=%s staged_bytes=%zu upload_dtype=%d persistent=%d resident=%d current_device_bytes=%zu cache_limit=%zu\n",
            name,
            upload->staged_bytes,
            upload->upload_dtype,
            upload->persistent,
            upload->resident,
            ctx->current_device_bytes,
            ctx->cache_limit_bytes);
    }
    return 1;
}

static int minicpmo_gpu_ensure_scratch(
    minicpmo_gpu_context *ctx,
    CUdeviceptr *ptr,
    size_t *capacity,
    size_t required,
    const char *label,
    char *error,
    size_t error_cap) {
    if (required <= *capacity) {
        return 1;
    }
    if (ptr == &ctx->scratch_x) {
        minicpmo_gpu_invalidate_q8_1_cache(ctx);
    }
    if (*ptr != 0) {
        if (!minicpmo_gpu_cuda_check(ctx, ctx->cuda_mem_free(*ptr), "free scratch", error, error_cap)) {
            return 0;
        }
        *ptr = 0;
        *capacity = 0;
    }
    if (!minicpmo_gpu_cuda_check(ctx, ctx->cuda_mem_alloc(ptr, required), label, error, error_cap)) {
        return 0;
    }
    *capacity = required;
    return 1;
}

static int minicpmo_gpu_ensure_slot(
    minicpmo_gpu_context *ctx,
    unsigned int slot,
    size_t required,
    const char *label,
    char *error,
    size_t error_cap) {
    minicpmo_gpu_slot *entry;
    if (ctx == NULL || slot >= MINICPMO_GPU_SLOT_COUNT) {
        minicpmo_gpu_set_error(ctx, error, error_cap, "error: invalid gpu slot");
        return 0;
    }
    entry = &ctx->slots[slot];
    if (required <= entry->bytes && entry->device_ptr != 0) {
        return 1;
    }
    if (entry->device_ptr != 0) {
        if (!minicpmo_gpu_cuda_check(ctx, ctx->cuda_mem_free(entry->device_ptr), "free slot", error, error_cap)) {
            return 0;
        }
        entry->device_ptr = 0;
        entry->bytes = 0;
    }
    if (!minicpmo_gpu_cuda_check(ctx, ctx->cuda_mem_alloc(&entry->device_ptr, required), label, error, error_cap)) {
        return 0;
    }
    entry->bytes = required;
    return 1;
}

static int minicpmo_gpu_ensure_rope_table(
    minicpmo_gpu_context *ctx,
    const float *rope_cos,
    const float *rope_sin,
    size_t rope_context,
    size_t half_dim,
    char *error,
    size_t error_cap) {
    size_t count;
    size_t bytes;
    if (ctx == NULL) {
        minicpmo_gpu_copy_error(error, error_cap, "error: null gpu rope cache context");
        return 0;
    }
    if (rope_cos == NULL || rope_sin == NULL) {
        ctx->rope_cache_valid = 0;
        return 1;
    }
    if (half_dim == 0 || rope_context == 0 || half_dim > ((size_t) -1) / rope_context) {
        minicpmo_gpu_set_error(ctx, error, error_cap, "error: invalid gpu rope cache size");
        return 0;
    }
    count = rope_context * half_dim;
    bytes = count * sizeof(float);
    if (bytes / sizeof(float) != count) {
        minicpmo_gpu_set_error(ctx, error, error_cap, "error: invalid gpu rope cache size");
        return 0;
    }
    if (ctx->rope_cache_valid &&
        ctx->rope_cos_host == rope_cos &&
        ctx->rope_sin_host == rope_sin &&
        ctx->rope_cache_context == rope_context &&
        ctx->rope_cache_half_dim == half_dim &&
        ctx->device_rope_bytes >= bytes) {
        return 1;
    }
    if (ctx->device_rope_cos == 0 || ctx->device_rope_sin == 0 || ctx->device_rope_bytes < bytes) {
        if (ctx->device_rope_cos != 0) {
            ctx->cuda_mem_free(ctx->device_rope_cos);
            ctx->device_rope_cos = 0;
        }
        if (ctx->device_rope_sin != 0) {
            ctx->cuda_mem_free(ctx->device_rope_sin);
            ctx->device_rope_sin = 0;
        }
        ctx->device_rope_bytes = 0;
        ctx->rope_cache_valid = 0;
        if (!minicpmo_gpu_cuda_check(ctx, ctx->cuda_mem_alloc(&ctx->device_rope_cos, bytes), "rope_cos_malloc", error, error_cap)) {
            return 0;
        }
        if (!minicpmo_gpu_cuda_check(ctx, ctx->cuda_mem_alloc(&ctx->device_rope_sin, bytes), "rope_sin_malloc", error, error_cap)) {
            ctx->cuda_mem_free(ctx->device_rope_cos);
            ctx->device_rope_cos = 0;
            return 0;
        }
        ctx->device_rope_bytes = bytes;
    }
    if (!minicpmo_gpu_cuda_check(ctx, ctx->cuda_memcpy_hto_d(ctx->device_rope_cos, rope_cos, bytes), "rope_cos_copy_h2d", error, error_cap) ||
        !minicpmo_gpu_cuda_check(ctx, ctx->cuda_memcpy_hto_d(ctx->device_rope_sin, rope_sin, bytes), "rope_sin_copy_h2d", error, error_cap)) {
        ctx->rope_cache_valid = 0;
        return 0;
    }
    ctx->stats.host_to_device_bytes += (uint64_t) (bytes * 2u);
    ctx->rope_cos_host = rope_cos;
    ctx->rope_sin_host = rope_sin;
    ctx->rope_cache_context = rope_context;
    ctx->rope_cache_half_dim = half_dim;
    ctx->rope_cache_valid = 1;
    return 1;
}

static int minicpmo_gpu_cuda_type_for_dtype(int dtype) {
    if (dtype == 0) {
        return CUDA_R_32F;
    }
    if (dtype == 1) {
        return CUDA_R_16F;
    }
    return -1;
}

static float minicpmo_gpu_f16_to_f32(uint16_t bits) {
    uint32_t sign = ((uint32_t) bits >> 15) & 1u;
    uint32_t exp = ((uint32_t) bits >> 10) & 31u;
    uint32_t frac = (uint32_t) bits & 1023u;
    uint32_t out_bits;
    union { uint32_t u; float f; } out;
    if (exp == 0) {
        if (frac == 0) {
            out_bits = sign << 31;
        } else {
            int sub_exp = -14;
            while ((frac & 1024u) == 0) {
                frac <<= 1u;
                sub_exp -= 1;
            }
            frac &= 1023u;
            out_bits = (sign << 31) | ((uint32_t) (sub_exp + 127) << 23) | (frac << 13);
        }
    } else if (exp == 31u) {
        out_bits = (sign << 31) | 0x7f800000u | (frac << 13);
    } else {
        out_bits = (sign << 31) | ((exp + (127u - 15u)) << 23) | (frac << 13);
    }
    out.u = out_bits;
    return out.f;
}

static float minicpmo_gpu_bf16_to_f32(uint16_t bits) {
    union { uint32_t u; float f; } out;
    out.u = ((uint32_t) bits) << 16;
    return out.f;
}

static uint16_t minicpmo_gpu_f32_to_f16(float value) {
    union { float f; uint32_t u; } in;
    uint32_t sign;
    uint32_t exp;
    uint32_t frac;
    uint16_t out;
    in.f = value;
    sign = (in.u >> 16) & 0x8000u;
    exp = (in.u >> 23) & 0xffu;
    frac = in.u & 0x7fffffu;
    if (exp == 255u) {
        if (frac != 0u) {
            return (uint16_t) (sign | 0x7e00u);
        }
        return (uint16_t) (sign | 0x7c00u);
    }
    if (exp > 142u) {
        return (uint16_t) (sign | 0x7c00u);
    }
    if (exp < 113u) {
        if (exp < 103u) {
            return (uint16_t) sign;
        }
        frac |= 0x800000u;
        out = (uint16_t) (sign | (frac >> (126u - exp)));
        if ((frac >> (125u - exp)) & 1u) {
            out = (uint16_t) (out + 1u);
        }
        return out;
    }
    out = (uint16_t) (sign | ((exp - 112u) << 10) | (frac >> 13));
    if (frac & 0x1000u) {
        out = (uint16_t) (out + 1u);
    }
    return out;
}

static float minicpmo_gpu_sqrtf_positive(float value) {
    float x;
    int i;
    if (value <= 0.0f) {
        return 0.0f;
    }
    x = value > 1.0f ? value : 1.0f;
    for (i = 0; i < 8; ++i) {
        x = 0.5f * (x + value / x);
    }
    return x;
}

int minicpmo_gpu_context_create(unsigned int device_index, int debug, void **out_handle, char *error, size_t error_cap) {
    static const char *const cuda_names[] = {
        "libcuda.so.1",
        "libcuda.so",
        NULL,
    };
    static const char *const cublas_names[] = {
        "libcublas.so.12",
        "libcublas.so",
        "libcublas.so.11",
        NULL,
    };
    uint64_t start_us;
    uint64_t end_us;
    int device_count = 0;
    minicpmo_gpu_context *ctx;
    if (out_handle == NULL) {
        minicpmo_gpu_copy_error(error, error_cap, "error: null gpu context out pointer");
        return 0;
    }
    *out_handle = NULL;
    ctx = (minicpmo_gpu_context *) calloc(1, sizeof(*ctx));
    if (ctx == NULL) {
        minicpmo_gpu_copy_error(error, error_cap, "error: gpu context allocation failed");
        return 0;
    }
    ctx->device_index = device_index;
    ctx->debug = debug;
    start_us = minicpmo_gpu_now_us();
    ctx->cuda_lib = minicpmo_gpu_try_dlopen(cuda_names);
    if (ctx->cuda_lib == NULL) {
        minicpmo_gpu_set_error(ctx, error, error_cap, "error: unable to load libcuda.so; install NVIDIA driver runtime");
        minicpmo_gpu_context_destroy(ctx);
        return 0;
    }
    ctx->cublas_lib = minicpmo_gpu_try_dlopen(cublas_names);
    if (ctx->cublas_lib == NULL) {
        minicpmo_gpu_set_error(ctx, error, error_cap, "error: unable to load libcublas.so; install NVIDIA cuBLAS runtime");
        minicpmo_gpu_context_destroy(ctx);
        return 0;
    }
    if (!minicpmo_gpu_resolve_symbol(ctx, ctx->cuda_lib, "cuInit", (void **) &ctx->cuda_init, error, error_cap) ||
        !minicpmo_gpu_resolve_symbol(ctx, ctx->cuda_lib, "cuDeviceGetCount", (void **) &ctx->cuda_device_get_count, error, error_cap) ||
        !minicpmo_gpu_resolve_symbol(ctx, ctx->cuda_lib, "cuDeviceGet", (void **) &ctx->cuda_device_get, error, error_cap) ||
        !minicpmo_gpu_resolve_symbol(ctx, ctx->cuda_lib, "cuDeviceGetAttribute", (void **) &ctx->cuda_device_get_attribute, error, error_cap) ||
        !minicpmo_gpu_resolve_symbol(ctx, ctx->cuda_lib, "cuCtxCreate_v2", (void **) &ctx->cuda_ctx_create, error, error_cap) ||
        !minicpmo_gpu_resolve_symbol(ctx, ctx->cuda_lib, "cuCtxDestroy_v2", (void **) &ctx->cuda_ctx_destroy, error, error_cap) ||
        !minicpmo_gpu_resolve_symbol(ctx, ctx->cuda_lib, "cuMemAlloc_v2", (void **) &ctx->cuda_mem_alloc, error, error_cap) ||
        !minicpmo_gpu_resolve_symbol(ctx, ctx->cuda_lib, "cuMemFree_v2", (void **) &ctx->cuda_mem_free, error, error_cap) ||
        !minicpmo_gpu_resolve_symbol(ctx, ctx->cuda_lib, "cuMemcpyHtoD_v2", (void **) &ctx->cuda_memcpy_hto_d, error, error_cap) ||
        !minicpmo_gpu_resolve_symbol(ctx, ctx->cuda_lib, "cuMemcpyDtoD_v2", (void **) &ctx->cuda_memcpy_dto_d, error, error_cap) ||
        !minicpmo_gpu_resolve_symbol(ctx, ctx->cuda_lib, "cuMemcpyDtoH_v2", (void **) &ctx->cuda_memcpy_dto_h, error, error_cap) ||
        !minicpmo_gpu_resolve_symbol(ctx, ctx->cuda_lib, "cuMemGetInfo_v2", (void **) &ctx->cuda_mem_get_info, error, error_cap) ||
        !minicpmo_gpu_resolve_symbol(ctx, ctx->cuda_lib, "cuGetErrorString", (void **) &ctx->cuda_get_error_string, error, error_cap) ||
        !minicpmo_gpu_resolve_symbol(ctx, ctx->cuda_lib, "cuCtxSynchronize", (void **) &ctx->cuda_ctx_synchronize, error, error_cap) ||
        !minicpmo_gpu_resolve_symbol(ctx, ctx->cublas_lib, "cublasCreate_v2", (void **) &ctx->cublas_create, error, error_cap) ||
        !minicpmo_gpu_resolve_symbol(ctx, ctx->cublas_lib, "cublasDestroy_v2", (void **) &ctx->cublas_destroy, error, error_cap) ||
        !minicpmo_gpu_resolve_symbol(ctx, ctx->cublas_lib, "cublasSgemv_v2", (void **) &ctx->cublas_sgemv, error, error_cap) ||
        !minicpmo_gpu_resolve_symbol(ctx, ctx->cublas_lib, "cublasGemmEx", (void **) &ctx->cublas_gemm_ex, error, error_cap)) {
        minicpmo_gpu_context_destroy(ctx);
        return 0;
    }
    if (!minicpmo_gpu_cuda_check(ctx, ctx->cuda_init(0), "init", error, error_cap)) {
        minicpmo_gpu_context_destroy(ctx);
        return 0;
    }
    if (!minicpmo_gpu_cuda_check(ctx, ctx->cuda_device_get_count(&device_count), "get_device_count", error, error_cap)) {
        minicpmo_gpu_context_destroy(ctx);
        return 0;
    }
    if (device_count <= 0) {
        minicpmo_gpu_set_error(ctx, error, error_cap, "error: no CUDA devices detected");
        minicpmo_gpu_context_destroy(ctx);
        return 0;
    }
    if ((int) device_index >= device_count) {
        char message[256];
        snprintf(message, sizeof(message), "error: unsupported cuda device index=%u available=%d first_version_device_index=0", device_index, device_count);
        minicpmo_gpu_set_error(ctx, error, error_cap, message);
        minicpmo_gpu_context_destroy(ctx);
        return 0;
    }
    if (!minicpmo_gpu_cuda_check(ctx, ctx->cuda_device_get(&ctx->cuda_device, (int) device_index), "device_get", error, error_cap)) {
        minicpmo_gpu_context_destroy(ctx);
        return 0;
    }
    if (!minicpmo_gpu_cuda_check(ctx, ctx->cuda_device_get_attribute(&ctx->compute_capability_major, MINICPMO_CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, ctx->cuda_device), "device_get_attribute_major", error, error_cap) ||
        !minicpmo_gpu_cuda_check(ctx, ctx->cuda_device_get_attribute(&ctx->compute_capability_minor, MINICPMO_CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, ctx->cuda_device), "device_get_attribute_minor", error, error_cap)) {
        minicpmo_gpu_context_destroy(ctx);
        return 0;
    }
    if (!minicpmo_gpu_cuda_check(ctx, ctx->cuda_ctx_create(&ctx->cuda_context, 0, ctx->cuda_device), "ctx_create", error, error_cap)) {
        minicpmo_gpu_context_destroy(ctx);
        return 0;
    }
    if (!minicpmo_gpu_cublas_check(ctx, ctx->cublas_create(&ctx->cublas_handle), "create", error, error_cap)) {
        minicpmo_gpu_context_destroy(ctx);
        return 0;
    }
    if (ctx->cuda_mem_get_info != NULL) {
        size_t free_bytes = 0;
        size_t total_bytes = 0;
        if (minicpmo_gpu_cuda_check(ctx, ctx->cuda_mem_get_info(&free_bytes, &total_bytes), "mem_get_info", error, error_cap)) {
            size_t env_limit = minicpmo_gpu_parse_env_size_mb("MINICPMO_GPU_CACHE_MB", 0);
            if (env_limit > 0 && env_limit < free_bytes) {
                ctx->cache_limit_bytes = env_limit;
            } else if (free_bytes > (256u * 1024u * 1024u)) {
                ctx->cache_limit_bytes = free_bytes - (256u * 1024u * 1024u);
            } else {
                ctx->cache_limit_bytes = free_bytes;
            }
            if (ctx->cache_limit_bytes == 0) {
                ctx->cache_limit_bytes = total_bytes / 2u;
            }
            {
                size_t env_persistent = minicpmo_gpu_parse_env_size_mb("MINICPMO_GPU_PERSISTENT_MB", 0);
                if (env_persistent > 0 && env_persistent < ctx->cache_limit_bytes) {
                    ctx->persistent_limit_bytes = env_persistent;
                } else if (ctx->cache_limit_bytes > (64u * 1024u * 1024u)) {
                    ctx->persistent_limit_bytes = ctx->cache_limit_bytes - (64u * 1024u * 1024u);
                } else {
                    ctx->persistent_limit_bytes = ctx->cache_limit_bytes;
                }
            }
        }
    }
    end_us = minicpmo_gpu_now_us();
    ctx->stats.backend_init_us = end_us - start_us;
    if (ctx->debug) {
        fprintf(stderr, "minicpmo-gpu: create device=%u cc=%d.%d init_us=%llu\n",
            device_index,
            ctx->compute_capability_major,
            ctx->compute_capability_minor,
            (unsigned long long) ctx->stats.backend_init_us);
    }
    *out_handle = ctx;
    minicpmo_gpu_copy_error(error, error_cap, "");
    return 1;
}

void minicpmo_gpu_context_destroy(void *handle) {
    minicpmo_gpu_context *ctx = (minicpmo_gpu_context *) handle;
    size_t i;
    if (ctx == NULL) {
        return;
    }
    if (ctx->cuda_mem_free != NULL) {
        for (i = 0; i < ctx->upload_count; ++i) {
            minicpmo_gpu_release_upload(ctx, &ctx->uploads[i]);
            if (ctx->uploads[i].staged_owned && ctx->uploads[i].staged_host_ptr != NULL) {
                free((void *) ctx->uploads[i].staged_host_ptr);
                ctx->uploads[i].staged_host_ptr = NULL;
            }
        }
        minicpmo_gpu_release_attention_cache(ctx);
        for (i = 0; i < MINICPMO_GPU_SLOT_COUNT; ++i) {
            if (ctx->slots[i].device_ptr != 0) {
                ctx->cuda_mem_free(ctx->slots[i].device_ptr);
                ctx->slots[i].device_ptr = 0;
                ctx->slots[i].bytes = 0;
            }
        }
        if (ctx->scratch_x != 0) {
            ctx->cuda_mem_free(ctx->scratch_x);
            ctx->scratch_x = 0;
        }
        if (ctx->scratch_y != 0) {
            ctx->cuda_mem_free(ctx->scratch_y);
            ctx->scratch_y = 0;
        }
        if (ctx->device_rope_cos != 0) {
            ctx->cuda_mem_free(ctx->device_rope_cos);
            ctx->device_rope_cos = 0;
        }
        if (ctx->device_rope_sin != 0) {
            ctx->cuda_mem_free(ctx->device_rope_sin);
            ctx->device_rope_sin = 0;
        }
    }
    if (ctx->cublas_destroy != NULL && ctx->cublas_handle != NULL) {
        ctx->cublas_destroy(ctx->cublas_handle);
        ctx->cublas_handle = NULL;
    }
    if (ctx->cuda_module_unload != NULL && ctx->quant_module != NULL) {
        ctx->cuda_module_unload(ctx->quant_module);
        ctx->quant_module = NULL;
        ctx->q4_k_matvec_fn = NULL;
        ctx->q6_k_matvec_fn = NULL;
        ctx->q4_k_matvec_multi_fn = NULL;
        ctx->q6_k_matvec_multi_fn = NULL;
        ctx->q8_1_quantize_fn = NULL;
        ctx->attention_decode_fn = NULL;
        ctx->f32_to_f16_fn = NULL;
        ctx->add_inplace_fn = NULL;
        ctx->silu_mul_fn = NULL;
        ctx->rope_fn = NULL;
        ctx->rms_norm_fn = NULL;
        ctx->head_norm_fn = NULL;
    }
    if (ctx->cuda_ctx_destroy != NULL && ctx->cuda_context != NULL) {
        ctx->cuda_ctx_destroy(ctx->cuda_context);
        ctx->cuda_context = NULL;
    }
    if (ctx->nvrtc_lib != NULL) {
        dlclose(ctx->nvrtc_lib);
        ctx->nvrtc_lib = NULL;
    }
    if (ctx->cublas_lib != NULL) {
        dlclose(ctx->cublas_lib);
        ctx->cublas_lib = NULL;
    }
    if (ctx->cuda_lib != NULL) {
        dlclose(ctx->cuda_lib);
        ctx->cuda_lib = NULL;
    }
    free(ctx);
}

int minicpmo_gpu_context_smoke(void *handle, char *error, size_t error_cap) {
    minicpmo_gpu_context *ctx = (minicpmo_gpu_context *) handle;
    float host_in[4] = {1.0f, -2.0f, 3.5f, 4.0f};
    float host_out[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    CUdeviceptr device_ptr = 0;
    size_t bytes = sizeof(host_in);
    if (ctx == NULL) {
        minicpmo_gpu_copy_error(error, error_cap, "error: null gpu smoke context");
        return 0;
    }
    if (!minicpmo_gpu_cuda_check(ctx, ctx->cuda_mem_alloc(&device_ptr, bytes), "smoke_malloc", error, error_cap)) {
        return 0;
    }
    if (!minicpmo_gpu_cuda_check(ctx, ctx->cuda_memcpy_hto_d(device_ptr, host_in, bytes), "smoke_copy_h2d", error, error_cap)) {
        ctx->cuda_mem_free(device_ptr);
        return 0;
    }
    ctx->stats.host_to_device_bytes += (uint64_t) bytes;
    if (!minicpmo_gpu_cuda_check(ctx, ctx->cuda_memcpy_dto_h(host_out, device_ptr, bytes), "smoke_copy_d2h", error, error_cap)) {
        ctx->cuda_mem_free(device_ptr);
        return 0;
    }
    ctx->stats.device_to_host_bytes += (uint64_t) bytes;
    ctx->cuda_mem_free(device_ptr);
    if (memcmp(host_in, host_out, bytes) != 0) {
        minicpmo_gpu_set_error(ctx, error, error_cap, "error: cuda smoke roundtrip mismatch");
        return 0;
    }
    return 1;
}

int minicpmo_gpu_context_upload_tensor(
    void *handle,
    const char *name,
    const void *host_ptr,
    size_t bytes,
    int dtype,
    size_t rows,
    size_t cols,
    char *error,
    size_t error_cap) {
    minicpmo_gpu_context *ctx = (minicpmo_gpu_context *) handle;
    minicpmo_gpu_upload *existing;
    minicpmo_gpu_upload *entry;
    int release_after_use = 0;
    if (ctx == NULL) {
        minicpmo_gpu_copy_error(error, error_cap, "error: null gpu upload context");
        return 0;
    }
    if (host_ptr == NULL || bytes == 0 || rows == 0 || cols == 0) {
        minicpmo_gpu_set_error(ctx, error, error_cap, "error: invalid gpu upload tensor");
        return 0;
    }
    existing = minicpmo_gpu_find_upload(ctx, host_ptr);
    if (existing != NULL) {
        existing->rows = rows;
        existing->cols = cols;
        if (!minicpmo_gpu_ensure_upload_device(ctx, existing, name, &release_after_use, error, error_cap)) {
            return 0;
        }
        if (release_after_use && existing->device_ptr != 0) {
            minicpmo_gpu_release_upload(ctx, existing);
        }
        return 1;
    }
    if (ctx->upload_count >= MINICPMO_GPU_MAX_UPLOADS) {
        minicpmo_gpu_set_error(ctx, error, error_cap, "error: gpu upload registry exhausted");
        return 0;
    }
    entry = &ctx->uploads[ctx->upload_count];
    entry->host_ptr = host_ptr;
    entry->staged_host_ptr = NULL;
    entry->device_ptr = 0;
    entry->bytes = bytes;
    entry->rows = rows;
    entry->cols = cols;
    entry->staged_bytes = 0;
    entry->estimated_device_bytes = minicpmo_gpu_estimate_device_bytes(dtype, rows, cols);
    entry->dtype = dtype;
    entry->upload_dtype = -1;
    entry->staged_owned = 0;
    entry->ever_uploaded = 0;
    entry->persistent_decided = 0;
    entry->persistent = 0;
    entry->resident = 0;
    entry->last_use_tick = 0;
    if (name != NULL) {
        snprintf(entry->name, sizeof(entry->name), "%s", name);
    } else {
        entry->name[0] = '\0';
    }
    ++ctx->upload_count;
    if (!minicpmo_gpu_ensure_upload_device(ctx, entry, name, &release_after_use, error, error_cap)) {
        return 0;
    }
    if (release_after_use && entry->device_ptr != 0) {
        minicpmo_gpu_release_upload(ctx, entry);
    }
    if (ctx->debug) {
        fprintf(stderr, "minicpmo-gpu: track tensor=%s bytes=%zu dtype=%d rows=%zu cols=%zu persistent=%d resident=%d persistent_reserved=%zu count=%zu\n", name != NULL ? name : "(null)", bytes, dtype, rows, cols, entry->persistent, entry->resident, ctx->persistent_reserved_bytes, ctx->upload_count);
    }
    return 1;
}

int minicpmo_gpu_context_matvec(
    void *handle,
    const char *tensor_name,
    const void *host_ptr,
    int dtype,
    const float *x,
    size_t in_dim,
    size_t out_dim,
    float *dst,
    char *error,
    size_t error_cap) {
    minicpmo_gpu_context *ctx = (minicpmo_gpu_context *) handle;
    minicpmo_gpu_upload *upload;
    int a_type = -1;
    const float alpha = 1.0f;
    const float beta = 0.0f;
    size_t x_bytes = in_dim * sizeof(float);
    size_t x_half_bytes = in_dim * sizeof(uint16_t);
    size_t x_q8_bytes = 0;
    size_t y_bytes = out_dim * sizeof(float);
    size_t scratch_x_required = 0;
    int release_after_use = 0;
    int used_native_quant = 0;
    int used_staged_quant = 0;
    int ok = 1;
    if (ctx == NULL) {
        minicpmo_gpu_copy_error(error, error_cap, "error: null gpu matvec context");
        return 0;
    }
    if (x == NULL || dst == NULL || host_ptr == NULL || in_dim == 0 || out_dim == 0) {
        minicpmo_gpu_set_error(ctx, error, error_cap, "error: invalid gpu matvec arguments");
        return 0;
    }
    upload = minicpmo_gpu_find_upload(ctx, host_ptr);
    if (upload == NULL) {
        char message[256];
        snprintf(message, sizeof(message), "error: gpu tensor not uploaded tensor=%s", tensor_name != NULL ? tensor_name : "(null)");
        minicpmo_gpu_set_error(ctx, error, error_cap, message);
        return 0;
    }
    if (upload->cols != in_dim || upload->rows != out_dim) {
        char message[256];
        snprintf(message, sizeof(message), "error: gpu tensor shape mismatch tensor=%s tracked_rows=%zu tracked_cols=%zu matvec_out=%zu matvec_in=%zu", tensor_name != NULL ? tensor_name : "(null)", upload->rows, upload->cols, out_dim, in_dim);
        minicpmo_gpu_set_error(ctx, error, error_cap, message);
        return 0;
    }
    if (!minicpmo_gpu_ensure_upload_device(ctx, upload, tensor_name, &release_after_use, error, error_cap)) {
        return 0;
    }
    if (in_dim > 2147483647u || out_dim > 2147483647u) {
        minicpmo_gpu_set_error(ctx, error, error_cap, "error: gpu matvec shape exceeds int32 kernel/cublas limit");
        return 0;
    }
    if (upload->upload_dtype == 0 || upload->upload_dtype == 10 || upload->upload_dtype == 12) {
        if (upload->upload_dtype == 0) {
            scratch_x_required = x_bytes;
        } else {
            x_q8_bytes = minicpmo_gpu_q8_1_super_bytes_for_cols(in_dim);
            if (x_q8_bytes == 0) {
                minicpmo_gpu_set_error(ctx, error, error_cap, "error: native quant matvec expects in_dim multiple of 256");
                return 0;
            }
            scratch_x_required = x_q8_bytes;
        }
    } else {
        a_type = minicpmo_gpu_cuda_type_for_dtype(upload->upload_dtype);
        if (a_type < 0) {
            char message[256];
            snprintf(message, sizeof(message), "error: gpu unsupported uploaded matvec dtype tensor=%s dtype=%d", tensor_name != NULL ? tensor_name : "(null)", upload->upload_dtype);
            minicpmo_gpu_set_error(ctx, error, error_cap, message);
            return 0;
        }
        scratch_x_required = x_bytes > x_half_bytes ? x_bytes : x_half_bytes;
    }
    if (!minicpmo_gpu_ensure_scratch(ctx, &ctx->scratch_x, &ctx->scratch_x_bytes, scratch_x_required, "scratch_x_malloc", error, error_cap) ||
        !minicpmo_gpu_ensure_scratch(ctx, &ctx->scratch_y, &ctx->scratch_y_bytes, y_bytes, "scratch_y_malloc", error, error_cap)) {
        return 0;
    }
    ctx->stats.scratch_x_bytes = (uint64_t) ctx->scratch_x_bytes;
    ctx->stats.scratch_y_bytes = (uint64_t) ctx->scratch_y_bytes;
    if (upload->upload_dtype == 0) {
        minicpmo_gpu_invalidate_q8_1_cache(ctx);
        if (!minicpmo_gpu_cuda_check(ctx, ctx->cuda_memcpy_hto_d(ctx->scratch_x, x, x_bytes), "matvec_copy_x_h2d", error, error_cap)) {
            ok = 0;
            goto matvec_cleanup;
        }
        ctx->stats.host_to_device_bytes += (uint64_t) x_bytes;
        if (!minicpmo_gpu_cublas_check(
                ctx,
                ctx->cublas_sgemv(
                    ctx->cublas_handle,
                    CUBLAS_OP_T,
                    (int) in_dim,
                    (int) out_dim,
                    &alpha,
                    (const float *)(uintptr_t) upload->device_ptr,
                    (int) in_dim,
                    (const float *)(uintptr_t) ctx->scratch_x,
                    1,
                    &beta,
                    (float *)(uintptr_t) ctx->scratch_y,
                    1),
                "sgemv",
                error,
                error_cap)) {
            ok = 0;
            goto matvec_cleanup;
        }
    } else if (upload->upload_dtype == 10 || upload->upload_dtype == 12) {
        unsigned char x_q8_host[x_q8_bytes];
        minicpmo_gpu_invalidate_q8_1_cache(ctx);
        if (!minicpmo_gpu_q8_1_quantize_row(x_q8_host, x, in_dim)) {
            minicpmo_gpu_set_error(ctx, error, error_cap, "error: native quant matvec Q8_1 quantize failed");
            ok = 0;
            goto matvec_cleanup;
        }
        if (!minicpmo_gpu_cuda_check(ctx, ctx->cuda_memcpy_hto_d(ctx->scratch_x, x_q8_host, x_q8_bytes), "matvec_copy_x_q8_1_h2d", error, error_cap)) {
            ok = 0;
            goto matvec_cleanup;
        }
        ctx->stats.host_to_device_bytes += (uint64_t) x_q8_bytes;
        ctx->stats.q8_1_x_bytes += (uint64_t) x_q8_bytes;
        if (!minicpmo_gpu_launch_quant_matvec(ctx, upload, out_dim, in_dim, error, error_cap)) {
            ok = 0;
            goto matvec_cleanup;
        }
        used_native_quant = 1;
    } else {
        size_t i;
        uint16_t x_half_host[in_dim];
        minicpmo_gpu_invalidate_q8_1_cache(ctx);
        for (i = 0; i < in_dim; ++i) {
            x_half_host[i] = minicpmo_gpu_f32_to_f16(x[i]);
        }
        if (!minicpmo_gpu_cuda_check(ctx, ctx->cuda_memcpy_hto_d(ctx->scratch_x, x_half_host, x_half_bytes), "matvec_copy_x_half_h2d", error, error_cap)) {
            ok = 0;
            goto matvec_cleanup;
        }
        ctx->stats.host_to_device_bytes += (uint64_t) x_half_bytes;
        if (!minicpmo_gpu_cublas_check(
                ctx,
                ctx->cublas_gemm_ex(
                    ctx->cublas_handle,
                    CUBLAS_OP_T,
                    CUBLAS_OP_N,
                    (int) out_dim,
                    1,
                    (int) in_dim,
                    &alpha,
                    (const void *)(uintptr_t) upload->device_ptr,
                    CUDA_R_16F,
                    (int) in_dim,
                    (const void *)(uintptr_t) ctx->scratch_x,
                    CUDA_R_16F,
                    (int) in_dim,
                    &beta,
                    (void *)(uintptr_t) ctx->scratch_y,
                    CUDA_R_32F,
                    (int) out_dim,
                    CUBLAS_COMPUTE_32F,
                    CUBLAS_GEMM_DEFAULT),
                "gemm_ex_f16",
                error,
                error_cap)) {
            ok = 0;
            goto matvec_cleanup;
        }
        if (upload->dtype == 10 || upload->dtype == 12) {
            used_staged_quant = 1;
        }
    }
    if (!minicpmo_gpu_cuda_check(ctx, ctx->cuda_memcpy_dto_h(dst, ctx->scratch_y, y_bytes), "matvec_copy_y_d2h", error, error_cap)) {
        ok = 0;
        goto matvec_cleanup;
    }
    ctx->stats.device_to_host_bytes += (uint64_t) y_bytes;
    if (used_native_quant) {
        ctx->stats.native_quant_matvec_count += 1u;
    }
    if (used_staged_quant) {
        ctx->stats.staged_quant_matvec_count += 1u;
    }
    if (ctx->debug) {
        fprintf(stderr, "minicpmo-gpu: matvec tensor=%s in=%zu out=%zu dtype=%d persistent=%d resident=%d current_device_bytes=%zu\n", tensor_name != NULL ? tensor_name : "(null)", in_dim, out_dim, dtype, upload->persistent, upload->resident, ctx->current_device_bytes);
    }
matvec_cleanup:
    if (release_after_use && upload->device_ptr != 0) {
        minicpmo_gpu_release_upload(ctx, upload);
    }
    return ok;
}

int minicpmo_gpu_context_slot_upload_f32(
    void *handle,
    unsigned int slot,
    const float *src,
    size_t len,
    char *error,
    size_t error_cap) {
    minicpmo_gpu_context *ctx = (minicpmo_gpu_context *) handle;
    size_t bytes = len * sizeof(float);
    if (ctx == NULL) {
        minicpmo_gpu_copy_error(error, error_cap, "error: null gpu slot upload context");
        return 0;
    }
    if (src == NULL || len == 0 || slot >= MINICPMO_GPU_SLOT_COUNT) {
        minicpmo_gpu_set_error(ctx, error, error_cap, "error: invalid gpu slot upload arguments");
        return 0;
    }
    if (!minicpmo_gpu_ensure_slot(ctx, slot, bytes, "slot_upload_malloc", error, error_cap)) {
        return 0;
    }
    if (!minicpmo_gpu_cuda_check(ctx, ctx->cuda_memcpy_hto_d(ctx->slots[slot].device_ptr, src, bytes), "slot_upload_h2d", error, error_cap)) {
        return 0;
    }
    minicpmo_gpu_mark_slot_written(ctx, slot);
    ctx->stats.host_to_device_bytes += (uint64_t) bytes;
    return 1;
}

int minicpmo_gpu_context_slot_download_f32(
    void *handle,
    unsigned int slot,
    float *dst,
    size_t len,
    char *error,
    size_t error_cap) {
    minicpmo_gpu_context *ctx = (minicpmo_gpu_context *) handle;
    size_t bytes = len * sizeof(float);
    if (ctx == NULL) {
        minicpmo_gpu_copy_error(error, error_cap, "error: null gpu slot download context");
        return 0;
    }
    if (dst == NULL || len == 0 || slot >= MINICPMO_GPU_SLOT_COUNT || ctx->slots[slot].device_ptr == 0 || ctx->slots[slot].bytes < bytes) {
        minicpmo_gpu_set_error(ctx, error, error_cap, "error: invalid gpu slot download arguments");
        return 0;
    }
    if (!minicpmo_gpu_cuda_check(ctx, ctx->cuda_ctx_synchronize(), "slot_download_sync", error, error_cap)) {
        return 0;
    }
    if (!minicpmo_gpu_cuda_check(ctx, ctx->cuda_memcpy_dto_h(dst, ctx->slots[slot].device_ptr, bytes), "slot_download_d2h", error, error_cap)) {
        return 0;
    }
    ctx->stats.device_to_host_bytes += (uint64_t) bytes;
    return 1;
}

int minicpmo_gpu_context_slot_matvec(
    void *handle,
    unsigned int dst_slot,
    const char *tensor_name,
    const void *host_ptr,
    int dtype,
    unsigned int src_slot,
    size_t in_dim,
    size_t out_dim,
    char *error,
    size_t error_cap) {
    minicpmo_gpu_context *ctx = (minicpmo_gpu_context *) handle;
    minicpmo_gpu_upload *upload;
    size_t x_bytes = in_dim * sizeof(float);
    size_t x_half_bytes = in_dim * sizeof(uint16_t);
    size_t x_q8_bytes = 0;
    size_t y_bytes = out_dim * sizeof(float);
    int release_after_use = 0;
    int a_type = -1;
    int ok = 1;
    int used_native_quant = 0;
    int used_staged_quant = 0;
    const float alpha = 1.0f;
    const float beta = 0.0f;
    CUdeviceptr src_ptr;
    CUdeviceptr dst_ptr;
    if (ctx == NULL) {
        minicpmo_gpu_copy_error(error, error_cap, "error: null gpu slot matvec context");
        return 0;
    }
    if (host_ptr == NULL || in_dim == 0 || out_dim == 0 || src_slot >= MINICPMO_GPU_SLOT_COUNT || dst_slot >= MINICPMO_GPU_SLOT_COUNT) {
        minicpmo_gpu_set_error(ctx, error, error_cap, "error: invalid gpu slot matvec arguments");
        return 0;
    }
    upload = minicpmo_gpu_find_upload(ctx, host_ptr);
    if (upload == NULL) {
        char message[256];
        snprintf(message, sizeof(message), "error: gpu tensor not uploaded tensor=%s", tensor_name != NULL ? tensor_name : "(null)");
        minicpmo_gpu_set_error(ctx, error, error_cap, message);
        return 0;
    }
    if (upload->cols != in_dim || upload->rows != out_dim) {
        char message[256];
        snprintf(message, sizeof(message), "error: gpu tensor shape mismatch tensor=%s tracked_rows=%zu tracked_cols=%zu matvec_out=%zu matvec_in=%zu", tensor_name != NULL ? tensor_name : "(null)", upload->rows, upload->cols, out_dim, in_dim);
        minicpmo_gpu_set_error(ctx, error, error_cap, message);
        return 0;
    }
    if (ctx->slots[src_slot].device_ptr == 0 || ctx->slots[src_slot].bytes < x_bytes) {
        minicpmo_gpu_set_error(ctx, error, error_cap, "error: gpu slot matvec source not ready");
        return 0;
    }
    if (!minicpmo_gpu_ensure_upload_device(ctx, upload, tensor_name, &release_after_use, error, error_cap)) {
        return 0;
    }
    if (!minicpmo_gpu_ensure_slot(ctx, dst_slot, y_bytes, "slot_matvec_dst_malloc", error, error_cap)) {
        ok = 0;
        goto slot_matvec_cleanup;
    }
    src_ptr = ctx->slots[src_slot].device_ptr;
    dst_ptr = ctx->slots[dst_slot].device_ptr;
    if (in_dim > 2147483647u || out_dim > 2147483647u) {
        minicpmo_gpu_set_error(ctx, error, error_cap, "error: gpu slot matvec shape exceeds int32 limit");
        ok = 0;
        goto slot_matvec_cleanup;
    }
    if (upload->upload_dtype == 0) {
        if (!minicpmo_gpu_cublas_check(
                ctx,
                ctx->cublas_sgemv(
                    ctx->cublas_handle,
                    CUBLAS_OP_T,
                    (int) in_dim,
                    (int) out_dim,
                    &alpha,
                    (const float *)(uintptr_t) upload->device_ptr,
                    (int) in_dim,
                    (const float *)(uintptr_t) src_ptr,
                    1,
                    &beta,
                    (float *)(uintptr_t) dst_ptr,
                    1),
                "slot_sgemv",
                error,
                error_cap)) {
            ok = 0;
            goto slot_matvec_cleanup;
        }
    } else if (upload->upload_dtype == 10 || upload->upload_dtype == 12) {
        int cols_i = (int) in_dim;
        void *qargs[3];
        int cache_hit = 0;
        if (!minicpmo_gpu_prepare_quant_kernels(ctx, error, error_cap) || ctx->q8_1_quantize_fn == NULL) {
            ok = 0;
            goto slot_matvec_cleanup;
        }
        x_q8_bytes = minicpmo_gpu_q8_1_super_bytes_for_cols(in_dim);
        if (x_q8_bytes == 0) {
            minicpmo_gpu_set_error(ctx, error, error_cap, "error: gpu slot native quant matvec expects in_dim multiple of 256");
            ok = 0;
            goto slot_matvec_cleanup;
        }
        if (!minicpmo_gpu_ensure_scratch(ctx, &ctx->scratch_x, &ctx->scratch_x_bytes, x_q8_bytes, "slot_q8_1_malloc", error, error_cap)) {
            ok = 0;
            goto slot_matvec_cleanup;
        }
        ctx->stats.scratch_x_bytes = (uint64_t) ctx->scratch_x_bytes;
        cache_hit = ctx->q8_1_cache_valid &&
            ctx->q8_1_cache_slot == src_slot &&
            ctx->q8_1_cache_cols == in_dim &&
            ctx->scratch_x != 0 &&
            ctx->scratch_x_bytes >= x_q8_bytes;
        if (!cache_hit) {
            qargs[0] = &src_ptr;
            qargs[1] = &ctx->scratch_x;
            qargs[2] = &cols_i;
            if (!minicpmo_gpu_cuda_check(
                    ctx,
                    ctx->cuda_launch_kernel(
                        ctx->q8_1_quantize_fn,
                        (unsigned int) (in_dim / MINICPMO_GPU_QK_K),
                        1,
                        1,
                        MINICPMO_GPU_ATTENTION_THREADS,
                        1,
                        1,
                        0,
                        NULL,
                        qargs,
                        NULL),
                    "launch_q8_1_quantize",
                    error,
                    error_cap)) {
                ok = 0;
                goto slot_matvec_cleanup;
            }
            ctx->q8_1_cache_valid = 1;
            ctx->q8_1_cache_slot = src_slot;
            ctx->q8_1_cache_cols = in_dim;
            ctx->stats.q8_1_x_bytes += (uint64_t) x_q8_bytes;
        }
        if (!minicpmo_gpu_launch_quant_matvec_to(ctx, upload, ctx->scratch_x, dst_ptr, out_dim, in_dim, error, error_cap)) {
            ok = 0;
            goto slot_matvec_cleanup;
        }
        used_native_quant = 1;
    } else {
        int n_i = (int) in_dim;
        void *hargs[3];
        a_type = minicpmo_gpu_cuda_type_for_dtype(upload->upload_dtype);
        if (a_type < 0) {
            char message[256];
            snprintf(message, sizeof(message), "error: gpu unsupported uploaded slot matvec dtype tensor=%s dtype=%d", tensor_name != NULL ? tensor_name : "(null)", upload->upload_dtype);
            minicpmo_gpu_set_error(ctx, error, error_cap, message);
            ok = 0;
            goto slot_matvec_cleanup;
        }
        if (!minicpmo_gpu_prepare_quant_kernels(ctx, error, error_cap) || ctx->f32_to_f16_fn == NULL) {
            ok = 0;
            goto slot_matvec_cleanup;
        }
        if (!minicpmo_gpu_ensure_scratch(ctx, &ctx->scratch_x, &ctx->scratch_x_bytes, x_half_bytes, "slot_half_x_malloc", error, error_cap)) {
            ok = 0;
            goto slot_matvec_cleanup;
        }
        minicpmo_gpu_invalidate_q8_1_cache(ctx);
        ctx->stats.scratch_x_bytes = (uint64_t) ctx->scratch_x_bytes;
        hargs[0] = &src_ptr;
        hargs[1] = &ctx->scratch_x;
        hargs[2] = &n_i;
        if (!minicpmo_gpu_cuda_check(
                ctx,
                ctx->cuda_launch_kernel(
                    ctx->f32_to_f16_fn,
                    (unsigned int) ((in_dim + (MINICPMO_GPU_VECTOR_THREADS - 1u)) / MINICPMO_GPU_VECTOR_THREADS),
                    1,
                    1,
                    MINICPMO_GPU_VECTOR_THREADS,
                    1,
                    1,
                    0,
                    NULL,
                    hargs,
                    NULL),
                "launch_f32_to_f16",
                error,
                error_cap) ||
            !minicpmo_gpu_cublas_check(
                ctx,
                ctx->cublas_gemm_ex(
                    ctx->cublas_handle,
                    CUBLAS_OP_T,
                    CUBLAS_OP_N,
                    (int) out_dim,
                    1,
                    (int) in_dim,
                    &alpha,
                    (const void *)(uintptr_t) upload->device_ptr,
                    CUDA_R_16F,
                    (int) in_dim,
                    (const void *)(uintptr_t) ctx->scratch_x,
                    CUDA_R_16F,
                    (int) in_dim,
                    &beta,
                    (void *)(uintptr_t) dst_ptr,
                    CUDA_R_32F,
                    (int) out_dim,
                    CUBLAS_COMPUTE_32F,
                    CUBLAS_GEMM_DEFAULT),
                "slot_gemm_ex_f16",
                error,
                error_cap)) {
            ok = 0;
            goto slot_matvec_cleanup;
        }
        if (dtype == 10 || dtype == 12) {
            used_staged_quant = 1;
        }
    }
    if (used_native_quant) {
        ctx->stats.native_quant_matvec_count += 1u;
    }
    if (used_staged_quant) {
        ctx->stats.staged_quant_matvec_count += 1u;
    }
slot_matvec_cleanup:
    if (ok) {
        minicpmo_gpu_mark_slot_written(ctx, dst_slot);
    }
    if (release_after_use && upload->device_ptr != 0) {
        minicpmo_gpu_release_upload(ctx, upload);
    }
    return ok;
}

static int minicpmo_gpu_context_slot_matvec_multi_fallback(
    void *handle,
    size_t count,
    unsigned int *dst_slots,
    const char **tensor_names,
    const void **host_ptrs,
    int *dtypes,
    size_t *out_dims,
    unsigned int src_slot,
    size_t in_dim,
    char *error,
    size_t error_cap) {
    size_t i;
    for (i = 0; i < count; ++i) {
        if (!minicpmo_gpu_context_slot_matvec(
                handle,
                dst_slots[i],
                tensor_names[i],
                host_ptrs[i],
                dtypes[i],
                src_slot,
                in_dim,
                out_dims[i],
                error,
                error_cap)) {
            return 0;
        }
    }
    return 1;
}

static int minicpmo_gpu_context_slot_matvec_multi_native(
    minicpmo_gpu_context *ctx,
    size_t count,
    unsigned int *dst_slots,
    const char **tensor_names,
    const void **host_ptrs,
    int *dtypes,
    size_t *out_dims,
    unsigned int src_slot,
    size_t in_dim,
    char *error,
    size_t error_cap) {
    minicpmo_gpu_upload *uploads[3] = {NULL, NULL, NULL};
    CUdeviceptr dst_ptrs[3] = {0, 0, 0};
    int release_after_use[3] = {0, 0, 0};
    size_t x_bytes = in_dim * sizeof(float);
    size_t x_q8_bytes = 0;
    int cols_i = (int) in_dim;
    void *qargs[3];
    int cache_hit = 0;
    int result = 1;
    size_t i;
    if (ctx == NULL || count < 2u || count > 3u || dst_slots == NULL || tensor_names == NULL || host_ptrs == NULL || dtypes == NULL || out_dims == NULL) {
        minicpmo_gpu_set_error(ctx, error, error_cap, "error: invalid grouped gpu slot matvec arguments");
        return 0;
    }
    if (dtypes[0] != 10 && dtypes[0] != 12) {
        return -1;
    }
    for (i = 1; i < count; ++i) {
        if (dtypes[i] != dtypes[0]) {
            return -1;
        }
    }
    if (in_dim == 0 || src_slot >= MINICPMO_GPU_SLOT_COUNT || ctx->slots[src_slot].device_ptr == 0 || ctx->slots[src_slot].bytes < x_bytes) {
        minicpmo_gpu_set_error(ctx, error, error_cap, "error: gpu grouped slot matvec source not ready");
        return 0;
    }
    if (in_dim > 2147483647u) {
        minicpmo_gpu_set_error(ctx, error, error_cap, "error: gpu grouped slot matvec shape exceeds int32 limit");
        return 0;
    }
    if (!minicpmo_gpu_prepare_quant_kernels(ctx, error, error_cap) ||
        ctx->q8_1_quantize_fn == NULL ||
        ctx->q4_k_matvec_multi_fn == NULL ||
        ctx->q6_k_matvec_multi_fn == NULL) {
        return -1;
    }
    for (i = 0; i < count; ++i) {
        if (host_ptrs[i] == NULL || out_dims[i] == 0 || out_dims[i] > 2147483647u || dst_slots[i] >= MINICPMO_GPU_SLOT_COUNT) {
            minicpmo_gpu_set_error(ctx, error, error_cap, "error: invalid grouped gpu slot matvec arguments");
            result = 0;
            goto grouped_cleanup;
        }
        uploads[i] = minicpmo_gpu_find_upload(ctx, host_ptrs[i]);
        if (uploads[i] == NULL) {
            char message[256];
            snprintf(message, sizeof(message), "error: gpu tensor not uploaded tensor=%s", tensor_names[i] != NULL ? tensor_names[i] : "(null)");
            minicpmo_gpu_set_error(ctx, error, error_cap, message);
            result = 0;
            goto grouped_cleanup;
        }
        if (uploads[i]->cols != in_dim || uploads[i]->rows != out_dims[i]) {
            char message[256];
            snprintf(message, sizeof(message), "error: gpu tensor shape mismatch tensor=%s tracked_rows=%zu tracked_cols=%zu matvec_out=%zu matvec_in=%zu", tensor_names[i] != NULL ? tensor_names[i] : "(null)", uploads[i]->rows, uploads[i]->cols, out_dims[i], in_dim);
            minicpmo_gpu_set_error(ctx, error, error_cap, message);
            result = 0;
            goto grouped_cleanup;
        }
        if (!minicpmo_gpu_ensure_upload_device(ctx, uploads[i], tensor_names[i], &release_after_use[i], error, error_cap)) {
            result = 0;
            goto grouped_cleanup;
        }
        if (uploads[i]->upload_dtype != dtypes[0]) {
            result = -1;
            goto grouped_cleanup;
        }
        if (!minicpmo_gpu_ensure_slot(ctx, dst_slots[i], out_dims[i] * sizeof(float), "slot_matvec_group_dst_malloc", error, error_cap)) {
            result = 0;
            goto grouped_cleanup;
        }
        dst_ptrs[i] = ctx->slots[dst_slots[i]].device_ptr;
    }
    x_q8_bytes = minicpmo_gpu_q8_1_super_bytes_for_cols(in_dim);
    if (x_q8_bytes == 0) {
        minicpmo_gpu_set_error(ctx, error, error_cap, "error: gpu grouped native quant matvec expects in_dim multiple of 256");
        result = 0;
        goto grouped_cleanup;
    }
    if (!minicpmo_gpu_ensure_scratch(ctx, &ctx->scratch_x, &ctx->scratch_x_bytes, x_q8_bytes, "slot_group_q8_1_malloc", error, error_cap)) {
        result = 0;
        goto grouped_cleanup;
    }
    ctx->stats.scratch_x_bytes = (uint64_t) ctx->scratch_x_bytes;
    cache_hit = ctx->q8_1_cache_valid &&
        ctx->q8_1_cache_slot == src_slot &&
        ctx->q8_1_cache_cols == in_dim &&
        ctx->scratch_x != 0 &&
        ctx->scratch_x_bytes >= x_q8_bytes;
    if (!cache_hit) {
        CUdeviceptr src_ptr = ctx->slots[src_slot].device_ptr;
        qargs[0] = &src_ptr;
        qargs[1] = &ctx->scratch_x;
        qargs[2] = &cols_i;
        if (!minicpmo_gpu_cuda_check(
                ctx,
                ctx->cuda_launch_kernel(
                    ctx->q8_1_quantize_fn,
                    (unsigned int) (in_dim / MINICPMO_GPU_QK_K),
                    1,
                    1,
                    MINICPMO_GPU_ATTENTION_THREADS,
                    1,
                    1,
                    0,
                    NULL,
                    qargs,
                    NULL),
                "launch_group_q8_1_quantize",
                error,
                error_cap)) {
            result = 0;
            goto grouped_cleanup;
        }
        ctx->q8_1_cache_valid = 1;
        ctx->q8_1_cache_slot = src_slot;
        ctx->q8_1_cache_cols = in_dim;
        ctx->stats.q8_1_x_bytes += (uint64_t) x_q8_bytes;
    }
    if (!minicpmo_gpu_launch_quant_matvec_multi_to(ctx, uploads, count, ctx->scratch_x, dst_ptrs, out_dims, in_dim, error, error_cap)) {
        result = 0;
        goto grouped_cleanup;
    }
    ctx->stats.native_quant_matvec_count += (uint64_t) count;
    for (i = 0; i < count; ++i) {
        minicpmo_gpu_mark_slot_written(ctx, dst_slots[i]);
    }

grouped_cleanup:
    for (i = 0; i < count; ++i) {
        if (release_after_use[i] && uploads[i] != NULL && uploads[i]->device_ptr != 0) {
            minicpmo_gpu_release_upload(ctx, uploads[i]);
        }
    }
    return result;
}

static int minicpmo_gpu_context_slot_matvec_multi(
    void *handle,
    size_t count,
    unsigned int *dst_slots,
    const char **tensor_names,
    const void **host_ptrs,
    int *dtypes,
    size_t *out_dims,
    unsigned int src_slot,
    size_t in_dim,
    char *error,
    size_t error_cap) {
    minicpmo_gpu_context *ctx = (minicpmo_gpu_context *) handle;
    int native_result;
    if (ctx == NULL) {
        minicpmo_gpu_copy_error(error, error_cap, "error: null gpu grouped slot matvec context");
        return 0;
    }
    if (count < 2u || count > 3u) {
        minicpmo_gpu_set_error(ctx, error, error_cap, "error: invalid grouped gpu slot matvec count");
        return 0;
    }
    if (dst_slots == NULL || tensor_names == NULL || host_ptrs == NULL || dtypes == NULL || out_dims == NULL) {
        minicpmo_gpu_set_error(ctx, error, error_cap, "error: invalid grouped gpu slot matvec arguments");
        return 0;
    }
    if ((dtypes[0] == 10 || dtypes[0] == 12) && minicpmo_gpu_native_kmatvec_requested(ctx, dtypes[0])) {
        char native_error[512];
        native_error[0] = '\0';
        native_result = minicpmo_gpu_context_slot_matvec_multi_native(ctx, count, dst_slots, tensor_names, host_ptrs, dtypes, out_dims, src_slot, in_dim, native_error, sizeof(native_error));
        if (native_result == 1) {
            return 1;
        }
        if (native_result == 0) {
            minicpmo_gpu_copy_error(error, error_cap, native_error);
            minicpmo_gpu_copy_error(ctx->last_error, sizeof(ctx->last_error), native_error);
            return 0;
        }
        if (ctx->debug && native_error[0] != '\0') {
            fprintf(stderr, "minicpmo-gpu: grouped native quant unavailable falling_back=slot_matvec reason=%s\n", native_error);
        }
    }
    return minicpmo_gpu_context_slot_matvec_multi_fallback(handle, count, dst_slots, tensor_names, host_ptrs, dtypes, out_dims, src_slot, in_dim, error, error_cap);
}

int minicpmo_gpu_context_slot_matvec2(
    void *handle,
    unsigned int dst_slot0,
    const char *tensor_name0,
    const void *host_ptr0,
    int dtype0,
    size_t out_dim0,
    unsigned int dst_slot1,
    const char *tensor_name1,
    const void *host_ptr1,
    int dtype1,
    size_t out_dim1,
    unsigned int src_slot,
    size_t in_dim,
    char *error,
    size_t error_cap) {
    unsigned int dst_slots[3] = {dst_slot0, dst_slot1, 0};
    const char *tensor_names[3] = {tensor_name0, tensor_name1, NULL};
    const void *host_ptrs[3] = {host_ptr0, host_ptr1, NULL};
    int dtypes[3] = {dtype0, dtype1, 0};
    size_t out_dims[3] = {out_dim0, out_dim1, 0};
    return minicpmo_gpu_context_slot_matvec_multi(handle, 2u, dst_slots, tensor_names, host_ptrs, dtypes, out_dims, src_slot, in_dim, error, error_cap);
}

int minicpmo_gpu_context_slot_matvec3(
    void *handle,
    unsigned int dst_slot0,
    const char *tensor_name0,
    const void *host_ptr0,
    int dtype0,
    size_t out_dim0,
    unsigned int dst_slot1,
    const char *tensor_name1,
    const void *host_ptr1,
    int dtype1,
    size_t out_dim1,
    unsigned int dst_slot2,
    const char *tensor_name2,
    const void *host_ptr2,
    int dtype2,
    size_t out_dim2,
    unsigned int src_slot,
    size_t in_dim,
    char *error,
    size_t error_cap) {
    unsigned int dst_slots[3] = {dst_slot0, dst_slot1, dst_slot2};
    const char *tensor_names[3] = {tensor_name0, tensor_name1, tensor_name2};
    const void *host_ptrs[3] = {host_ptr0, host_ptr1, host_ptr2};
    int dtypes[3] = {dtype0, dtype1, dtype2};
    size_t out_dims[3] = {out_dim0, out_dim1, out_dim2};
    return minicpmo_gpu_context_slot_matvec_multi(handle, 3u, dst_slots, tensor_names, host_ptrs, dtypes, out_dims, src_slot, in_dim, error, error_cap);
}

int minicpmo_gpu_context_slot_rms_norm(
    void *handle,
    unsigned int dst_slot,
    unsigned int src_slot,
    const void *weight_host_ptr,
    size_t n,
    float eps,
    char *error,
    size_t error_cap) {
    minicpmo_gpu_context *ctx = (minicpmo_gpu_context *) handle;
    minicpmo_gpu_upload *upload;
    int release_after_use = 0;
    int weight_dtype_i;
    int n_i;
    void *args[6];
    size_t bytes = n * sizeof(float);
    if (ctx == NULL) {
        minicpmo_gpu_copy_error(error, error_cap, "error: null gpu rms norm context");
        return 0;
    }
    if (weight_host_ptr == NULL || n == 0 || src_slot >= MINICPMO_GPU_SLOT_COUNT || dst_slot >= MINICPMO_GPU_SLOT_COUNT) {
        minicpmo_gpu_set_error(ctx, error, error_cap, "error: invalid gpu rms norm arguments");
        return 0;
    }
    upload = minicpmo_gpu_find_upload(ctx, weight_host_ptr);
    if (upload == NULL) {
        minicpmo_gpu_set_error(ctx, error, error_cap, "error: gpu rms norm weight not uploaded");
        return 0;
    }
    if (ctx->slots[src_slot].device_ptr == 0 || ctx->slots[src_slot].bytes < bytes) {
        minicpmo_gpu_set_error(ctx, error, error_cap, "error: gpu rms norm source slot not ready");
        return 0;
    }
    if (!minicpmo_gpu_prepare_quant_kernels(ctx, error, error_cap) || ctx->rms_norm_fn == NULL) {
        return 0;
    }
    if (!minicpmo_gpu_ensure_upload_device(ctx, upload, upload->name, &release_after_use, error, error_cap) ||
        !minicpmo_gpu_ensure_slot(ctx, dst_slot, bytes, "slot_rms_norm_malloc", error, error_cap)) {
        return 0;
    }
    if (upload->upload_dtype != 0 && upload->upload_dtype != 1) {
        minicpmo_gpu_set_error(ctx, error, error_cap, "error: gpu rms norm weight dtype unsupported");
        if (release_after_use && upload->device_ptr != 0) {
            minicpmo_gpu_release_upload(ctx, upload);
        }
        return 0;
    }
    weight_dtype_i = upload->upload_dtype;
    n_i = (int) n;
    args[0] = &ctx->slots[dst_slot].device_ptr;
    args[1] = &ctx->slots[src_slot].device_ptr;
    args[2] = &upload->device_ptr;
    args[3] = &weight_dtype_i;
    args[4] = &n_i;
    args[5] = &eps;
    if (!minicpmo_gpu_cuda_check(
            ctx,
            ctx->cuda_launch_kernel(
                ctx->rms_norm_fn,
                1,
                1,
                1,
                MINICPMO_GPU_VECTOR_THREADS,
                1,
                1,
                0,
                NULL,
                args,
                NULL),
            "launch_rms_norm",
            error,
            error_cap)) {
        if (release_after_use && upload->device_ptr != 0) {
            minicpmo_gpu_release_upload(ctx, upload);
        }
        return 0;
    }
    minicpmo_gpu_mark_slot_written(ctx, dst_slot);
    if (release_after_use && upload->device_ptr != 0) {
        minicpmo_gpu_release_upload(ctx, upload);
    }
    return 1;
}

int minicpmo_gpu_context_slot_head_norm(
    void *handle,
    unsigned int slot,
    const void *weight_host_ptr,
    size_t head_count,
    size_t head_dim,
    char *error,
    size_t error_cap) {
    minicpmo_gpu_context *ctx = (minicpmo_gpu_context *) handle;
    minicpmo_gpu_upload *upload;
    int release_after_use = 0;
    int weight_dtype_i;
    int head_dim_i;
    void *args[4];
    size_t bytes = head_count * head_dim * sizeof(float);
    if (ctx == NULL) {
        minicpmo_gpu_copy_error(error, error_cap, "error: null gpu head norm context");
        return 0;
    }
    if (weight_host_ptr == NULL || head_count == 0 || head_dim == 0 || head_dim > MINICPMO_GPU_VECTOR_THREADS || slot >= MINICPMO_GPU_SLOT_COUNT) {
        minicpmo_gpu_set_error(ctx, error, error_cap, "error: invalid gpu head norm arguments");
        return 0;
    }
    upload = minicpmo_gpu_find_upload(ctx, weight_host_ptr);
    if (upload == NULL) {
        minicpmo_gpu_set_error(ctx, error, error_cap, "error: gpu head norm weight not uploaded");
        return 0;
    }
    if (ctx->slots[slot].device_ptr == 0 || ctx->slots[slot].bytes < bytes) {
        minicpmo_gpu_set_error(ctx, error, error_cap, "error: gpu head norm slot not ready");
        return 0;
    }
    if (!minicpmo_gpu_prepare_quant_kernels(ctx, error, error_cap) || ctx->head_norm_fn == NULL) {
        return 0;
    }
    if (!minicpmo_gpu_ensure_upload_device(ctx, upload, upload->name, &release_after_use, error, error_cap)) {
        return 0;
    }
    if (upload->upload_dtype != 0 && upload->upload_dtype != 1) {
        minicpmo_gpu_set_error(ctx, error, error_cap, "error: gpu head norm weight dtype unsupported");
        if (release_after_use && upload->device_ptr != 0) {
            minicpmo_gpu_release_upload(ctx, upload);
        }
        return 0;
    }
    weight_dtype_i = upload->upload_dtype;
    head_dim_i = (int) head_dim;
    args[0] = &ctx->slots[slot].device_ptr;
    args[1] = &upload->device_ptr;
    args[2] = &weight_dtype_i;
    args[3] = &head_dim_i;
    if (!minicpmo_gpu_cuda_check(
            ctx,
            ctx->cuda_launch_kernel(
                ctx->head_norm_fn,
                (unsigned int) head_count,
                1,
                1,
                MINICPMO_GPU_VECTOR_THREADS,
                1,
                1,
                0,
                NULL,
                args,
                NULL),
            "launch_head_norm",
            error,
            error_cap)) {
        if (release_after_use && upload->device_ptr != 0) {
            minicpmo_gpu_release_upload(ctx, upload);
        }
        return 0;
    }
    minicpmo_gpu_mark_slot_written(ctx, slot);
    if (release_after_use && upload->device_ptr != 0) {
        minicpmo_gpu_release_upload(ctx, upload);
    }
    return 1;
}

int minicpmo_gpu_context_slot_rope(
    void *handle,
    unsigned int slot,
    size_t head_count,
    size_t head_dim,
    size_t pos,
    float freq_base,
    float freq_scale,
    const float *rope_cos,
    const float *rope_sin,
    size_t rope_context,
    char *error,
    size_t error_cap) {
    minicpmo_gpu_context *ctx = (minicpmo_gpu_context *) handle;
    int head_count_i;
    int head_dim_i;
    int pos_i;
    CUdeviceptr rope_cos_device = 0;
    CUdeviceptr rope_sin_device = 0;
    void *args[8];
    size_t bytes = head_count * head_dim * sizeof(float);
    size_t half_dim = head_dim / 2u;
    if (ctx == NULL) {
        minicpmo_gpu_copy_error(error, error_cap, "error: null gpu rope context");
        return 0;
    }
    if (head_count == 0 || head_dim == 0 || slot >= MINICPMO_GPU_SLOT_COUNT || half_dim > MINICPMO_GPU_VECTOR_THREADS || ctx->slots[slot].device_ptr == 0 || ctx->slots[slot].bytes < bytes) {
        minicpmo_gpu_set_error(ctx, error, error_cap, "error: invalid gpu rope arguments");
        return 0;
    }
    if (!minicpmo_gpu_prepare_quant_kernels(ctx, error, error_cap) || ctx->rope_fn == NULL) {
        return 0;
    }
    if (rope_cos != NULL && rope_sin != NULL) {
        if (pos >= rope_context) {
            minicpmo_gpu_set_error(ctx, error, error_cap, "error: gpu rope position exceeds table context");
            return 0;
        }
        if (!minicpmo_gpu_ensure_rope_table(ctx, rope_cos, rope_sin, rope_context, half_dim, error, error_cap)) {
            return 0;
        }
        rope_cos_device = ctx->device_rope_cos;
        rope_sin_device = ctx->device_rope_sin;
    }
    head_count_i = (int) head_count;
    head_dim_i = (int) head_dim;
    pos_i = (int) pos;
    args[0] = &ctx->slots[slot].device_ptr;
    args[1] = &head_count_i;
    args[2] = &head_dim_i;
    args[3] = &pos_i;
    args[4] = &freq_base;
    args[5] = &freq_scale;
    args[6] = &rope_cos_device;
    args[7] = &rope_sin_device;
    if (!minicpmo_gpu_cuda_check(
            ctx,
            ctx->cuda_launch_kernel(
                ctx->rope_fn,
                (unsigned int) head_count,
                1,
                1,
                MINICPMO_GPU_VECTOR_THREADS,
                1,
                1,
                0,
                NULL,
                args,
                NULL),
            "launch_rope",
            error,
            error_cap)) {
        return 0;
    }
    minicpmo_gpu_mark_slot_written(ctx, slot);
    return 1;
}

int minicpmo_gpu_context_slot_add_inplace(
    void *handle,
    unsigned int dst_slot,
    unsigned int src_slot,
    size_t len,
    char *error,
    size_t error_cap) {
    minicpmo_gpu_context *ctx = (minicpmo_gpu_context *) handle;
    int n_i;
    void *args[3];
    size_t bytes = len * sizeof(float);
    if (ctx == NULL) {
        minicpmo_gpu_copy_error(error, error_cap, "error: null gpu add context");
        return 0;
    }
    if (len == 0 || dst_slot >= MINICPMO_GPU_SLOT_COUNT || src_slot >= MINICPMO_GPU_SLOT_COUNT ||
        ctx->slots[dst_slot].device_ptr == 0 || ctx->slots[src_slot].device_ptr == 0 ||
        ctx->slots[dst_slot].bytes < bytes || ctx->slots[src_slot].bytes < bytes) {
        minicpmo_gpu_set_error(ctx, error, error_cap, "error: invalid gpu add arguments");
        return 0;
    }
    if (!minicpmo_gpu_prepare_quant_kernels(ctx, error, error_cap) || ctx->add_inplace_fn == NULL) {
        return 0;
    }
    n_i = (int) len;
    args[0] = &ctx->slots[dst_slot].device_ptr;
    args[1] = &ctx->slots[src_slot].device_ptr;
    args[2] = &n_i;
    if (!minicpmo_gpu_cuda_check(
            ctx,
            ctx->cuda_launch_kernel(
                ctx->add_inplace_fn,
                (unsigned int) ((len + (MINICPMO_GPU_VECTOR_THREADS - 1u)) / MINICPMO_GPU_VECTOR_THREADS),
                1,
                1,
                MINICPMO_GPU_VECTOR_THREADS,
                1,
                1,
                0,
                NULL,
                args,
                NULL),
            "launch_add_inplace",
            error,
            error_cap)) {
        return 0;
    }
    minicpmo_gpu_mark_slot_written(ctx, dst_slot);
    return 1;
}

int minicpmo_gpu_context_slot_silu_mul(
    void *handle,
    unsigned int dst_slot,
    unsigned int gate_slot,
    unsigned int up_slot,
    size_t len,
    char *error,
    size_t error_cap) {
    minicpmo_gpu_context *ctx = (minicpmo_gpu_context *) handle;
    int n_i;
    void *args[4];
    size_t bytes = len * sizeof(float);
    if (ctx == NULL) {
        minicpmo_gpu_copy_error(error, error_cap, "error: null gpu silu-mul context");
        return 0;
    }
    if (len == 0 || dst_slot >= MINICPMO_GPU_SLOT_COUNT || gate_slot >= MINICPMO_GPU_SLOT_COUNT || up_slot >= MINICPMO_GPU_SLOT_COUNT ||
        ctx->slots[gate_slot].device_ptr == 0 || ctx->slots[up_slot].device_ptr == 0 ||
        ctx->slots[gate_slot].bytes < bytes || ctx->slots[up_slot].bytes < bytes) {
        minicpmo_gpu_set_error(ctx, error, error_cap, "error: invalid gpu silu-mul arguments");
        return 0;
    }
    if (!minicpmo_gpu_prepare_quant_kernels(ctx, error, error_cap) || ctx->silu_mul_fn == NULL) {
        return 0;
    }
    if (!minicpmo_gpu_ensure_slot(ctx, dst_slot, bytes, "slot_silu_mul_malloc", error, error_cap)) {
        return 0;
    }
    n_i = (int) len;
    args[0] = &ctx->slots[dst_slot].device_ptr;
    args[1] = &ctx->slots[gate_slot].device_ptr;
    args[2] = &ctx->slots[up_slot].device_ptr;
    args[3] = &n_i;
    if (!minicpmo_gpu_cuda_check(
            ctx,
            ctx->cuda_launch_kernel(
                ctx->silu_mul_fn,
                (unsigned int) ((len + (MINICPMO_GPU_VECTOR_THREADS - 1u)) / MINICPMO_GPU_VECTOR_THREADS),
                1,
                1,
                MINICPMO_GPU_VECTOR_THREADS,
                1,
                1,
                0,
                NULL,
                args,
                NULL),
            "launch_silu_mul",
            error,
            error_cap)) {
        return 0;
    }
    minicpmo_gpu_mark_slot_written(ctx, dst_slot);
    return 1;
}

int minicpmo_gpu_context_attention_slots(
    void *handle,
    unsigned int dst_slot,
    unsigned int q_slot,
    unsigned int k_slot,
    unsigned int v_slot,
    size_t layer,
    size_t pos,
    size_t head_count,
    size_t kv_head_count,
    size_t head_dim,
    size_t context_capacity,
    char *error,
    size_t error_cap) {
    minicpmo_gpu_context *ctx = (minicpmo_gpu_context *) handle;
    size_t q_dim;
    size_t q_bytes;
    size_t kv_bytes;
    int layer_i;
    int pos_i;
    int head_count_i;
    int kv_head_count_i;
    int head_dim_i;
    int context_capacity_i;
    float scale;
    void *args[11];
    if (ctx == NULL) {
        minicpmo_gpu_copy_error(error, error_cap, "error: null gpu attention slot context");
        return 0;
    }
    if (dst_slot >= MINICPMO_GPU_SLOT_COUNT || q_slot >= MINICPMO_GPU_SLOT_COUNT || k_slot >= MINICPMO_GPU_SLOT_COUNT || v_slot >= MINICPMO_GPU_SLOT_COUNT ||
        head_count == 0 || kv_head_count == 0 || head_dim == 0) {
        minicpmo_gpu_set_error(ctx, error, error_cap, "error: invalid gpu attention slot arguments");
        return 0;
    }
    if (!minicpmo_gpu_prepare_quant_kernels(ctx, error, error_cap) || ctx->attention_decode_fn == NULL) {
        return 0;
    }
    if (ctx->device_k_cache == 0 || ctx->device_v_cache == 0 || ctx->kv_dim == 0) {
        minicpmo_gpu_set_error(ctx, error, error_cap, "error: gpu attention cache not prepared");
        return 0;
    }
    q_dim = head_count * head_dim;
    q_bytes = q_dim * sizeof(float);
    kv_bytes = (kv_head_count * head_dim) * sizeof(float);
    if (ctx->slots[q_slot].device_ptr == 0 || ctx->slots[k_slot].device_ptr == 0 || ctx->slots[v_slot].device_ptr == 0 ||
        ctx->slots[q_slot].bytes < q_bytes || ctx->slots[k_slot].bytes < kv_bytes || ctx->slots[v_slot].bytes < kv_bytes) {
        minicpmo_gpu_set_error(ctx, error, error_cap, "error: gpu attention slots not ready");
        return 0;
    }
    if (!minicpmo_gpu_ensure_slot(ctx, dst_slot, q_bytes, "slot_attention_dst_malloc", error, error_cap)) {
        return 0;
    }
    if (layer >= ctx->kv_layer_count || pos >= ctx->kv_context_capacity || context_capacity != ctx->kv_context_capacity) {
        minicpmo_gpu_set_error(ctx, error, error_cap, "error: gpu attention cache shape mismatch");
        return 0;
    }
    if (kv_head_count * head_dim != ctx->kv_dim) {
        minicpmo_gpu_set_error(ctx, error, error_cap, "error: gpu attention kv_dim mismatch");
        return 0;
    }
    {
        size_t kv_offset = ((layer * ctx->kv_context_capacity) + pos) * ctx->kv_dim * sizeof(float);
        if (!minicpmo_gpu_cuda_check(ctx, ctx->cuda_memcpy_dto_d(ctx->device_k_cache + kv_offset, ctx->slots[k_slot].device_ptr, kv_bytes), "attention_k_copy_d2d", error, error_cap) ||
            !minicpmo_gpu_cuda_check(ctx, ctx->cuda_memcpy_dto_d(ctx->device_v_cache + kv_offset, ctx->slots[v_slot].device_ptr, kv_bytes), "attention_v_copy_d2d", error, error_cap)) {
            return 0;
        }
    }
    layer_i = (int) layer;
    pos_i = (int) pos;
    head_count_i = (int) head_count;
    kv_head_count_i = (int) kv_head_count;
    head_dim_i = (int) head_dim;
    context_capacity_i = (int) context_capacity;
    scale = 1.0f / minicpmo_gpu_sqrtf_positive((float) head_dim);
    args[0] = &ctx->slots[q_slot].device_ptr;
    args[1] = &ctx->device_k_cache;
    args[2] = &ctx->device_v_cache;
    args[3] = &ctx->slots[dst_slot].device_ptr;
    args[4] = &layer_i;
    args[5] = &pos_i;
    args[6] = &head_count_i;
    args[7] = &kv_head_count_i;
    args[8] = &head_dim_i;
    args[9] = &context_capacity_i;
    args[10] = &scale;
    if (!minicpmo_gpu_cuda_check(
            ctx,
            ctx->cuda_launch_kernel(
                ctx->attention_decode_fn,
                (unsigned int) head_count,
                1,
                1,
                MINICPMO_GPU_ATTENTION_THREADS,
                1,
                1,
                0,
                NULL,
                args,
                NULL),
            "launch_attention_decode_slots",
            error,
            error_cap)) {
        return 0;
    }
    minicpmo_gpu_mark_slot_written(ctx, dst_slot);
    ctx->stats.attention_kernel_calls += 1u;
    return 1;
}

void minicpmo_gpu_context_snapshot(void *handle, minicpmo_gpu_stats *stats) {
    minicpmo_gpu_context *ctx = (minicpmo_gpu_context *) handle;
    if (ctx == NULL || stats == NULL) {
        return;
    }
    ctx->stats.scratch_x_bytes = (uint64_t) ctx->scratch_x_bytes;
    ctx->stats.scratch_y_bytes = (uint64_t) ctx->scratch_y_bytes;
    *stats = ctx->stats;
}
