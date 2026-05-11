#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include <pthread.h>

#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#endif

#define QWEN3_C_MAX_WORKERS 63

struct qwen3_q8_pool {
    int initialized;
    int stop;
    size_t worker_count;
    uint64_t generation;
    size_t remaining;
    pthread_t workers[QWEN3_C_MAX_WORKERS];
    size_t worker_ids[QWEN3_C_MAX_WORKERS];
    pthread_mutex_t mutex;
    pthread_cond_t start_cond;
    pthread_cond_t done_cond;
    float *dst;
    const uint8_t *matrix_data;
    const float *x;
    size_t in_dim;
    size_t out_dim;
    size_t total_threads;
};

static struct qwen3_q8_pool qwen3_pool;

static inline float qwen3_u32_as_f32(uint32_t bits) {
    float value;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

static inline float qwen3_f16_to_f32(uint16_t bits) {
    const uint32_t sign = ((uint32_t)bits & 0x8000u) << 16;
    uint32_t exp = ((uint32_t)bits >> 10) & 0x1fu;
    uint32_t frac = (uint32_t)bits & 0x03ffu;

    if (exp == 0) {
        if (frac == 0) {
            return qwen3_u32_as_f32(sign);
        }
        int shift = 0;
        while ((frac & 0x0400u) == 0) {
            frac <<= 1;
            shift++;
        }
        frac &= 0x03ffu;
        exp = (uint32_t)(127 - 14 - shift);
        return qwen3_u32_as_f32(sign | (exp << 23) | (frac << 13));
    }

    if (exp == 0x1fu) {
        return qwen3_u32_as_f32(sign | 0x7f800000u | (frac << 13));
    }

    return qwen3_u32_as_f32(sign | ((exp + 112u) << 23) | (frac << 13));
}

static inline uint16_t qwen3_read_u16_le(const uint8_t *p) {
    return (uint16_t)p[0] | (uint16_t)((uint16_t)p[1] << 8);
}

#if (defined(__x86_64__) || defined(__i386__)) && defined(__AVX2__)
static inline __m256 qwen3_q8_mul_8(const int8_t *q, const float *x, float scale) {
    const __m128i q8 = _mm_loadl_epi64((const __m128i *)q);
    const __m256 qf = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q8));
    return _mm256_mul_ps(_mm256_mul_ps(qf, _mm256_set1_ps(scale)), _mm256_loadu_ps(x));
}

static inline float qwen3_hsum256_ps(__m256 v) {
    const __m128 lo = _mm256_castps256_ps128(v);
    const __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 sum = _mm_add_ps(lo, hi);
    sum = _mm_add_ps(sum, _mm_movehl_ps(sum, sum));
    sum = _mm_add_ss(sum, _mm_shuffle_ps(sum, sum, 0x55));
    return _mm_cvtss_f32(sum);
}
#endif

void qwen3_q8_0_matvec_range_c(float *dst, const uint8_t *matrix_data, const float *x, size_t in_dim, size_t out_start, size_t out_end) {
    const size_t blocks_per_row = in_dim / 32;
    const size_t row_bytes = blocks_per_row * 34;

    for (size_t j = out_start; j < out_end; ++j) {
        const uint8_t *row = matrix_data + j * row_bytes;
#if (defined(__x86_64__) || defined(__i386__)) && defined(__AVX2__)
        __m256 acc = _mm256_setzero_ps();
        for (size_t b = 0; b < blocks_per_row; ++b) {
            const uint8_t *block = row + b * 34;
            const int8_t *qs = (const int8_t *)(block + 2);
            const float scale = qwen3_f16_to_f32(qwen3_read_u16_le(block));
            const float *xb = x + b * 32;
            acc = _mm256_add_ps(acc, qwen3_q8_mul_8(qs + 0, xb + 0, scale));
            acc = _mm256_add_ps(acc, qwen3_q8_mul_8(qs + 8, xb + 8, scale));
            acc = _mm256_add_ps(acc, qwen3_q8_mul_8(qs + 16, xb + 16, scale));
            acc = _mm256_add_ps(acc, qwen3_q8_mul_8(qs + 24, xb + 24, scale));
        }
        dst[j] = qwen3_hsum256_ps(acc);
#else
        float sum = 0.0f;
        for (size_t b = 0; b < blocks_per_row; ++b) {
            const uint8_t *block = row + b * 34;
            const int8_t *qs = (const int8_t *)(block + 2);
            const float scale = qwen3_f16_to_f32(qwen3_read_u16_le(block));
            const float *xb = x + b * 32;
            for (size_t k = 0; k < 32; ++k) {
                sum += xb[k] * scale * (float)qs[k];
            }
        }
        dst[j] = sum;
#endif
    }
}

static void qwen3_q8_0_matvec_shard_c(float *dst, const uint8_t *matrix_data, const float *x, size_t in_dim, size_t out_dim, size_t shard, size_t shard_count) {
    const size_t start = (out_dim * shard) / shard_count;
    const size_t end = (out_dim * (shard + 1)) / shard_count;
    qwen3_q8_0_matvec_range_c(dst, matrix_data, x, in_dim, start, end);
}

static void *qwen3_q8_worker_main(void *arg) {
    const size_t worker_id = *(const size_t *)arg;
    uint64_t seen_generation = 0;

    for (;;) {
        pthread_mutex_lock(&qwen3_pool.mutex);
        while (!qwen3_pool.stop && seen_generation == qwen3_pool.generation) {
            pthread_cond_wait(&qwen3_pool.start_cond, &qwen3_pool.mutex);
        }
        if (qwen3_pool.stop) {
            pthread_mutex_unlock(&qwen3_pool.mutex);
            return NULL;
        }

        seen_generation = qwen3_pool.generation;
        float *dst = qwen3_pool.dst;
        const uint8_t *matrix_data = qwen3_pool.matrix_data;
        const float *x = qwen3_pool.x;
        const size_t in_dim = qwen3_pool.in_dim;
        const size_t out_dim = qwen3_pool.out_dim;
        const size_t total_threads = qwen3_pool.total_threads;
        pthread_mutex_unlock(&qwen3_pool.mutex);

        qwen3_q8_0_matvec_shard_c(dst, matrix_data, x, in_dim, out_dim, worker_id, total_threads);

        pthread_mutex_lock(&qwen3_pool.mutex);
        qwen3_pool.remaining--;
        if (qwen3_pool.remaining == 0) {
            pthread_cond_signal(&qwen3_pool.done_cond);
        }
        pthread_mutex_unlock(&qwen3_pool.mutex);
    }
}

void qwen3_q8_0_thread_pool_shutdown_c(void) {
    if (!qwen3_pool.initialized) {
        return;
    }

    pthread_mutex_lock(&qwen3_pool.mutex);
    qwen3_pool.stop = 1;
    qwen3_pool.generation++;
    pthread_cond_broadcast(&qwen3_pool.start_cond);
    pthread_mutex_unlock(&qwen3_pool.mutex);

    for (size_t i = 0; i < qwen3_pool.worker_count; ++i) {
        pthread_join(qwen3_pool.workers[i], NULL);
    }

    pthread_cond_destroy(&qwen3_pool.start_cond);
    pthread_cond_destroy(&qwen3_pool.done_cond);
    pthread_mutex_destroy(&qwen3_pool.mutex);
    memset(&qwen3_pool, 0, sizeof(qwen3_pool));
}

static int qwen3_q8_0_thread_pool_ensure(size_t worker_count) {
    if (worker_count > QWEN3_C_MAX_WORKERS) {
        worker_count = QWEN3_C_MAX_WORKERS;
    }
    if (qwen3_pool.initialized && qwen3_pool.worker_count == worker_count) {
        return 1;
    }

    qwen3_q8_0_thread_pool_shutdown_c();
    memset(&qwen3_pool, 0, sizeof(qwen3_pool));
    qwen3_pool.worker_count = worker_count;
    if (pthread_mutex_init(&qwen3_pool.mutex, NULL) != 0) {
        return 0;
    }
    if (pthread_cond_init(&qwen3_pool.start_cond, NULL) != 0) {
        pthread_mutex_destroy(&qwen3_pool.mutex);
        memset(&qwen3_pool, 0, sizeof(qwen3_pool));
        return 0;
    }
    if (pthread_cond_init(&qwen3_pool.done_cond, NULL) != 0) {
        pthread_cond_destroy(&qwen3_pool.start_cond);
        pthread_mutex_destroy(&qwen3_pool.mutex);
        memset(&qwen3_pool, 0, sizeof(qwen3_pool));
        return 0;
    }

    for (size_t i = 0; i < worker_count; ++i) {
        qwen3_pool.worker_ids[i] = i + 1;
        if (pthread_create(&qwen3_pool.workers[i], NULL, qwen3_q8_worker_main, &qwen3_pool.worker_ids[i]) != 0) {
            qwen3_pool.worker_count = i;
            qwen3_pool.initialized = 1;
            qwen3_q8_0_thread_pool_shutdown_c();
            return 0;
        }
    }

    qwen3_pool.initialized = 1;
    return 1;
}

int32_t qwen3_q8_0_matvec_parallel_c(float *dst, const uint8_t *matrix_data, const float *x, size_t in_dim, size_t out_dim, size_t thread_count) {
    if (thread_count < 2 || out_dim < 64) {
        return 0;
    }
    if (thread_count > QWEN3_C_MAX_WORKERS + 1) {
        thread_count = QWEN3_C_MAX_WORKERS + 1;
    }
    if (thread_count > out_dim) {
        thread_count = out_dim;
    }

    const size_t worker_count = thread_count - 1;
    if (!qwen3_q8_0_thread_pool_ensure(worker_count)) {
        return 0;
    }

    pthread_mutex_lock(&qwen3_pool.mutex);
    qwen3_pool.dst = dst;
    qwen3_pool.matrix_data = matrix_data;
    qwen3_pool.x = x;
    qwen3_pool.in_dim = in_dim;
    qwen3_pool.out_dim = out_dim;
    qwen3_pool.total_threads = thread_count;
    qwen3_pool.remaining = worker_count;
    qwen3_pool.generation++;
    pthread_cond_broadcast(&qwen3_pool.start_cond);
    pthread_mutex_unlock(&qwen3_pool.mutex);

    qwen3_q8_0_matvec_shard_c(dst, matrix_data, x, in_dim, out_dim, 0, thread_count);

    pthread_mutex_lock(&qwen3_pool.mutex);
    while (qwen3_pool.remaining != 0) {
        pthread_cond_wait(&qwen3_pool.done_cond, &qwen3_pool.mutex);
    }
    pthread_mutex_unlock(&qwen3_pool.mutex);
    return 1;
}
