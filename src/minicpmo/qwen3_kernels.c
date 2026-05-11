#include <stddef.h>
#include <stdint.h>
#include <string.h>

#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#endif

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
