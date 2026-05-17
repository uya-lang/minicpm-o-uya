#include <stddef.h>
#include <stdint.h>

#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#endif

#define MINICPMO_DTYPE_Q8_0 6
#define MINICPMO_DTYPE_Q4_K 10
#define MINICPMO_DTYPE_Q6_K 12

typedef struct {
    float d;
    float s;
    int8_t qs[32];
} minicpmo_block_q8_1;

static float minicpmo_f16_to_f32(uint16_t bits) {
    const uint32_t sign = ((uint32_t) bits & 0x8000u) << 16;
    const uint32_t exp_bits = ((uint32_t) bits >> 10) & 31u;
    const uint32_t frac = (uint32_t) bits & 1023u;
    uint32_t out_bits;
    float out;

    if (exp_bits == 0) {
        if (frac == 0) {
            out_bits = sign;
        } else {
            uint32_t mant = frac;
            int exp = -14;
            while ((mant & 1024u) == 0) {
                mant <<= 1;
                exp -= 1;
            }
            mant &= 1023u;
            out_bits = sign | ((uint32_t) (exp + 127) << 23) | (mant << 13);
        }
    } else if (exp_bits == 31) {
        out_bits = sign | 0x7f800000u | (frac << 13);
    } else {
        out_bits = sign | ((exp_bits + 112u) << 23) | (frac << 13);
    }

    __builtin_memcpy(&out, &out_bits, sizeof(out));
    return out;
}

static uint16_t minicpmo_u16_le(const uint8_t *p) {
    return (uint16_t) p[0] | ((uint16_t) p[1] << 8);
}

static void minicpmo_q4_k_scale_min(int j, const uint8_t *scales, uint8_t *scale, uint8_t *minv) {
    if (j < 4) {
        *scale = scales[j] & 63u;
        *minv = scales[j + 4] & 63u;
    } else {
        *scale = (uint8_t) ((scales[j + 4] & 15u) | ((scales[j - 4] >> 6) << 4));
        *minv = (uint8_t) ((scales[j + 4] >> 4) | ((scales[j] >> 6) << 4));
    }
}

static float minicpmo_q8_0_dot_scalar(const uint8_t *row, const float *x, size_t in_dim) {
    const size_t blocks = in_dim / 32u;
    float sum = 0.0f;
    size_t block;
    for (block = 0; block < blocks; ++block) {
        const uint8_t *b = row + block * 34u;
        const float d = minicpmo_f16_to_f32(minicpmo_u16_le(b));
        const int8_t *qs = (const int8_t *) (const void *) (b + 2u);
        const float *xb = x + block * 32u;
        size_t i;
        for (i = 0; i < 32u; ++i) {
            sum += xb[i] * d * (float) qs[i];
        }
    }
    return sum;
}

static float minicpmo_q4_k_dot_scalar(const uint8_t *row, const float *x, size_t in_dim) {
    const size_t blocks = in_dim / 256u;
    float sum = 0.0f;
    size_t block;
    for (block = 0; block < blocks; ++block) {
        const uint8_t *b = row + block * 144u;
        const float d = minicpmo_f16_to_f32(minicpmo_u16_le(b));
        const float dmin = minicpmo_f16_to_f32(minicpmo_u16_le(b + 2u));
        const uint8_t *scales = b + 4u;
        const uint8_t *qs = b + 16u;
        const float *xb = x + block * 256u;
        int is = 0;
        size_t group;
        for (group = 0; group < 4u; ++group) {
            uint8_t sc, mn;
            const size_t qbase = group * 32u;
            size_t l;
            minicpmo_q4_k_scale_min(is, scales, &sc, &mn);
            const float d1 = d * (float) sc;
            const float m1 = dmin * (float) mn;
            minicpmo_q4_k_scale_min(is + 1, scales, &sc, &mn);
            const float d2 = d * (float) sc;
            const float m2 = dmin * (float) mn;
            for (l = 0; l < 32u; ++l) {
                sum += xb[group * 64u + l] * (d1 * (float) (qs[qbase + l] & 15u) - m1);
            }
            for (l = 0; l < 32u; ++l) {
                sum += xb[group * 64u + 32u + l] * (d2 * (float) (qs[qbase + l] >> 4) - m2);
            }
            is += 2;
        }
    }
    return sum;
}

static float minicpmo_q6_k_dot_scalar(const uint8_t *row, const float *x, size_t in_dim) {
    const size_t blocks = in_dim / 256u;
    float sum = 0.0f;
    size_t block;
    for (block = 0; block < blocks; ++block) {
        const uint8_t *b = row + block * 210u;
        const uint8_t *ql = b;
        const uint8_t *qh = b + 128u;
        const int8_t *sc = (const int8_t *) (const void *) (b + 192u);
        const float d = minicpmo_f16_to_f32(minicpmo_u16_le(b + 208u));
        const float *xb = x + block * 256u;
        size_t n;
        for (n = 0; n < 256u; n += 128u) {
            size_t l;
            for (l = 0; l < 32u; ++l) {
                const size_t isv = l / 16u;
                const int q1 = (int) ((ql[n / 2u + l] & 15u) | (((qh[n / 4u + l] >> 0) & 3u) << 4)) - 32;
                const int q2 = (int) ((ql[n / 2u + l + 32u] & 15u) | (((qh[n / 4u + l] >> 2) & 3u) << 4)) - 32;
                const int q3 = (int) ((ql[n / 2u + l] >> 4) | (((qh[n / 4u + l] >> 4) & 3u) << 4)) - 32;
                const int q4 = (int) ((ql[n / 2u + l + 32u] >> 4) | (((qh[n / 4u + l] >> 6) & 3u) << 4)) - 32;
                sum += xb[n + l] * d * (float) sc[n / 16u + isv + 0u] * (float) q1;
                sum += xb[n + 32u + l] * d * (float) sc[n / 16u + isv + 2u] * (float) q2;
                sum += xb[n + 64u + l] * d * (float) sc[n / 16u + isv + 4u] * (float) q3;
                sum += xb[n + 96u + l] * d * (float) sc[n / 16u + isv + 6u] * (float) q4;
            }
        }
    }
    return sum;
}

static int8_t minicpmo_quant_i8(float v) {
    int q = (int) (v >= 0.0f ? v + 0.5f : v - 0.5f);
    if (q > 127) {
        q = 127;
    } else if (q < -127) {
        q = -127;
    }
    return (int8_t) q;
}

static int minicpmo_quantize_q8_1_scalar(minicpmo_block_q8_1 *dst, const float *x, size_t in_dim) {
    const size_t blocks = in_dim / 32u;
    size_t block;
    for (block = 0; block < blocks; ++block) {
        const float *xb = x + block * 32u;
        float amax = 0.0f;
        int sum = 0;
        size_t i;
        for (i = 0; i < 32u; ++i) {
            const float v = xb[i];
            const float av = v < 0.0f ? -v : v;
            if (av > amax) {
                amax = av;
            }
        }
        dst[block].d = amax / 127.0f;
        const float id = amax != 0.0f ? 127.0f / amax : 0.0f;
        for (i = 0; i < 32u; ++i) {
            const int8_t q = minicpmo_quant_i8(xb[i] * id);
            dst[block].qs[i] = q;
            sum += (int) q;
        }
        dst[block].s = dst[block].d * (float) sum;
    }
    return 1;
}

static int32_t minicpmo_dot_u8_i8_32_scalar(const uint8_t *q, const int8_t *x) {
    int32_t sum = 0;
    size_t i;
    for (i = 0; i < 32u; ++i) {
        sum += (int32_t) q[i] * (int32_t) x[i];
    }
    return sum;
}

static float minicpmo_q4_k_dot_q8_1_scalar(const uint8_t *row, const minicpmo_block_q8_1 *xq, size_t in_dim) {
    const size_t blocks = in_dim / 256u;
    float sum = 0.0f;
    size_t block;
    for (block = 0; block < blocks; ++block) {
        const uint8_t *b = row + block * 144u;
        const float d = minicpmo_f16_to_f32(minicpmo_u16_le(b));
        const float dmin = minicpmo_f16_to_f32(minicpmo_u16_le(b + 2u));
        const uint8_t *scales = b + 4u;
        const uint8_t *qs = b + 16u;
        const minicpmo_block_q8_1 *xqb = xq + block * 8u;
        int is = 0;
        size_t group;
        for (group = 0; group < 4u; ++group) {
            uint8_t sc, mn;
            uint8_t qtmp[32];
            const size_t qbase = group * 32u;
            size_t l;
            minicpmo_q4_k_scale_min(is, scales, &sc, &mn);
            for (l = 0; l < 32u; ++l) {
                qtmp[l] = qs[qbase + l] & 15u;
            }
            sum += d * (float) sc * xqb[group * 2u].d * (float) minicpmo_dot_u8_i8_32_scalar(qtmp, xqb[group * 2u].qs);
            sum -= dmin * (float) mn * xqb[group * 2u].s;

            minicpmo_q4_k_scale_min(is + 1, scales, &sc, &mn);
            for (l = 0; l < 32u; ++l) {
                qtmp[l] = qs[qbase + l] >> 4;
            }
            sum += d * (float) sc * xqb[group * 2u + 1u].d * (float) minicpmo_dot_u8_i8_32_scalar(qtmp, xqb[group * 2u + 1u].qs);
            sum -= dmin * (float) mn * xqb[group * 2u + 1u].s;
            is += 2;
        }
    }
    return sum;
}

static float minicpmo_q6_k_dot_q8_1_scalar(const uint8_t *row, const minicpmo_block_q8_1 *xq, size_t in_dim) {
    const size_t blocks = in_dim / 256u;
    float sum = 0.0f;
    size_t block;
    for (block = 0; block < blocks; ++block) {
        const uint8_t *b = row + block * 210u;
        const uint8_t *ql = b;
        const uint8_t *qh = b + 128u;
        const int8_t *sc = (const int8_t *) (const void *) (b + 192u);
        const float d = minicpmo_f16_to_f32(minicpmo_u16_le(b + 208u));
        const minicpmo_block_q8_1 *xqb = xq + block * 8u;
        size_t n;
        for (n = 0; n < 256u; n += 128u) {
            const size_t xb = n / 32u;
            size_t l;
            for (l = 0; l < 32u; ++l) {
                const size_t isv = l / 16u;
                const int q1 = (int) ((ql[n / 2u + l] & 15u) | (((qh[n / 4u + l] >> 0) & 3u) << 4)) - 32;
                const int q2 = (int) ((ql[n / 2u + l + 32u] & 15u) | (((qh[n / 4u + l] >> 2) & 3u) << 4)) - 32;
                const int q3 = (int) ((ql[n / 2u + l] >> 4) | (((qh[n / 4u + l] >> 4) & 3u) << 4)) - 32;
                const int q4 = (int) ((ql[n / 2u + l + 32u] >> 4) | (((qh[n / 4u + l] >> 6) & 3u) << 4)) - 32;
                sum += d * (float) sc[n / 16u + isv + 0u] * xqb[xb + 0u].d * (float) (q1 * (int) xqb[xb + 0u].qs[l]);
                sum += d * (float) sc[n / 16u + isv + 2u] * xqb[xb + 1u].d * (float) (q2 * (int) xqb[xb + 1u].qs[l]);
                sum += d * (float) sc[n / 16u + isv + 4u] * xqb[xb + 2u].d * (float) (q3 * (int) xqb[xb + 2u].qs[l]);
                sum += d * (float) sc[n / 16u + isv + 6u] * xqb[xb + 3u].d * (float) (q4 * (int) xqb[xb + 3u].qs[l]);
            }
        }
    }
    return sum;
}

#if defined(__x86_64__) || defined(__i386__)
__attribute__((target("avx2")))
static inline float minicpmo_hsum256_ps(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 s = _mm_add_ps(lo, hi);
    s = _mm_hadd_ps(s, s);
    s = _mm_hadd_ps(s, s);
    return _mm_cvtss_f32(s);
}

__attribute__((target("avx2")))
static inline __m256i minicpmo_dot_u8_i8_32_lanes_avx2(__m256i q, const int8_t *x) {
    const __m256i xv = _mm256_loadu_si256((const __m256i *) (const void *) x);
    const __m256i ones = _mm256_set1_epi16(1);
    return _mm256_madd_epi16(_mm256_maddubs_epi16(q, xv), ones);
}

__attribute__((target("avx2")))
static inline __m256i minicpmo_dot_i8_i8_16_lanes_avx2(__m128i q, const int8_t *x) {
    const __m128i xv = _mm_loadu_si128((const __m128i *) (const void *) x);
    return _mm256_madd_epi16(_mm256_cvtepi8_epi16(q), _mm256_cvtepi8_epi16(xv));
}

__attribute__((target("avx2,fma")))
static float minicpmo_q8_0_dot_avx2(const uint8_t *row, const float *x, size_t in_dim) {
    const size_t blocks = in_dim / 32u;
    __m256 acc = _mm256_setzero_ps();
    size_t block;
    for (block = 0; block < blocks; ++block) {
        const uint8_t *b = row + block * 34u;
        const __m256 d = _mm256_set1_ps(minicpmo_f16_to_f32(minicpmo_u16_le(b)));
        const int8_t *qs = (const int8_t *) (const void *) (b + 2u);
        const float *xb = x + block * 32u;
        size_t i;
        for (i = 0; i < 32u; i += 8u) {
            const __m128i q8 = _mm_loadl_epi64((const __m128i *) (const void *) (qs + i));
            const __m256 qf = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q8));
            const __m256 xv = _mm256_loadu_ps(xb + i);
            acc = _mm256_fmadd_ps(xv, _mm256_mul_ps(d, qf), acc);
        }
    }
    return minicpmo_hsum256_ps(acc);
}

__attribute__((target("avx2,fma")))
static float minicpmo_q4_k_dot_avx2(const uint8_t *row, const float *x, size_t in_dim) {
    const size_t blocks = in_dim / 256u;
    const __m128i mask4 = _mm_set1_epi8(15);
    __m256 acc = _mm256_setzero_ps();
    size_t block;
    for (block = 0; block < blocks; ++block) {
        const uint8_t *b = row + block * 144u;
        const float d = minicpmo_f16_to_f32(minicpmo_u16_le(b));
        const float dmin = minicpmo_f16_to_f32(minicpmo_u16_le(b + 2u));
        const uint8_t *scales = b + 4u;
        const uint8_t *qs = b + 16u;
        const float *xb = x + block * 256u;
        int is = 0;
        size_t group;
        for (group = 0; group < 4u; ++group) {
            uint8_t sc, mn;
            const size_t qbase = group * 32u;
            size_t l;
            minicpmo_q4_k_scale_min(is, scales, &sc, &mn);
            const __m256 d1 = _mm256_set1_ps(d * (float) sc);
            const __m256 m1 = _mm256_set1_ps(dmin * (float) mn);
            minicpmo_q4_k_scale_min(is + 1, scales, &sc, &mn);
            const __m256 d2 = _mm256_set1_ps(d * (float) sc);
            const __m256 m2 = _mm256_set1_ps(dmin * (float) mn);
            for (l = 0; l < 32u; l += 8u) {
                const __m128i packed = _mm_loadl_epi64((const __m128i *) (const void *) (qs + qbase + l));
                const __m256 qf = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_and_si128(packed, mask4)));
                const __m256 xv = _mm256_loadu_ps(xb + group * 64u + l);
                acc = _mm256_fmadd_ps(xv, _mm256_sub_ps(_mm256_mul_ps(d1, qf), m1), acc);
            }
            for (l = 0; l < 32u; l += 8u) {
                const __m128i packed = _mm_loadl_epi64((const __m128i *) (const void *) (qs + qbase + l));
                const __m128i hi = _mm_and_si128(_mm_srli_epi16(packed, 4), mask4);
                const __m256 qf = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(hi));
                const __m256 xv = _mm256_loadu_ps(xb + group * 64u + 32u + l);
                acc = _mm256_fmadd_ps(xv, _mm256_sub_ps(_mm256_mul_ps(d2, qf), m2), acc);
            }
            is += 2;
        }
    }
    return minicpmo_hsum256_ps(acc);
}

__attribute__((target("avx2,fma")))
static float minicpmo_q6_k_dot_avx2(const uint8_t *row, const float *x, size_t in_dim) {
    const size_t blocks = in_dim / 256u;
    const __m128i mask3 = _mm_set1_epi8(3);
    const __m128i mask4 = _mm_set1_epi8(15);
    const __m256i minus32 = _mm256_set1_epi32(32);
    __m256 acc = _mm256_setzero_ps();
    size_t block;
    for (block = 0; block < blocks; ++block) {
        const uint8_t *b = row + block * 210u;
        const uint8_t *ql = b;
        const uint8_t *qh = b + 128u;
        const int8_t *sc = (const int8_t *) (const void *) (b + 192u);
        const float d = minicpmo_f16_to_f32(minicpmo_u16_le(b + 208u));
        const float *xb = x + block * 256u;
        size_t n;
        for (n = 0; n < 256u; n += 128u) {
            size_t half;
            for (half = 0; half < 2u; ++half) {
                const size_t l0 = half * 16u;
                size_t l;
                const __m256 d1 = _mm256_set1_ps(d * (float) sc[n / 16u + half + 0u]);
                const __m256 d2 = _mm256_set1_ps(d * (float) sc[n / 16u + half + 2u]);
                const __m256 d3 = _mm256_set1_ps(d * (float) sc[n / 16u + half + 4u]);
                const __m256 d4 = _mm256_set1_ps(d * (float) sc[n / 16u + half + 6u]);
                for (l = 0; l < 16u; l += 8u) {
                    const size_t li = l0 + l;
                    const __m128i ql0 = _mm_loadl_epi64((const __m128i *) (const void *) (ql + n / 2u + li));
                    const __m128i ql1 = _mm_loadl_epi64((const __m128i *) (const void *) (ql + n / 2u + 32u + li));
                    const __m128i qhb = _mm_loadl_epi64((const __m128i *) (const void *) (qh + n / 4u + li));
                    const __m256i q1 = _mm256_sub_epi32(_mm256_add_epi32(_mm256_cvtepu8_epi32(_mm_and_si128(ql0, mask4)), _mm256_slli_epi32(_mm256_cvtepu8_epi32(_mm_and_si128(qhb, mask3)), 4)), minus32);
                    const __m256i q2 = _mm256_sub_epi32(_mm256_add_epi32(_mm256_cvtepu8_epi32(_mm_and_si128(ql1, mask4)), _mm256_slli_epi32(_mm256_cvtepu8_epi32(_mm_and_si128(_mm_srli_epi16(qhb, 2), mask3)), 4)), minus32);
                    const __m256i q3 = _mm256_sub_epi32(_mm256_add_epi32(_mm256_cvtepu8_epi32(_mm_and_si128(_mm_srli_epi16(ql0, 4), mask4)), _mm256_slli_epi32(_mm256_cvtepu8_epi32(_mm_and_si128(_mm_srli_epi16(qhb, 4), mask3)), 4)), minus32);
                    const __m256i q4 = _mm256_sub_epi32(_mm256_add_epi32(_mm256_cvtepu8_epi32(_mm_and_si128(_mm_srli_epi16(ql1, 4), mask4)), _mm256_slli_epi32(_mm256_cvtepu8_epi32(_mm_and_si128(_mm_srli_epi16(qhb, 6), mask3)), 4)), minus32);
                    acc = _mm256_fmadd_ps(_mm256_loadu_ps(xb + n + li), _mm256_mul_ps(d1, _mm256_cvtepi32_ps(q1)), acc);
                    acc = _mm256_fmadd_ps(_mm256_loadu_ps(xb + n + 32u + li), _mm256_mul_ps(d2, _mm256_cvtepi32_ps(q2)), acc);
                    acc = _mm256_fmadd_ps(_mm256_loadu_ps(xb + n + 64u + li), _mm256_mul_ps(d3, _mm256_cvtepi32_ps(q3)), acc);
                    acc = _mm256_fmadd_ps(_mm256_loadu_ps(xb + n + 96u + li), _mm256_mul_ps(d4, _mm256_cvtepi32_ps(q4)), acc);
                }
            }
        }
    }
    return minicpmo_hsum256_ps(acc);
}

__attribute__((target("avx2")))
static float minicpmo_q4_k_dot_q8_1_avx2(const uint8_t *row, const minicpmo_block_q8_1 *xq, size_t in_dim) {
    const size_t blocks = in_dim / 256u;
    const __m256i mask4 = _mm256_set1_epi8(15);
    __m256 acc = _mm256_setzero_ps();
    float min_sum = 0.0f;
    size_t block;
    for (block = 0; block < blocks; ++block) {
        const uint8_t *b = row + block * 144u;
        const float d = minicpmo_f16_to_f32(minicpmo_u16_le(b));
        const float dmin = minicpmo_f16_to_f32(minicpmo_u16_le(b + 2u));
        const uint8_t *scales = b + 4u;
        const uint8_t *qs = b + 16u;
        const minicpmo_block_q8_1 *xqb = xq + block * 8u;
        int is = 0;
        size_t group;
        for (group = 0; group < 4u; ++group) {
            uint8_t sc, mn;
            const size_t qbase = group * 32u;
            const __m256i packed = _mm256_loadu_si256((const __m256i *) (const void *) (qs + qbase));
            const __m256i lo = _mm256_and_si256(packed, mask4);
            const __m256i hi = _mm256_and_si256(_mm256_srli_epi16(packed, 4), mask4);
            const minicpmo_block_q8_1 *x0 = xqb + group * 2u;
            const minicpmo_block_q8_1 *x1 = x0 + 1u;
            minicpmo_q4_k_scale_min(is, scales, &sc, &mn);
            acc = _mm256_add_ps(acc, _mm256_mul_ps(_mm256_set1_ps(d * (float) sc * x0->d), _mm256_cvtepi32_ps(minicpmo_dot_u8_i8_32_lanes_avx2(lo, x0->qs))));
            min_sum -= dmin * (float) mn * x0->s;
            minicpmo_q4_k_scale_min(is + 1, scales, &sc, &mn);
            acc = _mm256_add_ps(acc, _mm256_mul_ps(_mm256_set1_ps(d * (float) sc * x1->d), _mm256_cvtepi32_ps(minicpmo_dot_u8_i8_32_lanes_avx2(hi, x1->qs))));
            min_sum -= dmin * (float) mn * x1->s;
            is += 2;
        }
    }
    return min_sum + minicpmo_hsum256_ps(acc);
}

__attribute__((target("avx2")))
static float minicpmo_q6_k_dot_q8_1_avx2(const uint8_t *row, const minicpmo_block_q8_1 *xq, size_t in_dim) {
    const size_t blocks = in_dim / 256u;
    const __m256i mask3 = _mm256_set1_epi8(3);
    const __m256i mask4 = _mm256_set1_epi8(15);
    const __m256i minus32 = _mm256_set1_epi8(32);
    __m256 acc = _mm256_setzero_ps();
    size_t block;
    for (block = 0; block < blocks; ++block) {
        const uint8_t *b = row + block * 210u;
        const uint8_t *ql = b;
        const uint8_t *qh = b + 128u;
        const int8_t *sc = (const int8_t *) (const void *) (b + 192u);
        const float d = minicpmo_f16_to_f32(minicpmo_u16_le(b + 208u));
        const minicpmo_block_q8_1 *xqb = xq + block * 8u;
        size_t n;
        for (n = 0; n < 256u; n += 128u) {
            const size_t xb = n / 32u;
            const size_t sb = n / 16u;
            const __m256i ql0 = _mm256_loadu_si256((const __m256i *) (const void *) (ql + n / 2u));
            const __m256i ql1 = _mm256_loadu_si256((const __m256i *) (const void *) (ql + n / 2u + 32u));
            const __m256i qhb = _mm256_loadu_si256((const __m256i *) (const void *) (qh + n / 4u));
            const __m256i u1 = _mm256_or_si256(_mm256_and_si256(ql0, mask4), _mm256_slli_epi16(_mm256_and_si256(qhb, mask3), 4));
            const __m256i u2 = _mm256_or_si256(_mm256_and_si256(ql1, mask4), _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(qhb, 2), mask3), 4));
            const __m256i u3 = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(ql0, 4), mask4), _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(qhb, 4), mask3), 4));
            const __m256i u4 = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(ql1, 4), mask4), _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(qhb, 6), mask3), 4));
            const __m256i q1 = _mm256_sub_epi8(u1, minus32);
            const __m256i q2 = _mm256_sub_epi8(u2, minus32);
            const __m256i q3 = _mm256_sub_epi8(u3, minus32);
            const __m256i q4 = _mm256_sub_epi8(u4, minus32);
            const __m128i q1_lo = _mm256_castsi256_si128(q1);
            const __m128i q1_hi = _mm256_extracti128_si256(q1, 1);
            const __m128i q2_lo = _mm256_castsi256_si128(q2);
            const __m128i q2_hi = _mm256_extracti128_si256(q2, 1);
            const __m128i q3_lo = _mm256_castsi256_si128(q3);
            const __m128i q3_hi = _mm256_extracti128_si256(q3, 1);
            const __m128i q4_lo = _mm256_castsi256_si128(q4);
            const __m128i q4_hi = _mm256_extracti128_si256(q4, 1);
            const minicpmo_block_q8_1 *x0 = xqb + xb + 0u;
            const minicpmo_block_q8_1 *x1 = xqb + xb + 1u;
            const minicpmo_block_q8_1 *x2 = xqb + xb + 2u;
            const minicpmo_block_q8_1 *x3 = xqb + xb + 3u;
            acc = _mm256_add_ps(acc, _mm256_mul_ps(_mm256_set1_ps(d * x0->d * (float) sc[sb + 0u]), _mm256_cvtepi32_ps(minicpmo_dot_i8_i8_16_lanes_avx2(q1_lo, x0->qs))));
            acc = _mm256_add_ps(acc, _mm256_mul_ps(_mm256_set1_ps(d * x0->d * (float) sc[sb + 1u]), _mm256_cvtepi32_ps(minicpmo_dot_i8_i8_16_lanes_avx2(q1_hi, x0->qs + 16u))));
            acc = _mm256_add_ps(acc, _mm256_mul_ps(_mm256_set1_ps(d * x1->d * (float) sc[sb + 2u]), _mm256_cvtepi32_ps(minicpmo_dot_i8_i8_16_lanes_avx2(q2_lo, x1->qs))));
            acc = _mm256_add_ps(acc, _mm256_mul_ps(_mm256_set1_ps(d * x1->d * (float) sc[sb + 3u]), _mm256_cvtepi32_ps(minicpmo_dot_i8_i8_16_lanes_avx2(q2_hi, x1->qs + 16u))));
            acc = _mm256_add_ps(acc, _mm256_mul_ps(_mm256_set1_ps(d * x2->d * (float) sc[sb + 4u]), _mm256_cvtepi32_ps(minicpmo_dot_i8_i8_16_lanes_avx2(q3_lo, x2->qs))));
            acc = _mm256_add_ps(acc, _mm256_mul_ps(_mm256_set1_ps(d * x2->d * (float) sc[sb + 5u]), _mm256_cvtepi32_ps(minicpmo_dot_i8_i8_16_lanes_avx2(q3_hi, x2->qs + 16u))));
            acc = _mm256_add_ps(acc, _mm256_mul_ps(_mm256_set1_ps(d * x3->d * (float) sc[sb + 6u]), _mm256_cvtepi32_ps(minicpmo_dot_i8_i8_16_lanes_avx2(q4_lo, x3->qs))));
            acc = _mm256_add_ps(acc, _mm256_mul_ps(_mm256_set1_ps(d * x3->d * (float) sc[sb + 7u]), _mm256_cvtepi32_ps(minicpmo_dot_i8_i8_16_lanes_avx2(q4_hi, x3->qs + 16u))));
        }
    }
    return minicpmo_hsum256_ps(acc);
}
#endif

size_t minicpmo_cpu_q8_1_cache_bytes(size_t in_dim) {
    if (in_dim == 0 || in_dim % 32u != 0) {
        return 0;
    }
    return (in_dim / 32u) * sizeof(minicpmo_block_q8_1);
}

int minicpmo_cpu_quantize_q8_1(void *cache, const float *x, size_t in_dim) {
    if (cache == 0 || x == 0 || in_dim == 0 || in_dim % 32u != 0) {
        return 0;
    }
    return minicpmo_quantize_q8_1_scalar((minicpmo_block_q8_1 *) cache, x, in_dim);
}

int minicpmo_cpu_matvec_q8_1_range(float *dst, const uint8_t *data, int dtype, const void *xq_cache, size_t in_dim, size_t out_start, size_t out_end) {
    size_t row;
#if defined(__x86_64__) || defined(__i386__)
    const int use_avx2 = __builtin_cpu_supports("avx2");
#endif
    const minicpmo_block_q8_1 *xq = (const minicpmo_block_q8_1 *) xq_cache;
    if (dst == 0 || data == 0 || xq == 0 || out_end < out_start || in_dim == 0 || in_dim % 256u != 0) {
        return 0;
    }
    if (dtype == MINICPMO_DTYPE_Q4_K) {
        const size_t row_bytes = (in_dim / 256u) * 144u;
        for (row = out_start; row < out_end; ++row) {
#if defined(__x86_64__) || defined(__i386__)
            if (use_avx2) {
                dst[row] = minicpmo_q4_k_dot_q8_1_avx2(data + row * row_bytes, xq, in_dim);
            } else
#endif
            {
                dst[row] = minicpmo_q4_k_dot_q8_1_scalar(data + row * row_bytes, xq, in_dim);
            }
        }
        return 1;
    }
    if (dtype == MINICPMO_DTYPE_Q6_K) {
        const size_t row_bytes = (in_dim / 256u) * 210u;
        for (row = out_start; row < out_end; ++row) {
#if defined(__x86_64__) || defined(__i386__)
            if (use_avx2) {
                dst[row] = minicpmo_q6_k_dot_q8_1_avx2(data + row * row_bytes, xq, in_dim);
            } else
#endif
            {
                dst[row] = minicpmo_q6_k_dot_q8_1_scalar(data + row * row_bytes, xq, in_dim);
            }
        }
        return 1;
    }
    return 0;
}

int minicpmo_cpu_matvec_range(float *dst, const uint8_t *data, int dtype, const float *x, size_t in_dim, size_t out_start, size_t out_end) {
    size_t row;
#if defined(__x86_64__) || defined(__i386__)
    const int use_avx2 = __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
#endif
    if (dst == 0 || data == 0 || x == 0 || out_end < out_start) {
        return 0;
    }
    if (dtype == MINICPMO_DTYPE_Q8_0) {
        const size_t row_bytes = (in_dim / 32u) * 34u;
        if (in_dim == 0 || in_dim % 32u != 0) {
            return 0;
        }
        for (row = out_start; row < out_end; ++row) {
#if defined(__x86_64__) || defined(__i386__)
            if (use_avx2) {
                dst[row] = minicpmo_q8_0_dot_avx2(data + row * row_bytes, x, in_dim);
            } else
#endif
            {
                dst[row] = minicpmo_q8_0_dot_scalar(data + row * row_bytes, x, in_dim);
            }
        }
        return 1;
    }
    if (dtype == MINICPMO_DTYPE_Q4_K) {
        const size_t row_bytes = (in_dim / 256u) * 144u;
        if (in_dim == 0 || in_dim % 256u != 0) {
            return 0;
        }
        for (row = out_start; row < out_end; ++row) {
#if defined(__x86_64__) || defined(__i386__)
            if (use_avx2) {
                dst[row] = minicpmo_q4_k_dot_avx2(data + row * row_bytes, x, in_dim);
            } else
#endif
            {
                dst[row] = minicpmo_q4_k_dot_scalar(data + row * row_bytes, x, in_dim);
            }
        }
        return 1;
    }
    if (dtype == MINICPMO_DTYPE_Q6_K) {
        const size_t row_bytes = (in_dim / 256u) * 210u;
        if (in_dim == 0 || in_dim % 256u != 0) {
            return 0;
        }
        for (row = out_start; row < out_end; ++row) {
#if defined(__x86_64__) || defined(__i386__)
            if (use_avx2) {
                dst[row] = minicpmo_q6_k_dot_avx2(data + row * row_bytes, x, in_dim);
            } else
#endif
            {
                dst[row] = minicpmo_q6_k_dot_scalar(data + row * row_bytes, x, in_dim);
            }
        }
        return 1;
    }
    return 0;
}
