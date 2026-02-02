#ifndef SIMD_H
#define SIMD_H

#include <immintrin.h>
#include <stddef.h>

//AVX-256 SIMD operations

//horizontal sum of __m256
static inline float hsum_avx(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    lo = _mm_add_ps(lo, hi);
    __m128 shuf = _mm_movehdup_ps(lo);
    __m128 sums = _mm_add_ps(lo, shuf);
    shuf = _mm_movehl_ps(shuf, sums);
    sums = _mm_add_ss(sums, shuf);
    return _mm_cvtss_f32(sums);
}

//dot product of two float arrays (AVX accelerated)
static inline float dot_avx(const float *a, const float *b, size_t n) {
    __m256 sum = _mm256_setzero_ps();
    
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        sum = _mm256_fmadd_ps(va, vb, sum);
    }
    
    float result = hsum_avx(sum);
    
    //handle remainder
    for (; i < n; i++) {
        result += a[i] * b[i];
    }
    
    return result;
}

//a = a + scale * b (AVX accelerated)
static inline void axpy_avx(float *a, const float *b, float scale, size_t n) {
    __m256 vscale = _mm256_set1_ps(scale);
    
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        va = _mm256_fmadd_ps(vb, vscale, va);
        _mm256_storeu_ps(a + i, va);
    }
    
    for (; i < n; i++) {
        a[i] += scale * b[i];
    }
}

//a = a * scale (AVX accelerated)
static inline void scale_avx(float *a, float scale, size_t n) {
    __m256 vscale = _mm256_set1_ps(scale);
    
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        va = _mm256_mul_ps(va, vscale);
        _mm256_storeu_ps(a + i, va);
    }
    
    for (; i < n; i++) {
        a[i] *= scale;
    }
}

//zero memory (AVX accelerated)
static inline void zero_avx(float *a, size_t n) {
    __m256 vzero = _mm256_setzero_ps();
    
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        _mm256_storeu_ps(a + i, vzero);
    }
    
    for (; i < n; i++) {
        a[i] = 0.0f;
    }
}

#endif
