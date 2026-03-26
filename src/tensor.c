#include "tensor.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <time.h>

#ifdef USE_OPENCL
#include "opencl_backend.h"
#endif

static int rand_initialized = 0;

static void ensure_rand_init(void) {
    if (!rand_initialized) {
        srand((unsigned int)time(NULL));
        rand_initialized = 1;
    }
}

static float rand_uniform(float min, float max) {
    ensure_rand_init();
    float t = (float)rand() / (float)RAND_MAX;
    return min + t * (max - min);
}

static float rand_normal(float mean, float stddev) {
    ensure_rand_init();
    static int have_spare = 0;
    static float spare;
    
    if (have_spare) {
        have_spare = 0;
        return spare * stddev + mean;
    }
    
    float u, v, s;
    do {
        u = rand_uniform(-1.0f, 1.0f);
        v = rand_uniform(-1.0f, 1.0f);
        s = u * u + v * v;
    } while (s >= 1.0f || s == 0.0f);
    
    s = sqrtf(-2.0f * logf(s) / s);
    spare = v * s;
    have_spare = 1;
    
    return mean + stddev * u * s;
}

Tensor *tensor_create(size_t ndim, const size_t *shape) {
    Tensor *t = malloc(sizeof(Tensor));
    if (!t) return NULL;
    
    t->ndim = ndim;
    t->shape = malloc(ndim * sizeof(size_t));
    t->strides = malloc(ndim * sizeof(size_t));
    if (!t->shape || !t->strides) {
        free(t->shape);
        free(t->strides);
        free(t);
        return NULL;
    }
    
    t->size = 1;
    for (size_t i = 0; i < ndim; i++) {
        t->shape[i] = shape[i];
        t->size *= shape[i];
    }
    
    size_t stride = 1;
    for (int i = (int)ndim - 1; i >= 0; i--) {
        t->strides[i] = stride;
        stride *= shape[i];
    }
    
    t->data = calloc(t->size, sizeof(float));
    if (!t->data) {
        free(t->shape);
        free(t->strides);
        free(t);
        return NULL;
    }
    
    t->owns_data = 1;
    
#ifdef USE_OPENCL
    t->gpu = NULL;
    t->owns_gpu = 1;
    t->gpu_valid = 0;
    t->cpu_valid = 1;  //CPU data is valid after creation
#endif
    
    return t;
}

Tensor *tensor_create_1d(size_t d0) {
    size_t shape[] = {d0};
    return tensor_create(1, shape);
}

Tensor *tensor_create_2d(size_t d0, size_t d1) {
    size_t shape[] = {d0, d1};
    return tensor_create(2, shape);
}

Tensor *tensor_create_3d(size_t d0, size_t d1, size_t d2) {
    size_t shape[] = {d0, d1, d2};
    return tensor_create(3, shape);
}

Tensor *tensor_create_4d(size_t d0, size_t d1, size_t d2, size_t d3) {
    size_t shape[] = {d0, d1, d2, d3};
    return tensor_create(4, shape);
}

Tensor *tensor_clone(const Tensor *src) {
#ifdef USE_OPENCL
    //ensure CPU data is valid before cloning
    tensor_to_cpu((Tensor*)src);
#endif
    Tensor *dst = tensor_create(src->ndim, src->shape);
    if (!dst) return NULL;
    memcpy(dst->data, src->data, src->size * sizeof(float));
    return dst;
}


void tensor_destroy(Tensor *t) {
    if (!t) return;
#ifdef USE_OPENCL
    if (t->gpu && t->owns_gpu) gpu_buffer_destroy(t->gpu);
#endif
    if (t->owns_data) free(t->data);
    free(t->shape);
    free(t->strides);
    free(t);
}

//view and reshape
Tensor *tensor_view(Tensor *src) {
    Tensor *v = malloc(sizeof(Tensor));
    if (!v) return NULL;
    
    v->ndim = src->ndim;
    v->shape = malloc(src->ndim * sizeof(size_t));
    v->strides = malloc(src->ndim * sizeof(size_t));
    if (!v->shape || !v->strides) {
        free(v->shape);
        free(v->strides);
        free(v);
        return NULL;
    }
    
    memcpy(v->shape, src->shape, src->ndim * sizeof(size_t));
    memcpy(v->strides, src->strides, src->ndim * sizeof(size_t));
    v->size = src->size;
    v->data = src->data;
    v->owns_data = 0;
#ifdef USE_OPENCL
    v->gpu = src->gpu;
    v->owns_gpu = 0;
    v->gpu_valid = src->gpu_valid;
    v->cpu_valid = src->cpu_valid;
#endif
    
    return v;
}

Tensor *tensor_reshape(Tensor *src, size_t ndim, const size_t *new_shape) {
    //verify total size matches
    size_t new_size = 1;
    for (size_t i = 0; i < ndim; i++) {
        new_size *= new_shape[i];
    }
    if (new_size != src->size) return NULL;
    
    Tensor *v = malloc(sizeof(Tensor));
    if (!v) return NULL;
    
    v->ndim = ndim;
    v->shape = malloc(ndim * sizeof(size_t));
    v->strides = malloc(ndim * sizeof(size_t));
    if (!v->shape || !v->strides) {
        free(v->shape);
        free(v->strides);
        free(v);
        return NULL;
    }
    
    //new strides
    size_t stride = 1;
    for (int i = (int)ndim - 1; i >= 0; i--) {
        v->shape[i] = new_shape[i];
        v->strides[i] = stride;
        stride *= new_shape[i];
    }
    
    v->size = src->size;
    v->data = src->data;
    v->owns_data = 0;
#ifdef USE_OPENCL
    v->gpu = src->gpu;
    v->owns_gpu = 0;
    v->gpu_valid = src->gpu_valid;
    v->cpu_valid = src->cpu_valid;
#endif
    
    return v;
}

//initialization
void tensor_fill(Tensor *t, float value) {
    for (size_t i = 0; i < t->size; i++) {
        t->data[i] = value;
    }
}

void tensor_zero(Tensor *t) {
    memset(t->data, 0, t->size * sizeof(float));
}

void tensor_rand_uniform(Tensor *t, float min, float max) {
    for (size_t i = 0; i < t->size; i++) {
        t->data[i] = rand_uniform(min, max);
    }
}

void tensor_rand_normal(Tensor *t, float mean, float stddev) {
    for (size_t i = 0; i < t->size; i++) {
        t->data[i] = rand_normal(mean, stddev);
    }
}

void tensor_he_init(Tensor *t, size_t fan_in) {
    float stddev = sqrtf(2.0f / (float)fan_in);
    tensor_rand_normal(t, 0.0f, stddev);
}

void tensor_xavier_init(Tensor *t, size_t fan_in, size_t fan_out) {
    float limit = sqrtf(6.0f / (float)(fan_in + fan_out));
    tensor_rand_uniform(t, -limit, limit);
}

//basic operations
void tensor_add(Tensor *dst, const Tensor *a, const Tensor *b) {
    for (size_t i = 0; i < dst->size; i++) {
        dst->data[i] = a->data[i] + b->data[i];
    }
}

void tensor_sub(Tensor *dst, const Tensor *a, const Tensor *b) {
    for (size_t i = 0; i < dst->size; i++) {
        dst->data[i] = a->data[i] - b->data[i];
    }
}

void tensor_mul(Tensor *dst, const Tensor *a, const Tensor *b) {
    for (size_t i = 0; i < dst->size; i++) {
        dst->data[i] = a->data[i] * b->data[i];
    }
}

void tensor_scale(Tensor *t, float scalar) {
    for (size_t i = 0; i < t->size; i++) {
        t->data[i] *= scalar;
    }
}

void tensor_add_scalar(Tensor *t, float scalar) {
    for (size_t i = 0; i < t->size; i++) {
        t->data[i] += scalar;
    }
}

//reductions
float tensor_sum(const Tensor *t) {
    float sum = 0.0f;
    for (size_t i = 0; i < t->size; i++) {
        sum += t->data[i];
    }
    return sum;
}

float tensor_mean(const Tensor *t) {
    return tensor_sum(t) / (float)t->size;
}

float tensor_max(const Tensor *t) {
    float max = t->data[0];
    for (size_t i = 1; i < t->size; i++) {
        if (t->data[i] > max) max = t->data[i];
    }
    return max;
}

float tensor_min(const Tensor *t) {
    float min = t->data[0];
    for (size_t i = 1; i < t->size; i++) {
        if (t->data[i] < min) min = t->data[i];
    }
    return min;
}

//linear algebra
void tensor_matmul(Tensor *dst, const Tensor *a, const Tensor *b) {
    //a: [M, K], b: [K, N] -> dst: [M, N]
    size_t M = a->shape[0];
    size_t K = a->shape[1];
    size_t N = b->shape[1];
    
    for (size_t i = 0; i < M; i++) {
        for (size_t j = 0; j < N; j++) {
            float sum = 0.0f;
            for (size_t k = 0; k < K; k++) {
                sum += tensor_get_2d(a, i, k) * tensor_get_2d(b, k, j);
            }
            tensor_set_2d(dst, i, j, sum);
        }
    }
}

void tensor_transpose_2d(Tensor *dst, const Tensor *src) {
    for (size_t i = 0; i < src->shape[0]; i++) {
        for (size_t j = 0; j < src->shape[1]; j++) {
            tensor_set_2d(dst, j, i, tensor_get_2d(src, i, j));
        }
    }
}

//copy
void tensor_copy(Tensor *dst, const Tensor *src) {
    memcpy(dst->data, src->data, src->size * sizeof(float));
}

void tensor_copy_data(Tensor *dst, const float *data, size_t count) {
    memcpy(dst->data, data, count * sizeof(float));
}

//debug
void tensor_print_shape(const Tensor *t, const char *name) {
    printf("%s: [", name);
    for (size_t i = 0; i < t->ndim; i++) {
        printf("%zu", t->shape[i]);
        if (i < t->ndim - 1) printf(", ");
    }
    printf("] (%zu elements)\n", t->size);
}

void tensor_print(const Tensor *t, const char *name) {
    tensor_print_shape(t, name);
    
    if (t->ndim == 1) {
        printf("[");
        for (size_t i = 0; i < t->shape[0] && i < 10; i++) {
            printf("%.4f", t->data[i]);
            if (i < t->shape[0] - 1) printf(", ");
        }
        if (t->shape[0] > 10) printf(", ...");
        printf("]\n");
    } else if (t->ndim == 2) {
        for (size_t i = 0; i < t->shape[0] && i < 5; i++) {
            printf("[");
            for (size_t j = 0; j < t->shape[1] && j < 8; j++) {
                printf("%7.4f", tensor_get_2d(t, i, j));
                if (j < t->shape[1] - 1) printf(", ");
            }
            if (t->shape[1] > 8) printf(", ...");
            printf("]\n");
        }
        if (t->shape[0] > 5) printf("...\n");
    } else {
        printf("(tensor too high-dimensional to print)\n");
    }
}

#ifdef USE_OPENCL
//GPU sync functions

void tensor_ensure_gpu(Tensor *t) {
    if (!t->gpu) {
        t->gpu = gpu_buffer_create(t->size * sizeof(float));
        t->owns_gpu = 1;
    }
}

void tensor_to_gpu(Tensor *t) {
    if (t->gpu_valid) return;  //already on GPU
    tensor_ensure_gpu(t);
    if (t->cpu_valid) {
        gpu_buffer_write(t->gpu, t->data, t->size * sizeof(float));
    }
    t->gpu_valid = 1;
}

void tensor_to_cpu(Tensor *t) {
    if (t->cpu_valid) return;  //already on CPU
    if (t->gpu && t->gpu_valid) {
        gpu_buffer_read(t->gpu, t->data, t->size * sizeof(float));
    }
    t->cpu_valid = 1;
}

void tensor_gpu_invalidate_cpu(Tensor *t) {
    t->cpu_valid = 0;
}

void tensor_gpu_invalidate_gpu(Tensor *t) {
    t->gpu_valid = 0;
}

struct GPUBuffer *tensor_get_gpu(Tensor *t) {
    tensor_to_gpu(t);
    return t->gpu;
}
#endif
