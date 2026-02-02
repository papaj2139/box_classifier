#ifndef TENSOR_H
#define TENSOR_H

#include <stddef.h>
#include <stdint.h>

#ifdef USE_OPENCL
struct GPUBuffer;  //forward declaration
#endif

typedef struct {
    float *data;//CPU data storage
    size_t *shape; //dimension sizes
    size_t *strides; //strides for indexing
    size_t ndim; //number of dimensions
    size_t size; //total elements
    int owns_data; //should free data on destroy
    
#ifdef USE_OPENCL
    struct GPUBuffer *gpu; //GPU buffer (NULL if not allocated)
    int gpu_valid; //is GPU copy current?
    int cpu_valid; //is CPU copy current?
#endif
} Tensor;

//lifecycle
Tensor *tensor_create(size_t ndim, const size_t *shape);
Tensor *tensor_create_1d(size_t d0);
Tensor *tensor_create_2d(size_t d0, size_t d1);
Tensor *tensor_create_3d(size_t d0, size_t d1, size_t d2);
Tensor *tensor_create_4d(size_t d0, size_t d1, size_t d2, size_t d3);
Tensor *tensor_clone(const Tensor *src);
void tensor_destroy(Tensor *t);

//view (shares data, doesn't own)
Tensor *tensor_view(Tensor *src);
Tensor *tensor_reshape(Tensor *src, size_t ndim, const size_t *new_shape);

//initialization
void tensor_fill(Tensor *t, float value);
void tensor_zero(Tensor *t);
void tensor_rand_uniform(Tensor *t, float min, float max);
void tensor_rand_normal(Tensor *t, float mean, float stddev);
void tensor_he_init(Tensor *t, size_t fan_in); //he initialization for ReLU
void tensor_xavier_init(Tensor *t, size_t fan_in, size_t fan_out);

//element access
static inline float tensor_get_1d(const Tensor *t, size_t i) {
    return t->data[i];
}
static inline float tensor_get_2d(const Tensor *t, size_t i, size_t j) {
    return t->data[i * t->strides[0] + j * t->strides[1]];
}
static inline float tensor_get_3d(const Tensor *t, size_t i, size_t j, size_t k) {
    return t->data[i * t->strides[0] + j * t->strides[1] + k * t->strides[2]];
}
static inline float tensor_get_4d(const Tensor *t, size_t i, size_t j, size_t k, size_t l) {
    return t->data[i * t->strides[0] + j * t->strides[1] + k * t->strides[2] + l * t->strides[3]];
}

static inline void tensor_set_1d(Tensor *t, size_t i, float val) {
    t->data[i] = val;
}
static inline void tensor_set_2d(Tensor *t, size_t i, size_t j, float val) {
    t->data[i * t->strides[0] + j * t->strides[1]] = val;
}
static inline void tensor_set_3d(Tensor *t, size_t i, size_t j, size_t k, float val) {
    t->data[i * t->strides[0] + j * t->strides[1] + k * t->strides[2]] = val;
}
static inline void tensor_set_4d(Tensor *t, size_t i, size_t j, size_t k, size_t l, float val) {
    t->data[i * t->strides[0] + j * t->strides[1] + k * t->strides[2] + l * t->strides[3]] = val;
}

//basic operations (in-place where possible)
void tensor_add(Tensor *dst, const Tensor *a, const Tensor *b);
void tensor_sub(Tensor *dst, const Tensor *a, const Tensor *b);
void tensor_mul(Tensor *dst, const Tensor *a, const Tensor *b); //element-wise
void tensor_scale(Tensor *t, float scalar);
void tensor_add_scalar(Tensor *t, float scalar);

//reductions
float tensor_sum(const Tensor *t);
float tensor_mean(const Tensor *t);
float tensor_max(const Tensor *t);
float tensor_min(const Tensor *t);

//linear algebra
void tensor_matmul(Tensor *dst, const Tensor *a, const Tensor *b); //2D only
void tensor_transpose_2d(Tensor *dst, const Tensor *src);

//copy
void tensor_copy(Tensor *dst, const Tensor *src);
void tensor_copy_data(Tensor *dst, const float *data, size_t count);

//debug
void tensor_print(const Tensor *t, const char *name);
void tensor_print_shape(const Tensor *t, const char *name);

#ifdef USE_OPENCL
//GPU sync functions
void tensor_to_gpu(Tensor *t); //upload CPU->GPU if needed
void tensor_to_cpu(Tensor *t); //download GPU->CPU if needed
void tensor_ensure_gpu(Tensor *t); //allocate GPU buffer if needed
void tensor_gpu_invalidate_cpu(Tensor *t); //mark CPU copy stale
void tensor_gpu_invalidate_gpu(Tensor *t); //mark GPU copy stale
struct GPUBuffer *tensor_get_gpu(Tensor *t); //get GPU buffer (uploads if needed)
#endif

#endif
