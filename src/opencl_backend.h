#ifndef OPENCL_BACKEND_H
#define OPENCL_BACKEND_H

#ifdef USE_OPENCL

#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>
#include "tensor.h"

//OpenCL context (global singleton)
typedef struct {
    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    
    //compiled kernels
    cl_program program;
    cl_kernel conv2d_forward_kernel;
    cl_kernel conv2d_backward_input_kernel;
    cl_kernel conv2d_backward_weights_kernel;
    cl_kernel dense_forward_kernel;
    cl_kernel dense_backward_weights_kernel;
    cl_kernel dense_backward_input_kernel;
    cl_kernel dense_backward_bias_kernel;
    cl_kernel relu_forward_kernel;
    cl_kernel relu_backward_kernel;
    cl_kernel sigmoid_forward_kernel;
    cl_kernel sigmoid_backward_kernel;
    cl_kernel maxpool_forward_kernel;
    cl_kernel maxpool_backward_kernel;
    cl_kernel sgd_update_kernel;
    cl_kernel zero_buffer_kernel;
    
    int initialized;
} OpenCLContext;

//initialize/cleanup
int opencl_init(void);
void opencl_cleanup(void);
OpenCLContext *opencl_get_context(void);

//gpu buffer management
typedef struct GPUBuffer {
    cl_mem buffer;
    size_t size;
} GPUBuffer;

GPUBuffer *gpu_buffer_create(size_t size_bytes);
void gpu_buffer_destroy(GPUBuffer *buf);
void gpu_buffer_write(GPUBuffer *buf, const float *data, size_t size_bytes);
void gpu_buffer_read(GPUBuffer *buf, float *data, size_t size_bytes);

//GPU tensor (wrapper around Tensor with optional GPU buffer)
typedef struct {
    Tensor *cpu; //CPU data (may be stale)
    GPUBuffer *gpu;  //GPU data
    int gpu_valid; //is GPU data current?
    int cpu_valid; //is CPU data current?
} GPUTensor;

GPUTensor *gpu_tensor_create(size_t ndim, const size_t *shape);
GPUTensor *gpu_tensor_from_cpu(Tensor *t);
void gpu_tensor_destroy(GPUTensor *t);
void gpu_tensor_to_gpu(GPUTensor *t);
void gpu_tensor_to_cpu(GPUTensor *t);

//kernel launches
void opencl_conv2d_forward(GPUBuffer *input, GPUBuffer *weights, GPUBuffer *bias,
                           GPUBuffer *output, size_t B, size_t in_c, size_t out_c,
                           size_t in_h, size_t in_w, size_t out_h, size_t out_w,
                           size_t k, size_t stride, size_t padding);

void opencl_dense_forward(GPUBuffer *input, GPUBuffer *weights, GPUBuffer *bias,
                          GPUBuffer *output, size_t B, size_t in_f, size_t out_f);

void opencl_relu_forward(GPUBuffer *input, GPUBuffer *output, GPUBuffer *mask, size_t n);

void opencl_sigmoid_forward(GPUBuffer *input, GPUBuffer *output, size_t n);

#endif // USE_OPENCL

#endif // OPENCL_BACKEND_H
