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
    cl_kernel conv2d_forward_winograd_3x3_kernel;
    cl_kernel conv2d_transform_weights_winograd_3x3_kernel;
    cl_kernel conv2d_backward_input_kernel;
    cl_kernel conv2d_backward_input_winograd_3x3_kernel;
    cl_kernel conv2d_backward_weights_kernel;
    cl_kernel conv2d_backward_weights_3x3_kernel;
    cl_kernel conv2d_backward_weights_3x3_ic1_kernel;
    cl_kernel conv2d_backward_weights_winograd_3x3_kernel;
    cl_kernel conv2d_backward_weights_winograd_reduce_kernel;
    cl_kernel conv2d_backward_weights_tiled_kernel;
    cl_kernel conv2d_backward_weights_reduce_kernel;
    cl_kernel dense_forward_kernel;
    cl_kernel dense_backward_weights_kernel;
    cl_kernel dense_backward_input_kernel;
    cl_kernel dense_backward_bias_kernel;
    cl_kernel relu_forward_kernel;
    cl_kernel relu_backward_kernel;
    cl_kernel sigmoid_forward_kernel;
    cl_kernel sigmoid_backward_kernel;
    cl_kernel dropout_forward_kernel;
    cl_kernel maxpool_forward_kernel;
    cl_kernel maxpool_backward_kernel;
    cl_kernel adamw_update_kernel;
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
void opencl_conv2d_forward(GPUBuffer *input, GPUBuffer *weights, GPUBuffer *weights_t, GPUBuffer *bias,
                           GPUBuffer *output, size_t B, size_t in_c, size_t out_c,
                           size_t in_h, size_t in_w, size_t out_h, size_t out_w,
                           size_t k, size_t stride, size_t padding);

void opencl_conv2d_backward_input(GPUBuffer *grad_output, GPUBuffer *weights, GPUBuffer *weights_t,
                                  GPUBuffer *grad_input, size_t B, size_t in_c,
                                  size_t out_c, size_t in_h, size_t in_w,
                                  size_t out_h, size_t out_w,
                                  size_t k, size_t stride, size_t padding);

void opencl_dense_forward(GPUBuffer *input, GPUBuffer *weights, GPUBuffer *bias,
                          GPUBuffer *output, size_t B, size_t in_f, size_t out_f);

void opencl_relu_forward(GPUBuffer *input, GPUBuffer *output, GPUBuffer *mask, size_t n);

void opencl_sigmoid_forward(GPUBuffer *input, GPUBuffer *output, size_t n);
void opencl_sigmoid_backward(GPUBuffer *grad_output, GPUBuffer *sigmoid_output,
                             GPUBuffer *grad_input, size_t n);
void opencl_dropout_forward(GPUBuffer *input, GPUBuffer *output, GPUBuffer *mask,
                            float keep_prob, unsigned int seed, size_t n);
void opencl_dropout_backward(GPUBuffer *grad_output, GPUBuffer *mask,
                             GPUBuffer *grad_input, size_t n);

void opencl_conv2d_backward_weights(GPUBuffer *input_cache, GPUBuffer *grad_output,
                                    GPUBuffer *d_weights, GPUBuffer *d_bias,
                                    size_t B, size_t in_c, size_t out_c,
                                    size_t in_h, size_t in_w, size_t out_h, size_t out_w,
                                    size_t k, size_t stride, size_t padding);

void opencl_conv2d_transform_weights_winograd_3x3(GPUBuffer *weights, GPUBuffer *weights_t,
                                                  size_t out_c, size_t in_c, int reverse);

void opencl_dense_backward_weights(GPUBuffer *input_cache, GPUBuffer *grad_output,
                                   GPUBuffer *d_weights, size_t B, size_t in_f, size_t out_f);

void opencl_dense_backward_bias(GPUBuffer *grad_output, GPUBuffer *d_bias,
                                size_t B, size_t out_f);

void opencl_adamw_update(GPUBuffer *weights, GPUBuffer *gradients, GPUBuffer *m, GPUBuffer *v,
                         float lr, float beta1, float beta2, float eps, float wd, float m_corr, float v_corr, size_t n);

void opencl_zero_buffer(GPUBuffer *buffer, size_t n);

#endif //USE_OPENCL

#endif //OPENCL_BACKEND_H
