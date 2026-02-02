#include "layers.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

#ifdef USE_OPENCL
#include "opencl_backend.h"
#endif

size_t calc_output_size(size_t input_size, size_t kernel_size, 
                        size_t stride, size_t padding) {
    return (input_size + 2 * padding - kernel_size) / stride + 1;
}

//conv2D
typedef struct {
    Conv2DLayer base;
#ifdef USE_OPENCL
    GPUBuffer *weights_gpu;
    GPUBuffer *bias_gpu;
    GPUBuffer *d_weights_gpu;
    GPUBuffer *d_bias_gpu;
    GPUBuffer *input_gpu;
    GPUBuffer *output_gpu;
#endif
} Conv2DLayerGPU;

static Tensor *conv2d_forward(Layer *self, Tensor *input, int training) {
    Conv2DLayerGPU *lg = (Conv2DLayerGPU *)self->impl;
    Conv2DLayer *l = &lg->base;
    (void)training;
    
    size_t B = input->shape[0];
    size_t in_c = l->in_channels;
    size_t out_c = l->out_channels;
    size_t k = l->kernel_size;
    
    size_t in_h = input->shape[2];
    size_t in_w = input->shape[3];
    size_t out_h = calc_output_size(in_h, k, l->stride, l->padding);
    size_t out_w = calc_output_size(in_w, k, l->stride, l->padding);
    
    if (l->input_cache) tensor_destroy(l->input_cache);
#ifdef USE_OPENCL
    tensor_to_cpu((Tensor*)input);  //need CPU data for backward pass cache
#endif
    l->input_cache = tensor_clone(input);
    
    Tensor *output = tensor_create_4d(B, out_c, out_h, out_w);
    
#ifdef USE_OPENCL
    //get input GPU buffer (uploads if needed)
    GPUBuffer *input_gpu = tensor_get_gpu((Tensor*)input);
    
    //ensure output has GPU buffer
    tensor_ensure_gpu(output);
    
    //run kernel
    opencl_conv2d_forward(input_gpu, lg->weights_gpu, lg->bias_gpu,
                          output->gpu, B, in_c, out_c,
                          in_h, in_w, out_h, out_w,
                          k, l->stride, l->padding);
    
    //mark output as GPU-valid, CPU-stale
    output->gpu_valid = 1;
    output->cpu_valid = 0;
#else
    //CPU fallback
    #pragma omp parallel for collapse(2) schedule(dynamic)
    for (size_t b = 0; b < B; b++) {
        for (size_t oc = 0; oc < out_c; oc++) {
            for (size_t oh = 0; oh < out_h; oh++) {
                for (size_t ow = 0; ow < out_w; ow++) {
                    float sum = l->bias->data[oc];
                    
                    for (size_t ic = 0; ic < in_c; ic++) {
                        for (size_t kh = 0; kh < k; kh++) {
                            for (size_t kw = 0; kw < k; kw++) {
                                int ih = (int)(oh * l->stride + kh) - (int)l->padding;
                                int iw = (int)(ow * l->stride + kw) - (int)l->padding;
                                
                                if (ih >= 0 && ih < (int)in_h && iw >= 0 && iw < (int)in_w) {
                                    size_t in_idx = b * (in_c * in_h * in_w) + 
                                                    ic * (in_h * in_w) + 
                                                    (size_t)ih * in_w + (size_t)iw;
                                    float inp = input->data[in_idx];
                                    float wgt = tensor_get_4d(l->weights, oc, ic, kh, kw);
                                    sum += inp * wgt;
                                }
                            }
                        }
                    }
                    
                    size_t out_idx = b * (out_c * out_h * out_w) + 
                                     oc * (out_h * out_w) + 
                                     oh * out_w + ow;
                    output->data[out_idx] = sum;
                }
            }
        }
    }
#endif
    
    return output;
}

static Tensor *conv2d_backward(Layer *self, Tensor *grad_output) {
    Conv2DLayerGPU *lg = (Conv2DLayerGPU *)self->impl;
    Conv2DLayer *l = &lg->base;
    
    size_t B = grad_output->shape[0];
    size_t in_c = l->in_channels;
    size_t out_c = l->out_channels;
    size_t k = l->kernel_size;
    
    size_t in_h = l->input_cache->shape[2];
    size_t in_w = l->input_cache->shape[3];
    size_t out_h = grad_output->shape[2];
    size_t out_w = grad_output->shape[3];
    
    Tensor *grad_input = tensor_create_4d(B, in_c, in_h, in_w);
    tensor_zero(grad_input);

#ifdef USE_OPENCL
    OpenCLContext *ctx = opencl_get_context();
    
    //sync grad_output from GPU before using
    tensor_to_cpu((Tensor*)grad_output);
    
    //upload grad_output for GPU backward input kernel
    GPUBuffer *grad_out_gpu = gpu_buffer_create(grad_output->size * sizeof(float));
    gpu_buffer_write(grad_out_gpu, grad_output->data, grad_output->size * sizeof(float));
    
    //allocate grad_input on GPU
    GPUBuffer *grad_in_gpu = gpu_buffer_create(grad_input->size * sizeof(float));
    
    //zero grad_input
    cl_kernel zero_kernel = ctx->zero_buffer_kernel;
    int n = (int)grad_input->size;
    clSetKernelArg(zero_kernel, 0, sizeof(cl_mem), &grad_in_gpu->buffer);
    clSetKernelArg(zero_kernel, 1, sizeof(int), &n);
    size_t global = ((grad_input->size + 255) / 256) * 256;
    size_t local = 256;
    clEnqueueNDRangeKernel(ctx->queue, zero_kernel, 1, NULL, &global, &local, 0, NULL, NULL);
    
    //run backward input kernel
    cl_kernel kernel = ctx->conv2d_backward_input_kernel;
    int iB = (int)B, i_in_c = (int)in_c, i_out_c = (int)out_c;
    int i_in_h = (int)in_h, i_in_w = (int)in_w;
    int i_out_h = (int)out_h, i_out_w = (int)out_w;
    int ik = (int)k, i_stride = (int)l->stride, i_padding = (int)l->padding;
    
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &grad_out_gpu->buffer);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &lg->weights_gpu->buffer);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &grad_in_gpu->buffer);
    clSetKernelArg(kernel, 3, sizeof(int), &iB);
    clSetKernelArg(kernel, 4, sizeof(int), &i_in_c);
    clSetKernelArg(kernel, 5, sizeof(int), &i_out_c);
    clSetKernelArg(kernel, 6, sizeof(int), &i_in_h);
    clSetKernelArg(kernel, 7, sizeof(int), &i_in_w);
    clSetKernelArg(kernel, 8, sizeof(int), &i_out_h);
    clSetKernelArg(kernel, 9, sizeof(int), &i_out_w);
    clSetKernelArg(kernel, 10, sizeof(int), &ik);
    clSetKernelArg(kernel, 11, sizeof(int), &i_stride);
    clSetKernelArg(kernel, 12, sizeof(int), &i_padding);
    
    global = ((B * in_c * in_h * in_w + 255) / 256) * 256;
    clEnqueueNDRangeKernel(ctx->queue, kernel, 1, NULL, &global, &local, 0, NULL, NULL);
    
    //download grad_input immediately (avoids sync stalls later)
    gpu_buffer_read(grad_in_gpu, grad_input->data, grad_input->size * sizeof(float));
    
    gpu_buffer_destroy(grad_out_gpu);
    gpu_buffer_destroy(grad_in_gpu);
    
    //weight gradients on CPU
    //this is so cursed
    for (size_t oc = 0; oc < out_c; oc++) {

        float bias_grad = 0.0f;
        for (size_t b = 0; b < B; b++) {
            for (size_t oh = 0; oh < out_h; oh++) {
                for (size_t ow = 0; ow < out_w; ow++) {
                    size_t go_idx = b * (out_c * out_h * out_w) + oc * (out_h * out_w) + oh * out_w + ow;
                    float grad = grad_output->data[go_idx];
                    bias_grad += grad;
                    
                    for (size_t ic = 0; ic < in_c; ic++) {
                        for (size_t kh = 0; kh < k; kh++) {
                            for (size_t kw = 0; kw < k; kw++) {
                                int ih = (int)(oh * l->stride + kh) - (int)l->padding;
                                int iw = (int)(ow * l->stride + kw) - (int)l->padding;
                                if (ih >= 0 && ih < (int)in_h && iw >= 0 && iw < (int)in_w) {
                                    size_t in_idx = b * (in_c * in_h * in_w) + ic * (in_h * in_w) + (size_t)ih * in_w + (size_t)iw;
                                    float inp = l->input_cache->data[in_idx];
                                    size_t w_idx = oc * (in_c * k * k) + ic * (k * k) + kh * k + kw;
                                    l->d_weights->data[w_idx] += grad * inp;
                                }
                            }
                        }
                    }
                }
            }
        }
        l->d_bias->data[oc] += bias_grad;
    }

#else
    //CPU fallback
    #pragma omp parallel for collapse(2) schedule(static)
    for (size_t b = 0; b < B; b++) {
        for (size_t ic = 0; ic < in_c; ic++) {
            for (size_t ih = 0; ih < in_h; ih++) {
                for (size_t iw = 0; iw < in_w; iw++) {
                    float sum = 0.0f;
                    for (size_t oc = 0; oc < out_c; oc++) {
                        for (size_t kh = 0; kh < k; kh++) {
                            for (size_t kw = 0; kw < k; kw++) {
                                int oh = (int)(ih + l->padding - kh);
                                int ow_pos = (int)(iw + l->padding - kw);
                                if (oh >= 0 && oh % (int)l->stride == 0 && ow_pos >= 0 && ow_pos % (int)l->stride == 0) {
                                    size_t oh_idx = (size_t)oh / l->stride;
                                    size_t ow_idx = (size_t)ow_pos / l->stride;
                                    if (oh_idx < out_h && ow_idx < out_w) {
                                        size_t go_idx = b * (out_c * out_h * out_w) + oc * (out_h * out_w) + oh_idx * out_w + ow_idx;
                                        float wgt = tensor_get_4d(l->weights, oc, ic, kh, kw);
                                        sum += grad_output->data[go_idx] * wgt;
                                    }
                                }
                            }
                        }
                    }
                    size_t gi_idx = b * (in_c * in_h * in_w) + ic * (in_h * in_w) + ih * in_w + iw;
                    grad_input->data[gi_idx] = sum;
                }
            }
        }
    }
    
    for (size_t oc = 0; oc < out_c; oc++) {
        float bias_grad = 0.0f;
        for (size_t b = 0; b < B; b++) {
            for (size_t oh = 0; oh < out_h; oh++) {
                for (size_t ow = 0; ow < out_w; ow++) {
                    size_t go_idx = b * (out_c * out_h * out_w) + oc * (out_h * out_w) + oh * out_w + ow;
                    float grad = grad_output->data[go_idx];
                    bias_grad += grad;
                    for (size_t ic = 0; ic < in_c; ic++) {
                        for (size_t kh = 0; kh < k; kh++) {
                            for (size_t kw = 0; kw < k; kw++) {
                                int ih = (int)(oh * l->stride + kh) - (int)l->padding;
                                int iw = (int)(ow * l->stride + kw) - (int)l->padding;
                                if (ih >= 0 && ih < (int)in_h && iw >= 0 && iw < (int)in_w) {
                                    size_t in_idx = b * (in_c * in_h * in_w) + ic * (in_h * in_w) + (size_t)ih * in_w + (size_t)iw;
                                    float inp = l->input_cache->data[in_idx];
                                    size_t w_idx = oc * (in_c * k * k) + ic * (k * k) + kh * k + kw;
                                    l->d_weights->data[w_idx] += grad * inp;
                                }
                            }
                        }
                    }
                }
            }
        }
        l->d_bias->data[oc] += bias_grad;
    }
#endif
    
    return grad_input;
}

static void conv2d_zero_grad(Layer *self) {
    Conv2DLayerGPU *lg = (Conv2DLayerGPU *)self->impl;
    tensor_zero(lg->base.d_weights);
    tensor_zero(lg->base.d_bias);
}

static void conv2d_update(Layer *self, float lr) {
    Conv2DLayerGPU *lg = (Conv2DLayerGPU *)self->impl;
    Conv2DLayer *l = &lg->base;
    
    for (size_t i = 0; i < l->weights->size; i++) {
        l->weights->data[i] -= lr * l->d_weights->data[i];
    }
    for (size_t i = 0; i < l->bias->size; i++) {
        l->bias->data[i] -= lr * l->d_bias->data[i];
    }
    
#ifdef USE_OPENCL
    //sync weights to GPU
    gpu_buffer_write(lg->weights_gpu, l->weights->data, l->weights->size * sizeof(float));
    gpu_buffer_write(lg->bias_gpu, l->bias->data, l->bias->size * sizeof(float));
#endif
}

static void conv2d_destroy(Layer *self) {
    Conv2DLayerGPU *lg = (Conv2DLayerGPU *)self->impl;
    Conv2DLayer *l = &lg->base;
    tensor_destroy(l->weights);
    tensor_destroy(l->bias);
    tensor_destroy(l->d_weights);
    tensor_destroy(l->d_bias);
    if (l->input_cache) tensor_destroy(l->input_cache);
#ifdef USE_OPENCL
    if (lg->weights_gpu) gpu_buffer_destroy(lg->weights_gpu);
    if (lg->bias_gpu) gpu_buffer_destroy(lg->bias_gpu);
    if (lg->input_gpu) gpu_buffer_destroy(lg->input_gpu);
    if (lg->output_gpu) gpu_buffer_destroy(lg->output_gpu);
#endif
    free(lg);
    free(self);
}

Layer *layer_conv2d_create(size_t in_channels, size_t out_channels, 
                            size_t kernel_size, size_t stride, size_t padding) {
    Layer *layer = malloc(sizeof(Layer));
    Conv2DLayerGPU *lg = calloc(1, sizeof(Conv2DLayerGPU));
    Conv2DLayer *l = &lg->base;
    
    l->in_channels = in_channels;
    l->out_channels = out_channels;
    l->kernel_size = kernel_size;
    l->stride = stride;
    l->padding = padding;
    
    l->weights = tensor_create_4d(out_channels, in_channels, kernel_size, kernel_size);
    l->bias = tensor_create_1d(out_channels);
    l->d_weights = tensor_create_4d(out_channels, in_channels, kernel_size, kernel_size);
    l->d_bias = tensor_create_1d(out_channels);
    l->input_cache = NULL;
    
    size_t fan_in = in_channels * kernel_size * kernel_size;
    tensor_he_init(l->weights, fan_in);
    tensor_zero(l->bias);
    
#ifdef USE_OPENCL
    lg->weights_gpu = gpu_buffer_create(l->weights->size * sizeof(float));
    lg->bias_gpu = gpu_buffer_create(l->bias->size * sizeof(float));
    gpu_buffer_write(lg->weights_gpu, l->weights->data, l->weights->size * sizeof(float));
    gpu_buffer_write(lg->bias_gpu, l->bias->data, l->bias->size * sizeof(float));
#endif
    
    layer->type = LAYER_CONV2D;
    layer->impl = lg;
    layer->forward = conv2d_forward;
    layer->backward = conv2d_backward;
    layer->update = conv2d_update;
    layer->zero_grad = conv2d_zero_grad;
    layer->destroy = conv2d_destroy;
    
    return layer;
}

//maxpool2D
typedef struct {
    MaxPool2DLayer base;
#ifdef USE_OPENCL
    GPUBuffer *input_gpu;
    GPUBuffer *output_gpu;
    GPUBuffer *indices_gpu;
#endif
} MaxPool2DLayerGPU;

static Tensor *maxpool2d_forward(Layer *self, Tensor *input, int training) {
    MaxPool2DLayerGPU *lg = (MaxPool2DLayerGPU *)self->impl;
    MaxPool2DLayer *l = &lg->base;
    (void)training;
    
    size_t B = input->shape[0];
    size_t c = input->shape[1];
    size_t in_h = input->shape[2];
    size_t in_w = input->shape[3];
    size_t out_h = calc_output_size(in_h, l->pool_size, l->stride, 0);
    size_t out_w = calc_output_size(in_w, l->pool_size, l->stride, 0);
    
    l->batch_size = B;
    l->channels = c;
    l->input_h = in_h;
    l->input_w = in_w;
    
    if (l->max_indices) tensor_destroy(l->max_indices);
    l->max_indices = tensor_create_4d(B, c, out_h, out_w);
    
    Tensor *output = tensor_create_4d(B, c, out_h, out_w);
    
#ifdef USE_OPENCL
    OpenCLContext *ctx = opencl_get_context();
    
    //get input GPU buffer (uploads if needed)
    GPUBuffer *input_gpu = tensor_get_gpu(input);
    
    //ensure output and indices have GPU buffers
    tensor_ensure_gpu(output);
    tensor_ensure_gpu(l->max_indices);
    
    cl_kernel kernel = ctx->maxpool_forward_kernel;
    int iB = (int)B, iC = (int)c;
    int i_in_h = (int)in_h, i_in_w = (int)in_w;
    int i_out_h = (int)out_h, i_out_w = (int)out_w;
    int i_pool = (int)l->pool_size, i_stride = (int)l->stride;
    
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &input_gpu->buffer);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &output->gpu->buffer);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &l->max_indices->gpu->buffer);
    clSetKernelArg(kernel, 3, sizeof(int), &iB);
    clSetKernelArg(kernel, 4, sizeof(int), &iC);
    clSetKernelArg(kernel, 5, sizeof(int), &i_in_h);
    clSetKernelArg(kernel, 6, sizeof(int), &i_in_w);
    clSetKernelArg(kernel, 7, sizeof(int), &i_out_h);
    clSetKernelArg(kernel, 8, sizeof(int), &i_out_w);
    clSetKernelArg(kernel, 9, sizeof(int), &i_pool);
    clSetKernelArg(kernel, 10, sizeof(int), &i_stride);
    
    size_t global = ((B * c * out_h * out_w + 255) / 256) * 256;
    size_t local = 256;
    clEnqueueNDRangeKernel(ctx->queue, kernel, 1, NULL, &global, &local, 0, NULL, NULL);
    
    //mark output and indices as GPU-valid, CPU-stale
    output->gpu_valid = 1;
    output->cpu_valid = 0;
    l->max_indices->gpu_valid = 1;
    l->max_indices->cpu_valid = 0;
#else
    #pragma omp parallel for collapse(3) schedule(static)
    for (size_t b = 0; b < B; b++) {
        for (size_t ch = 0; ch < c; ch++) {
            for (size_t oh = 0; oh < out_h; oh++) {
                for (size_t ow = 0; ow < out_w; ow++) {
                    float max_val = -1e30f;
                    size_t max_idx = 0;
                    for (size_t ph = 0; ph < l->pool_size; ph++) {
                        for (size_t pw = 0; pw < l->pool_size; pw++) {
                            size_t ih = oh * l->stride + ph;
                            size_t iw = ow * l->stride + pw;
                            if (ih < in_h && iw < in_w) {
                                size_t in_idx = b * (c * in_h * in_w) + ch * (in_h * in_w) + ih * in_w + iw;
                                float val = input->data[in_idx];
                                if (val > max_val) {
                                    max_val = val;
                                    max_idx = ph * l->pool_size + pw;
                                }
                            }
                        }
                    }
                    size_t out_idx = b * (c * out_h * out_w) + ch * (out_h * out_w) + oh * out_w + ow;
                    output->data[out_idx] = max_val;
                    l->max_indices->data[out_idx] = (float)max_idx;
                }
            }
        }
    }
#endif
    
    return output;
}

static Tensor *maxpool2d_backward(Layer *self, Tensor *grad_output) {
    MaxPool2DLayerGPU *lg = (MaxPool2DLayerGPU *)self->impl;
    MaxPool2DLayer *l = &lg->base;
    
    size_t B = l->batch_size;
    size_t c = l->channels;
    size_t in_h = l->input_h;
    size_t in_w = l->input_w;
    size_t out_h = grad_output->shape[2];
    size_t out_w = grad_output->shape[3];
    
    Tensor *grad_input = tensor_create_4d(B, c, in_h, in_w);
    tensor_zero(grad_input);
    
#ifdef USE_OPENCL
    OpenCLContext *ctx = opencl_get_context();
    
    //get GPU buffers
    GPUBuffer *grad_out_gpu = tensor_get_gpu((Tensor*)grad_output);
    GPUBuffer *indices_gpu = tensor_get_gpu(l->max_indices);
    tensor_ensure_gpu(grad_input);
    
    //zero grad_input on GPU
    cl_kernel zero_kernel = ctx->zero_buffer_kernel;
    int n = (int)grad_input->size;
    clSetKernelArg(zero_kernel, 0, sizeof(cl_mem), &grad_input->gpu->buffer);
    clSetKernelArg(zero_kernel, 1, sizeof(int), &n);
    size_t global = ((grad_input->size + 255) / 256) * 256;
    size_t local = 256;
    clEnqueueNDRangeKernel(ctx->queue, zero_kernel, 1, NULL, &global, &local, 0, NULL, NULL);
    
    //run maxpool backward kernel
    cl_kernel kernel = ctx->maxpool_backward_kernel;
    int iB = (int)B, iC = (int)c;
    int i_in_h = (int)in_h, i_in_w = (int)in_w;
    int i_out_h = (int)out_h, i_out_w = (int)out_w;
    int i_pool = (int)l->pool_size, i_stride = (int)l->stride;
    
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &grad_out_gpu->buffer);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &indices_gpu->buffer);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &grad_input->gpu->buffer);
    clSetKernelArg(kernel, 3, sizeof(int), &iB);
    clSetKernelArg(kernel, 4, sizeof(int), &iC);
    clSetKernelArg(kernel, 5, sizeof(int), &i_in_h);
    clSetKernelArg(kernel, 6, sizeof(int), &i_in_w);
    clSetKernelArg(kernel, 7, sizeof(int), &i_out_h);
    clSetKernelArg(kernel, 8, sizeof(int), &i_out_w);
    clSetKernelArg(kernel, 9, sizeof(int), &i_pool);
    clSetKernelArg(kernel, 10, sizeof(int), &i_stride);
    
    global = ((B * c * out_h * out_w + 255) / 256) * 256;
    clEnqueueNDRangeKernel(ctx->queue, kernel, 1, NULL, &global, &local, 0, NULL, NULL);
    
    grad_input->gpu_valid = 1;
    grad_input->cpu_valid = 0;
#else
    for (size_t b = 0; b < B; b++) {
        for (size_t ch = 0; ch < c; ch++) {
            for (size_t oh = 0; oh < out_h; oh++) {
                for (size_t ow = 0; ow < out_w; ow++) {
                    size_t go_idx = b * (c * out_h * out_w) + ch * (out_h * out_w) + oh * out_w + ow;
                    float grad = grad_output->data[go_idx];
                    size_t max_idx = (size_t)l->max_indices->data[go_idx];
                    size_t ph = max_idx / l->pool_size;
                    size_t pw = max_idx % l->pool_size;
                    size_t ih = oh * l->stride + ph;
                    size_t iw = ow * l->stride + pw;
                    size_t gi_idx = b * (c * in_h * in_w) + ch * (in_h * in_w) + ih * in_w + iw;
                    grad_input->data[gi_idx] += grad;
                }
            }
        }
    }
#endif
    
    return grad_input;
}


static void maxpool2d_destroy(Layer *self) {
    MaxPool2DLayerGPU *lg = (MaxPool2DLayerGPU *)self->impl;
    if (lg->base.max_indices) tensor_destroy(lg->base.max_indices);
#ifdef USE_OPENCL
    if (lg->input_gpu) gpu_buffer_destroy(lg->input_gpu);
    if (lg->output_gpu) gpu_buffer_destroy(lg->output_gpu);
    if (lg->indices_gpu) gpu_buffer_destroy(lg->indices_gpu);
#endif
    free(lg);
    free(self);
}

Layer *layer_maxpool2d_create(size_t pool_size, size_t stride) {
    Layer *layer = malloc(sizeof(Layer));
    MaxPool2DLayerGPU *lg = calloc(1, sizeof(MaxPool2DLayerGPU));
    
    lg->base.pool_size = pool_size;
    lg->base.stride = stride;
    lg->base.max_indices = NULL;
    
    layer->type = LAYER_MAXPOOL2D;
    layer->impl = lg;
    layer->forward = maxpool2d_forward;
    layer->backward = maxpool2d_backward;
    layer->update = NULL;
    layer->zero_grad = NULL;
    layer->destroy = maxpool2d_destroy;
    
    return layer;
}

//dense
typedef struct {
    DenseLayer base;
#ifdef USE_OPENCL
    GPUBuffer *weights_gpu;
    GPUBuffer *bias_gpu;
    GPUBuffer *input_gpu;
    GPUBuffer *output_gpu;
#endif
} DenseLayerGPU;

static Tensor *dense_forward(Layer *self, Tensor *input, int training) {
    DenseLayerGPU *lg = (DenseLayerGPU *)self->impl;
    DenseLayer *l = &lg->base;
    (void)training;
    
    size_t B = input->shape[0];
    size_t in_f = l->in_features;
    size_t out_f = l->out_features;
    
    if (l->input_cache) tensor_destroy(l->input_cache);
#ifdef USE_OPENCL
    tensor_to_cpu((Tensor*)input);  //need CPU data for backward pass cache
#endif
    l->input_cache = tensor_clone(input);
    
    Tensor *output = tensor_create_2d(B, out_f);
    
#ifdef USE_OPENCL
    //get input GPU buffer (uploads if needed)
    GPUBuffer *input_gpu = tensor_get_gpu(input);
    
    //ensure output has GPU buffer
    tensor_ensure_gpu(output);
    
    opencl_dense_forward(input_gpu, lg->weights_gpu, lg->bias_gpu,
                         output->gpu, B, in_f, out_f);
    
    //mark output as GPU-valid, CPU-stale
    output->gpu_valid = 1;
    output->cpu_valid = 0;
#else
    #pragma omp parallel for collapse(2) schedule(static)
    for (size_t b = 0; b < B; b++) {
        for (size_t j = 0; j < out_f; j++) {
            float sum = l->bias->data[j];
            for (size_t i = 0; i < in_f; i++) {
                sum += input->data[b * in_f + i] * l->weights->data[i * out_f + j];
            }
            output->data[b * out_f + j] = sum;
        }
    }
#endif
    
    return output;
}

static Tensor *dense_backward(Layer *self, Tensor *grad_output) {
    DenseLayerGPU *lg = (DenseLayerGPU *)self->impl;
    DenseLayer *l = &lg->base;
    
    size_t B = grad_output->shape[0];
    size_t in_f = l->in_features;
    size_t out_f = l->out_features;
    
    Tensor *grad_input = tensor_create_2d(B, in_f);
    
#ifdef USE_OPENCL
    OpenCLContext *ctx = opencl_get_context();
    
    //get GPU buffers
    GPUBuffer *grad_out_gpu = tensor_get_gpu((Tensor*)grad_output);
    tensor_ensure_gpu(grad_input);
    
    //input gradient on GPU
    cl_kernel kernel = ctx->dense_backward_input_kernel;
    int iB = (int)B, i_in_f = (int)in_f, i_out_f = (int)out_f;
    
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &grad_out_gpu->buffer);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &lg->weights_gpu->buffer);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &grad_input->gpu->buffer);
    clSetKernelArg(kernel, 3, sizeof(int), &iB);
    clSetKernelArg(kernel, 4, sizeof(int), &i_in_f);
    clSetKernelArg(kernel, 5, sizeof(int), &i_out_f);
    
    size_t global = ((B * in_f + 255) / 256) * 256;
    size_t local = 256;
    clEnqueueNDRangeKernel(ctx->queue, kernel, 1, NULL, &global, &local, 0, NULL, NULL);
    
    grad_input->gpu_valid = 1;
    grad_input->cpu_valid = 0;
    
    //weight and bias gradients still on CPU (as uh accumulation easier there lowkey)
    tensor_to_cpu((Tensor*)grad_output);
    
    #pragma omp parallel for schedule(static)
    for (size_t j = 0; j < out_f; j++) {
        float sum = 0.0f;
        for (size_t b = 0; b < B; b++) {
            sum += grad_output->data[b * out_f + j];
        }
        l->d_bias->data[j] += sum;
    }
    
    #pragma omp parallel for collapse(2) schedule(static)
    for (size_t i = 0; i < in_f; i++) {
        for (size_t j = 0; j < out_f; j++) {
            float sum = 0.0f;
            for (size_t b = 0; b < B; b++) {
                sum += l->input_cache->data[b * in_f + i] * grad_output->data[b * out_f + j];
            }
            l->d_weights->data[i * out_f + j] += sum;
        }
    }
#else
    #pragma omp parallel for schedule(static)
    for (size_t j = 0; j < out_f; j++) {
        float sum = 0.0f;
        for (size_t b = 0; b < B; b++) {
            sum += grad_output->data[b * out_f + j];
        }
        l->d_bias->data[j] += sum;
    }
    
    #pragma omp parallel for collapse(2) schedule(static)
    for (size_t i = 0; i < in_f; i++) {
        for (size_t j = 0; j < out_f; j++) {
            float sum = 0.0f;
            for (size_t b = 0; b < B; b++) {
                sum += l->input_cache->data[b * in_f + i] * grad_output->data[b * out_f + j];
            }
            l->d_weights->data[i * out_f + j] += sum;
        }
    }
    
    #pragma omp parallel for collapse(2) schedule(static)
    for (size_t b = 0; b < B; b++) {
        for (size_t i = 0; i < in_f; i++) {
            float sum = 0.0f;
            for (size_t j = 0; j < out_f; j++) {
                sum += grad_output->data[b * out_f + j] * l->weights->data[i * out_f + j];
            }
            grad_input->data[b * in_f + i] = sum;
        }
    }
#endif
    
    return grad_input;
}


static void dense_zero_grad(Layer *self) {
    DenseLayerGPU *lg = (DenseLayerGPU *)self->impl;
    tensor_zero(lg->base.d_weights);
    tensor_zero(lg->base.d_bias);
}

static void dense_update(Layer *self, float lr) {
    DenseLayerGPU *lg = (DenseLayerGPU *)self->impl;
    DenseLayer *l = &lg->base;
    
    for (size_t i = 0; i < l->weights->size; i++) {
        l->weights->data[i] -= lr * l->d_weights->data[i];
    }
    for (size_t i = 0; i < l->bias->size; i++) {
        l->bias->data[i] -= lr * l->d_bias->data[i];
    }
    
#ifdef USE_OPENCL
    gpu_buffer_write(lg->weights_gpu, l->weights->data, l->weights->size * sizeof(float));
    gpu_buffer_write(lg->bias_gpu, l->bias->data, l->bias->size * sizeof(float));
#endif
}

static void dense_destroy(Layer *self) {
    DenseLayerGPU *lg = (DenseLayerGPU *)self->impl;
    DenseLayer *l = &lg->base;
    tensor_destroy(l->weights);
    tensor_destroy(l->bias);
    tensor_destroy(l->d_weights);
    tensor_destroy(l->d_bias);
    if (l->input_cache) tensor_destroy(l->input_cache);
#ifdef USE_OPENCL
    if (lg->weights_gpu) gpu_buffer_destroy(lg->weights_gpu);
    if (lg->bias_gpu) gpu_buffer_destroy(lg->bias_gpu);
    if (lg->input_gpu) gpu_buffer_destroy(lg->input_gpu);
    if (lg->output_gpu) gpu_buffer_destroy(lg->output_gpu);
#endif
    free(lg);
    free(self);
}

Layer *layer_dense_create(size_t in_features, size_t out_features) {
    Layer *layer = malloc(sizeof(Layer));
    DenseLayerGPU *lg = calloc(1, sizeof(DenseLayerGPU));
    DenseLayer *l = &lg->base;
    
    l->in_features = in_features;
    l->out_features = out_features;
    
    l->weights = tensor_create_2d(in_features, out_features);
    l->bias = tensor_create_1d(out_features);
    l->d_weights = tensor_create_2d(in_features, out_features);
    l->d_bias = tensor_create_1d(out_features);
    l->input_cache = NULL;
    
    tensor_he_init(l->weights, in_features);
    tensor_zero(l->bias);
    
#ifdef USE_OPENCL
    lg->weights_gpu = gpu_buffer_create(l->weights->size * sizeof(float));
    lg->bias_gpu = gpu_buffer_create(l->bias->size * sizeof(float));
    gpu_buffer_write(lg->weights_gpu, l->weights->data, l->weights->size * sizeof(float));
    gpu_buffer_write(lg->bias_gpu, l->bias->data, l->bias->size * sizeof(float));
#endif
    
    layer->type = LAYER_DENSE;
    layer->impl = lg;
    layer->forward = dense_forward;
    layer->backward = dense_backward;
    layer->update = dense_update;
    layer->zero_grad = dense_zero_grad;
    layer->destroy = dense_destroy;
    
    return layer;
}

//flatten (CPU only)
static Tensor *flatten_forward(Layer *self, Tensor *input, int training) {
    FlattenLayer *l = (FlattenLayer *)self->impl;
    (void)training;
    
    l->batch_size = input->shape[0];
    l->original_ndim = input->ndim;
    for (size_t i = 0; i < input->ndim; i++) {
        l->original_shape[i] = input->shape[i];
    }
    
    size_t flat_size = 1;
    for (size_t i = 1; i < input->ndim; i++) {
        flat_size *= input->shape[i];
    }
    
    Tensor *output = tensor_create_2d(l->batch_size, flat_size);
    
#ifdef USE_OPENCL
    //ensure input CPU data is valid before memcpy
    tensor_to_cpu((Tensor*)input);
#endif
    
    memcpy(output->data, input->data, input->size * sizeof(float));
    return output;
}

static Tensor *flatten_backward(Layer *self, Tensor *grad_output) {
    FlattenLayer *l = (FlattenLayer *)self->impl;
    Tensor *grad_input = tensor_create(l->original_ndim, l->original_shape);
#ifdef USE_OPENCL
    tensor_to_cpu((Tensor*)grad_output);
#endif
    memcpy(grad_input->data, grad_output->data, grad_output->size * sizeof(float));
    return grad_input;
}


static void flatten_destroy(Layer *self) {
    free(self->impl);
    free(self);
}

Layer *layer_flatten_create(void) {
    Layer *layer = malloc(sizeof(Layer));
    FlattenLayer *l = calloc(1, sizeof(FlattenLayer));
    layer->type = LAYER_FLATTEN;
    layer->impl = l;
    layer->forward = flatten_forward;
    layer->backward = flatten_backward;
    layer->update = NULL;
    layer->zero_grad = NULL;
    layer->destroy = flatten_destroy;
    return layer;
}

//relu
typedef struct {
    ReLULayer base;
#ifdef USE_OPENCL
    GPUBuffer *input_gpu;
    GPUBuffer *output_gpu;
    GPUBuffer *mask_gpu;
#endif
} ReLULayerGPU;

static Tensor *relu_forward(Layer *self, Tensor *input, int training) {
    ReLULayerGPU *lg = (ReLULayerGPU *)self->impl;
    ReLULayer *l = &lg->base;
    (void)training;
    
    if (l->mask) tensor_destroy(l->mask);
    l->mask = tensor_create(input->ndim, input->shape);
    Tensor *output = tensor_create(input->ndim, input->shape);
    
#ifdef USE_OPENCL
    size_t n = input->size;
    
    //get input GPU buffer (uploads if needed)
    GPUBuffer *input_gpu = tensor_get_gpu(input);
    
    //ensure output and mask have GPU buffers
    tensor_ensure_gpu(output);
    tensor_ensure_gpu(l->mask);
    
    opencl_relu_forward(input_gpu, output->gpu, l->mask->gpu, n);
    
    //mark output as GPU-valid, CPU-stale
    output->gpu_valid = 1;
    output->cpu_valid = 0;
    l->mask->gpu_valid = 1;
    l->mask->cpu_valid = 0;
#else
    #pragma omp parallel for simd schedule(static)
    for (size_t i = 0; i < input->size; i++) {
        if (input->data[i] > 0) {
            output->data[i] = input->data[i];
            l->mask->data[i] = 1.0f;
        } else {
            output->data[i] = 0.0f;
            l->mask->data[i] = 0.0f;
        }
    }
#endif
    
    return output;
}

static Tensor *relu_backward(Layer *self, Tensor *grad_output) {
    ReLULayerGPU *lg = (ReLULayerGPU *)self->impl;
    ReLULayer *l = &lg->base;
    Tensor *grad_input = tensor_create(grad_output->ndim, grad_output->shape);
    
#ifdef USE_OPENCL
    OpenCLContext *ctx = opencl_get_context();
    size_t n = grad_output->size;
    
    //get GPU buffers (uploads if needed)
    GPUBuffer *grad_out_gpu = tensor_get_gpu((Tensor*)grad_output);
    GPUBuffer *mask_gpu = tensor_get_gpu(l->mask);
    tensor_ensure_gpu(grad_input);
    
    //run relu backward kernel
    cl_kernel kernel = ctx->relu_backward_kernel;
    int in = (int)n;
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &grad_out_gpu->buffer);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &mask_gpu->buffer);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &grad_input->gpu->buffer);
    clSetKernelArg(kernel, 3, sizeof(int), &in);
    
    size_t global = ((n + 255) / 256) * 256;
    size_t local = 256;
    clEnqueueNDRangeKernel(ctx->queue, kernel, 1, NULL, &global, &local, 0, NULL, NULL);
    
    grad_input->gpu_valid = 1;
    grad_input->cpu_valid = 0;
#else
    #pragma omp parallel for simd schedule(static)
    for (size_t i = 0; i < grad_output->size; i++) {
        grad_input->data[i] = grad_output->data[i] * l->mask->data[i];
    }
#endif
    
    return grad_input;
}


static void relu_destroy(Layer *self) {
    ReLULayerGPU *lg = (ReLULayerGPU *)self->impl;
    if (lg->base.mask) tensor_destroy(lg->base.mask);
#ifdef USE_OPENCL
    if (lg->input_gpu) gpu_buffer_destroy(lg->input_gpu);
    if (lg->output_gpu) gpu_buffer_destroy(lg->output_gpu);
    if (lg->mask_gpu) gpu_buffer_destroy(lg->mask_gpu);
#endif
    free(lg);
    free(self);
}

Layer *layer_relu_create(void) {
    Layer *layer = malloc(sizeof(Layer));
    ReLULayerGPU *lg = calloc(1, sizeof(ReLULayerGPU));
    layer->type = LAYER_RELU;
    layer->impl = lg;
    layer->forward = relu_forward;
    layer->backward = relu_backward;
    layer->update = NULL;
    layer->zero_grad = NULL;
    layer->destroy = relu_destroy;
    return layer;
}

//sigmoid
typedef struct {
    SigmoidLayer base;
#ifdef USE_OPENCL
    GPUBuffer *input_gpu;
    GPUBuffer *output_gpu;
#endif
} SigmoidLayerGPU;

static Tensor *sigmoid_forward(Layer *self, Tensor *input, int training) {
    SigmoidLayerGPU *lg = (SigmoidLayerGPU *)self->impl;
    SigmoidLayer *l = &lg->base;
    (void)training;
    
    if (l->output_cache) tensor_destroy(l->output_cache);
    Tensor *output = tensor_create(input->ndim, input->shape);
    
#ifdef USE_OPENCL
    size_t n = input->size;
    
    //get input GPU buffer (uploads if needed)
    GPUBuffer *input_gpu = tensor_get_gpu(input);
    
    //ensure output has GPU buffer
    tensor_ensure_gpu(output);
    
    opencl_sigmoid_forward(input_gpu, output->gpu, n);
    
    //mark output as GPU-valid, CPU-stale
    output->gpu_valid = 1;
    output->cpu_valid = 0;
#else
    #pragma omp parallel for simd schedule(static)
    for (size_t i = 0; i < input->size; i++) {
        output->data[i] = 1.0f / (1.0f + expf(-input->data[i]));
    }
#endif
    
#ifdef USE_OPENCL
    //need CPU data for backward pass
    tensor_to_cpu(output);
#endif
    l->output_cache = tensor_clone(output);
    return output;
}

static Tensor *sigmoid_backward(Layer *self, Tensor *grad_output) {
    SigmoidLayerGPU *lg = (SigmoidLayerGPU *)self->impl;
    SigmoidLayer *l = &lg->base;
    Tensor *grad_input = tensor_create(grad_output->ndim, grad_output->shape);
    
#ifdef USE_OPENCL
    tensor_to_cpu((Tensor*)grad_output);
#endif
    
    #pragma omp parallel for simd schedule(static)
    for (size_t i = 0; i < grad_output->size; i++) {
        float s = l->output_cache->data[i];
        grad_input->data[i] = grad_output->data[i] * s * (1.0f - s);
    }
    
    return grad_input;
}

static void sigmoid_destroy(Layer *self) {
    SigmoidLayerGPU *lg = (SigmoidLayerGPU *)self->impl;
    if (lg->base.output_cache) tensor_destroy(lg->base.output_cache);
#ifdef USE_OPENCL
    if (lg->input_gpu) gpu_buffer_destroy(lg->input_gpu);
    if (lg->output_gpu) gpu_buffer_destroy(lg->output_gpu);
#endif
    free(lg);
    free(self);
}

Layer *layer_sigmoid_create(void) {
    Layer *layer = malloc(sizeof(Layer));
    SigmoidLayerGPU *lg = calloc(1, sizeof(SigmoidLayerGPU));
    layer->type = LAYER_SIGMOID;
    layer->impl = lg;
    layer->forward = sigmoid_forward;
    layer->backward = sigmoid_backward;
    layer->update = NULL;
    layer->zero_grad = NULL;
    layer->destroy = sigmoid_destroy;
    return layer;
}

//dropout (CPU only)
static Tensor *dropout_forward(Layer *self, Tensor *input, int training) {
    DropoutLayer *l = (DropoutLayer *)self->impl;
    
    if (!training) return tensor_clone(input);
    
    if (l->mask) tensor_destroy(l->mask);
    l->mask = tensor_create(input->ndim, input->shape);
    Tensor *output = tensor_create(input->ndim, input->shape);
    float scale = 1.0f / (1.0f - l->p);
    
#ifdef USE_OPENCL
    //sync input from GPU before using on CPU
    tensor_to_cpu((Tensor*)input);
#endif
    
    for (size_t i = 0; i < input->size; i++) {
        float r = (float)rand() / (float)RAND_MAX;
        if (r > l->p) {
            output->data[i] = input->data[i] * scale;
            l->mask->data[i] = scale;
        } else {
            output->data[i] = 0.0f;
            l->mask->data[i] = 0.0f;
        }
    }
    
    return output;
}

static Tensor *dropout_backward(Layer *self, Tensor *grad_output) {
    DropoutLayer *l = (DropoutLayer *)self->impl;
    Tensor *grad_input = tensor_create(grad_output->ndim, grad_output->shape);
    
#ifdef USE_OPENCL
    //sync grad_output from GPU before using on CPU
    tensor_to_cpu((Tensor*)grad_output);
#endif
    
    for (size_t i = 0; i < grad_output->size; i++) {

        grad_input->data[i] = grad_output->data[i] * l->mask->data[i];
    }
    
    return grad_input;
}

static void dropout_destroy(Layer *self) {
    DropoutLayer *l = (DropoutLayer *)self->impl;
    if (l->mask) tensor_destroy(l->mask);
    free(l);
    free(self);
}

Layer *layer_dropout_create(float p) {
    Layer *layer = malloc(sizeof(Layer));
    DropoutLayer *l = calloc(1, sizeof(DropoutLayer));
    l->p = p;
    layer->type = LAYER_DROPOUT;
    layer->impl = l;
    layer->forward = dropout_forward;
    layer->backward = dropout_backward;
    layer->update = NULL;
    layer->zero_grad = NULL;
    layer->destroy = dropout_destroy;
    return layer;
}
