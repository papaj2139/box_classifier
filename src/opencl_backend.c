#ifdef USE_OPENCL

#include "opencl_backend.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

//global context
static OpenCLContext g_ctx = {0};

//initialization
static char *load_kernel_source(const char *filename) {
    FILE *f = fopen(filename, "r");
    if (!f) {
        fprintf(stderr, "Failed to open kernel file: %s\n", filename);
        return NULL;
    }
    
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);
    
    char *source = malloc(size + 1);
    fread(source, 1, size, f);
    source[size] = '\0';
    fclose(f);
    
    return source;
}

int opencl_init(void) {
    if (g_ctx.initialized) return 0;
    
    cl_int err;
    
    //get platform
    err = clGetPlatformIDs(1, &g_ctx.platform, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Failed to get OpenCL platform\n");
        return -1;
    }
    
    //get device (prefer GPU, fallback to CPU)
    err = clGetDeviceIDs(g_ctx.platform, CL_DEVICE_TYPE_GPU, 1, &g_ctx.device, NULL);
    if (err != CL_SUCCESS) {
        printf("No GPU found, trying CPU...\n");
        err = clGetDeviceIDs(g_ctx.platform, CL_DEVICE_TYPE_CPU, 1, &g_ctx.device, NULL);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Failed to get OpenCL device\n");
            return -1;
        }
    }
    
    //print device info
    char device_name[256];
    clGetDeviceInfo(g_ctx.device, CL_DEVICE_NAME, sizeof(device_name), device_name, NULL);
    printf("OpenCL device: %s\n", device_name);
    
    //create context
    g_ctx.context = clCreateContext(NULL, 1, &g_ctx.device, NULL, NULL, &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Failed to create OpenCL context\n");
        return -1;
    }
    
    //create command queue
    g_ctx.queue = clCreateCommandQueue(g_ctx.context, g_ctx.device, 0, &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Failed to create command queue\n");
        return -1;
    }
    
    //load and compile kernels
    char *source = load_kernel_source("src/kernels.cl");
    if (!source) return -1;
    
    const char *sources[] = {source};
    size_t lengths[] = {strlen(source)};
    
    g_ctx.program = clCreateProgramWithSource(g_ctx.context, 1, sources, lengths, &err);
    free(source);
    
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Failed to create program\n");
        return -1;
    }
    
    err = clBuildProgram(g_ctx.program, 1, &g_ctx.device, "-cl-mad-enable", NULL, NULL);
    if (err != CL_SUCCESS) {
        //print build log
        size_t log_size;
        clGetProgramBuildInfo(g_ctx.program, g_ctx.device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        char *log = malloc(log_size);
        clGetProgramBuildInfo(g_ctx.program, g_ctx.device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
        fprintf(stderr, "Build error:\n%s\n", log);
        free(log);
        return -1;
    }
    
    //create kernels
    g_ctx.conv2d_forward_kernel = clCreateKernel(g_ctx.program, "conv2d_forward", &err);
    g_ctx.conv2d_forward_winograd_3x3_kernel = clCreateKernel(g_ctx.program, "conv2d_forward_winograd_3x3", &err);
    g_ctx.conv2d_transform_weights_winograd_3x3_kernel = clCreateKernel(g_ctx.program, "conv2d_transform_weights_winograd_3x3", &err);
    g_ctx.conv2d_backward_input_kernel = clCreateKernel(g_ctx.program, "conv2d_backward_input", &err);
    g_ctx.conv2d_backward_input_winograd_3x3_kernel = clCreateKernel(g_ctx.program, "conv2d_backward_input_winograd_3x3", &err);
    g_ctx.conv2d_backward_weights_kernel = clCreateKernel(g_ctx.program, "conv2d_backward_weights", &err);
    g_ctx.conv2d_backward_weights_3x3_kernel = clCreateKernel(g_ctx.program, "conv2d_backward_weights_3x3_s1p1", &err);
    g_ctx.conv2d_backward_weights_3x3_ic1_kernel = clCreateKernel(g_ctx.program, "conv2d_backward_weights_3x3_ic1_s1p1", &err);
    g_ctx.conv2d_backward_weights_winograd_3x3_kernel = clCreateKernel(g_ctx.program, "conv2d_backward_weights_winograd_3x3", &err);
    g_ctx.conv2d_backward_weights_winograd_reduce_kernel = clCreateKernel(g_ctx.program, "conv2d_backward_weights_winograd_reduce", &err);
    g_ctx.conv2d_backward_weights_tiled_kernel = clCreateKernel(g_ctx.program, "conv2d_backward_weights_tiled", &err);
    g_ctx.conv2d_backward_weights_reduce_kernel = clCreateKernel(g_ctx.program, "conv2d_backward_weights_reduce", &err);
    g_ctx.dense_forward_kernel = clCreateKernel(g_ctx.program, "dense_forward", &err);
    g_ctx.dense_backward_weights_kernel = clCreateKernel(g_ctx.program, "dense_backward_weights", &err);
    g_ctx.dense_backward_input_kernel = clCreateKernel(g_ctx.program, "dense_backward_input", &err);
    g_ctx.dense_backward_bias_kernel = clCreateKernel(g_ctx.program, "dense_backward_bias", &err);
    g_ctx.relu_forward_kernel = clCreateKernel(g_ctx.program, "relu_forward", &err);
    g_ctx.relu_backward_kernel = clCreateKernel(g_ctx.program, "relu_backward", &err);
    g_ctx.sigmoid_forward_kernel = clCreateKernel(g_ctx.program, "sigmoid_forward", &err);
    g_ctx.sigmoid_backward_kernel = clCreateKernel(g_ctx.program, "sigmoid_backward", &err);
    g_ctx.dropout_forward_kernel = clCreateKernel(g_ctx.program, "dropout_forward", &err);
    g_ctx.maxpool_forward_kernel = clCreateKernel(g_ctx.program, "maxpool_forward", &err);
    g_ctx.maxpool_backward_kernel = clCreateKernel(g_ctx.program, "maxpool_backward", &err);
    g_ctx.adamw_update_kernel = clCreateKernel(g_ctx.program, "adamw_update", &err);
    g_ctx.zero_buffer_kernel = clCreateKernel(g_ctx.program, "zero_buffer", &err);
    
    g_ctx.initialized = 1;
    printf("OpenCL initialized successfully\n");
    return 0;
}

void opencl_cleanup(void) {
    if (!g_ctx.initialized) return;
    
    clReleaseKernel(g_ctx.conv2d_forward_kernel);
    clReleaseKernel(g_ctx.conv2d_forward_winograd_3x3_kernel);
    clReleaseKernel(g_ctx.conv2d_transform_weights_winograd_3x3_kernel);
    clReleaseKernel(g_ctx.conv2d_backward_input_kernel);
    clReleaseKernel(g_ctx.conv2d_backward_input_winograd_3x3_kernel);
    clReleaseKernel(g_ctx.conv2d_backward_weights_kernel);
    clReleaseKernel(g_ctx.conv2d_backward_weights_3x3_kernel);
    clReleaseKernel(g_ctx.conv2d_backward_weights_3x3_ic1_kernel);
    clReleaseKernel(g_ctx.conv2d_backward_weights_winograd_3x3_kernel);
    clReleaseKernel(g_ctx.conv2d_backward_weights_winograd_reduce_kernel);
    clReleaseKernel(g_ctx.conv2d_backward_weights_tiled_kernel);
    clReleaseKernel(g_ctx.conv2d_backward_weights_reduce_kernel);
    clReleaseKernel(g_ctx.dense_forward_kernel);
    clReleaseKernel(g_ctx.dense_backward_weights_kernel);
    clReleaseKernel(g_ctx.dense_backward_input_kernel);
    clReleaseKernel(g_ctx.dense_backward_bias_kernel);
    clReleaseKernel(g_ctx.relu_forward_kernel);
    clReleaseKernel(g_ctx.relu_backward_kernel);
    clReleaseKernel(g_ctx.sigmoid_forward_kernel);
    clReleaseKernel(g_ctx.sigmoid_backward_kernel);
    clReleaseKernel(g_ctx.dropout_forward_kernel);
    clReleaseKernel(g_ctx.maxpool_forward_kernel);
    clReleaseKernel(g_ctx.maxpool_backward_kernel);
    clReleaseKernel(g_ctx.zero_buffer_kernel);
    
    clReleaseProgram(g_ctx.program);
    clReleaseCommandQueue(g_ctx.queue);
    clReleaseContext(g_ctx.context);
    
    g_ctx.initialized = 0;
}

OpenCLContext *opencl_get_context(void) {
    return &g_ctx;
}

//gpu buffer management
GPUBuffer *gpu_buffer_create(size_t size_bytes) {
    GPUBuffer *buf = malloc(sizeof(GPUBuffer));
    cl_int err;
    
    buf->buffer = clCreateBuffer(g_ctx.context, CL_MEM_READ_WRITE, size_bytes, NULL, &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Failed to create GPU buffer\n");
        free(buf);
        return NULL;
    }
    
    buf->size = size_bytes;
    return buf;
}

void gpu_buffer_destroy(GPUBuffer *buf) {
    if (!buf) return;
    clReleaseMemObject(buf->buffer);
    free(buf);
}

void gpu_buffer_write(GPUBuffer *buf, const float *data, size_t size_bytes) {
    clEnqueueWriteBuffer(g_ctx.queue, buf->buffer, CL_TRUE, 0, size_bytes, data, 0, NULL, NULL);
}

void gpu_buffer_read(GPUBuffer *buf, float *data, size_t size_bytes) {
    clEnqueueReadBuffer(g_ctx.queue, buf->buffer, CL_TRUE, 0, size_bytes, data, 0, NULL, NULL);
}

//gpu tensor
GPUTensor *gpu_tensor_create(size_t ndim, const size_t *shape) {
    GPUTensor *t = malloc(sizeof(GPUTensor));
    t->cpu = tensor_create(ndim, shape);
    t->gpu = gpu_buffer_create(t->cpu->size * sizeof(float));
    t->gpu_valid = 0;
    t->cpu_valid = 1;
    return t;
}

GPUTensor *gpu_tensor_from_cpu(Tensor *cpu) {
    GPUTensor *t = malloc(sizeof(GPUTensor));
    t->cpu = cpu;
    t->gpu = gpu_buffer_create(cpu->size * sizeof(float));
    t->gpu_valid = 0;
    t->cpu_valid = 1;
    return t;
}

void gpu_tensor_destroy(GPUTensor *t) {
    if (!t) return;
    tensor_destroy(t->cpu);
    gpu_buffer_destroy(t->gpu);
    free(t);
}

void gpu_tensor_to_gpu(GPUTensor *t) {
    if (t->gpu_valid) return;
    gpu_buffer_write(t->gpu, t->cpu->data, t->cpu->size * sizeof(float));
    t->gpu_valid = 1;
}

void gpu_tensor_to_cpu(GPUTensor *t) {
    if (t->cpu_valid) return;
    gpu_buffer_read(t->gpu, t->cpu->data, t->cpu->size * sizeof(float));
    t->cpu_valid = 1;
}

//kernel launches
static size_t round_up(size_t value, size_t multiple) {
    return ((value + multiple - 1) / multiple) * multiple;
}

void opencl_conv2d_transform_weights_winograd_3x3(GPUBuffer *weights, GPUBuffer *weights_t,
                                                  size_t out_c, size_t in_c, int reverse) {
    cl_kernel kernel = g_ctx.conv2d_transform_weights_winograd_3x3_kernel;
    int i_out_c = (int)out_c;
    int i_in_c = (int)in_c;
    int i_reverse = reverse ? 1 : 0;

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &weights->buffer);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &weights_t->buffer);
    clSetKernelArg(kernel, 2, sizeof(int), &i_out_c);
    clSetKernelArg(kernel, 3, sizeof(int), &i_in_c);
    clSetKernelArg(kernel, 4, sizeof(int), &i_reverse);

    size_t global = round_up(out_c * in_c, 256);
    size_t local = 256;
    clEnqueueNDRangeKernel(g_ctx.queue, kernel, 1, NULL, &global, &local, 0, NULL, NULL);
}

void opencl_conv2d_forward(GPUBuffer *input, GPUBuffer *weights, GPUBuffer *weights_t, GPUBuffer *bias,
                           GPUBuffer *output, size_t B, size_t in_c, size_t out_c,
                           size_t in_h, size_t in_w, size_t out_h, size_t out_w,
                           size_t k, size_t stride, size_t padding) {
    cl_kernel kernel = g_ctx.conv2d_forward_kernel;
    int use_winograd = (k == 3 && stride == 1 && padding == 1);
    if (use_winograd) kernel = g_ctx.conv2d_forward_winograd_3x3_kernel;
    
    int iB = (int)B, i_in_c = (int)in_c, i_out_c = (int)out_c;
    int i_in_h = (int)in_h, i_in_w = (int)in_w;
    int i_out_h = (int)out_h, i_out_w = (int)out_w;
    int ik = (int)k, i_stride = (int)stride, i_padding = (int)padding;
    
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &input->buffer);
    cl_mem wbuf = (use_winograd && weights_t) ? weights_t->buffer : weights->buffer;
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &wbuf);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &bias->buffer);
    clSetKernelArg(kernel, 3, sizeof(cl_mem), &output->buffer);
    if (use_winograd) {
        clSetKernelArg(kernel, 4, sizeof(int), &iB);
        clSetKernelArg(kernel, 5, sizeof(int), &i_in_c);
        clSetKernelArg(kernel, 6, sizeof(int), &i_out_c);
        clSetKernelArg(kernel, 7, sizeof(int), &i_in_h);
        clSetKernelArg(kernel, 8, sizeof(int), &i_in_w);
        clSetKernelArg(kernel, 9, sizeof(int), &i_out_h);
        clSetKernelArg(kernel, 10, sizeof(int), &i_out_w);
        size_t tile_h = (out_h + 1) / 2;
        size_t tile_w = (out_w + 1) / 2;
        size_t global = round_up(B * out_c * tile_h * tile_w, 256);
        size_t local = 256;
        clEnqueueNDRangeKernel(g_ctx.queue, kernel, 1, NULL, &global, &local, 0, NULL, NULL);
        return;
    }

    clSetKernelArg(kernel, 4, sizeof(int), &iB);
    clSetKernelArg(kernel, 5, sizeof(int), &i_in_c);
    clSetKernelArg(kernel, 6, sizeof(int), &i_out_c);
    clSetKernelArg(kernel, 7, sizeof(int), &i_in_h);
    clSetKernelArg(kernel, 8, sizeof(int), &i_in_w);
    clSetKernelArg(kernel, 9, sizeof(int), &i_out_h);
    clSetKernelArg(kernel, 10, sizeof(int), &i_out_w);
    clSetKernelArg(kernel, 11, sizeof(int), &ik);
    clSetKernelArg(kernel, 12, sizeof(int), &i_stride);
    clSetKernelArg(kernel, 13, sizeof(int), &i_padding);
    
    size_t global = round_up(B * out_c * out_h * out_w, 256);
    size_t local = 256;
    
    clEnqueueNDRangeKernel(g_ctx.queue, kernel, 1, NULL, &global, &local, 0, NULL, NULL);
}

void opencl_conv2d_backward_input(GPUBuffer *grad_output, GPUBuffer *weights, GPUBuffer *weights_t,
                                  GPUBuffer *grad_input, size_t B, size_t in_c,
                                  size_t out_c, size_t in_h, size_t in_w,
                                  size_t out_h, size_t out_w,
                                  size_t k, size_t stride, size_t padding) {
    cl_kernel kernel = g_ctx.conv2d_backward_input_kernel;
    int use_winograd = (k == 3 && stride == 1 && padding == 1);
    if (use_winograd) kernel = g_ctx.conv2d_backward_input_winograd_3x3_kernel;

    int iB = (int)B, i_in_c = (int)in_c, i_out_c = (int)out_c;
    int i_in_h = (int)in_h, i_in_w = (int)in_w;
    int i_out_h = (int)out_h, i_out_w = (int)out_w;
    int ik = (int)k, i_stride = (int)stride, i_padding = (int)padding;

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &grad_output->buffer);
    cl_mem wbuf = (use_winograd && weights_t) ? weights_t->buffer : weights->buffer;
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &wbuf);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &grad_input->buffer);

    if (use_winograd) {
        clSetKernelArg(kernel, 3, sizeof(int), &iB);
        clSetKernelArg(kernel, 4, sizeof(int), &i_in_c);
        clSetKernelArg(kernel, 5, sizeof(int), &i_out_c);
        clSetKernelArg(kernel, 6, sizeof(int), &i_in_h);
        clSetKernelArg(kernel, 7, sizeof(int), &i_in_w);
        clSetKernelArg(kernel, 8, sizeof(int), &i_out_h);
        clSetKernelArg(kernel, 9, sizeof(int), &i_out_w);
        size_t tile_h = (in_h + 1) / 2;
        size_t tile_w = (in_w + 1) / 2;
        size_t global = round_up(B * in_c * tile_h * tile_w, 256);
        size_t local = 256;
        clEnqueueNDRangeKernel(g_ctx.queue, kernel, 1, NULL, &global, &local, 0, NULL, NULL);
        return;
    }

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

    size_t global = round_up(B * in_c * in_h * in_w, 256);
    size_t local = 256;
    clEnqueueNDRangeKernel(g_ctx.queue, kernel, 1, NULL, &global, &local, 0, NULL, NULL);
}

void opencl_dense_forward(GPUBuffer *input, GPUBuffer *weights, GPUBuffer *bias,
                          GPUBuffer *output, size_t B, size_t in_f, size_t out_f) {
    cl_kernel kernel = g_ctx.dense_forward_kernel;
    
    int iB = (int)B, i_in_f = (int)in_f, i_out_f = (int)out_f;
    
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &input->buffer);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &weights->buffer);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &bias->buffer);
    clSetKernelArg(kernel, 3, sizeof(cl_mem), &output->buffer);
    clSetKernelArg(kernel, 4, sizeof(int), &iB);
    clSetKernelArg(kernel, 5, sizeof(int), &i_in_f);
    clSetKernelArg(kernel, 6, sizeof(int), &i_out_f);
    
    size_t global = round_up(B * out_f, 256);
    size_t local = 256;
    
    clEnqueueNDRangeKernel(g_ctx.queue, kernel, 1, NULL, &global, &local, 0, NULL, NULL);
}

void opencl_relu_forward(GPUBuffer *input, GPUBuffer *output, GPUBuffer *mask, size_t n) {
    cl_kernel kernel = g_ctx.relu_forward_kernel;
    
    int in = (int)n;
    
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &input->buffer);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &output->buffer);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &mask->buffer);
    clSetKernelArg(kernel, 3, sizeof(int), &in);
    
    size_t global = round_up(n, 256);
    size_t local = 256;
    
    clEnqueueNDRangeKernel(g_ctx.queue, kernel, 1, NULL, &global, &local, 0, NULL, NULL);
}

void opencl_sigmoid_forward(GPUBuffer *input, GPUBuffer *output, size_t n) {
    cl_kernel kernel = g_ctx.sigmoid_forward_kernel;
    
    int in = (int)n;
    
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &input->buffer);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &output->buffer);
    clSetKernelArg(kernel, 2, sizeof(int), &in);
    
    size_t global = round_up(n, 256);
    size_t local = 256;
    
    clEnqueueNDRangeKernel(g_ctx.queue, kernel, 1, NULL, &global, &local, 0, NULL, NULL);
}

void opencl_sigmoid_backward(GPUBuffer *grad_output, GPUBuffer *sigmoid_output,
                             GPUBuffer *grad_input, size_t n) {
    cl_kernel kernel = g_ctx.sigmoid_backward_kernel;

    int in = (int)n;
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &grad_output->buffer);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &sigmoid_output->buffer);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &grad_input->buffer);
    clSetKernelArg(kernel, 3, sizeof(int), &in);

    size_t global = round_up(n, 256);
    size_t local = 256;

    clEnqueueNDRangeKernel(g_ctx.queue, kernel, 1, NULL, &global, &local, 0, NULL, NULL);
}

void opencl_dropout_forward(GPUBuffer *input, GPUBuffer *output, GPUBuffer *mask,
                            float keep_prob, unsigned int seed, size_t n) {
    cl_kernel kernel = g_ctx.dropout_forward_kernel;

    int in = (int)n;
    cl_uint useed = (cl_uint)seed;
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &input->buffer);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &output->buffer);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &mask->buffer);
    clSetKernelArg(kernel, 3, sizeof(float), &keep_prob);
    clSetKernelArg(kernel, 4, sizeof(cl_uint), &useed);
    clSetKernelArg(kernel, 5, sizeof(int), &in);

    size_t global = round_up(n, 256);
    size_t local = 256;

    clEnqueueNDRangeKernel(g_ctx.queue, kernel, 1, NULL, &global, &local, 0, NULL, NULL);
}

void opencl_dropout_backward(GPUBuffer *grad_output, GPUBuffer *mask,
                             GPUBuffer *grad_input, size_t n) {
    cl_kernel kernel = g_ctx.relu_backward_kernel;

    int in = (int)n;
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &grad_output->buffer);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &mask->buffer);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &grad_input->buffer);
    clSetKernelArg(kernel, 3, sizeof(int), &in);

    size_t global = round_up(n, 256);
    size_t local = 256;

    clEnqueueNDRangeKernel(g_ctx.queue, kernel, 1, NULL, &global, &local, 0, NULL, NULL);
}

void opencl_conv2d_backward_weights(GPUBuffer *input_cache, GPUBuffer *grad_output,
                                    GPUBuffer *d_weights, GPUBuffer *d_bias,
                                    size_t B, size_t in_c, size_t out_c,
                                    size_t in_h, size_t in_w, size_t out_h, size_t out_w,
                                    size_t k, size_t stride, size_t padding) {
    if (k == 3 && stride == 1 && padding == 1 && in_c == 1) {
        cl_kernel kernel = g_ctx.conv2d_backward_weights_winograd_3x3_kernel;
        int iB = (int)B, i_in_c = (int)in_c, i_out_c = (int)out_c;
        int i_in_h = (int)in_h, i_in_w = (int)in_w;
        int i_out_h = (int)out_h, i_out_w = (int)out_w;

        size_t tiles_w = (out_w + 1) / 2;
        size_t tiles_h = (out_h + 1) / 2;
        size_t tiles_per_oc = B * tiles_h * tiles_w;
        size_t groups = out_c * tiles_per_oc;
        size_t partial_weights_bytes = groups * in_c * 9 * sizeof(float);
        size_t partial_bias_bytes = groups * sizeof(float);

        cl_mem partial_weights = clCreateBuffer(g_ctx.context, CL_MEM_READ_WRITE, partial_weights_bytes, NULL, NULL);
        cl_mem partial_bias = clCreateBuffer(g_ctx.context, CL_MEM_READ_WRITE, partial_bias_bytes, NULL, NULL);

        clSetKernelArg(kernel, 0, sizeof(cl_mem), &input_cache->buffer);
        clSetKernelArg(kernel, 1, sizeof(cl_mem), &grad_output->buffer);
        clSetKernelArg(kernel, 2, sizeof(cl_mem), &partial_weights);
        clSetKernelArg(kernel, 3, sizeof(cl_mem), &partial_bias);
        clSetKernelArg(kernel, 4, sizeof(int), &iB);
        clSetKernelArg(kernel, 5, sizeof(int), &i_in_c);
        clSetKernelArg(kernel, 6, sizeof(int), &i_out_c);
        clSetKernelArg(kernel, 7, sizeof(int), &i_in_h);
        clSetKernelArg(kernel, 8, sizeof(int), &i_in_w);
        clSetKernelArg(kernel, 9, sizeof(int), &i_out_h);
        clSetKernelArg(kernel, 10, sizeof(int), &i_out_w);

        size_t global = round_up(groups, 256);
        size_t local = 256;
        clEnqueueNDRangeKernel(g_ctx.queue, kernel, 1, NULL, &global, &local, 0, NULL, NULL);

        kernel = g_ctx.conv2d_backward_weights_winograd_reduce_kernel;
        clSetKernelArg(kernel, 0, sizeof(cl_mem), &partial_weights);
        clSetKernelArg(kernel, 1, sizeof(cl_mem), &partial_bias);
        clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_weights->buffer);
        clSetKernelArg(kernel, 3, sizeof(cl_mem), &d_bias->buffer);
        clSetKernelArg(kernel, 4, sizeof(int), &i_out_c);
        clSetKernelArg(kernel, 5, sizeof(int), &i_in_c);
        int i_tiles_per_oc = (int)tiles_per_oc;
        clSetKernelArg(kernel, 6, sizeof(int), &i_tiles_per_oc);

        size_t reduce_global = round_up(out_c * in_c * 9, 256);
        clEnqueueNDRangeKernel(g_ctx.queue, kernel, 1, NULL, &reduce_global, &local, 0, NULL, NULL);

        clReleaseMemObject(partial_weights);
        clReleaseMemObject(partial_bias);
        return;
    }

    cl_kernel kernel = g_ctx.conv2d_backward_weights_kernel;

    if (k == 3 && stride == 1 && padding == 1 && in_c == 1) {
        kernel = g_ctx.conv2d_backward_weights_3x3_ic1_kernel;
    } else if (k == 3 && stride == 1 && padding == 1) {
        kernel = g_ctx.conv2d_backward_weights_3x3_kernel;
    }

    int iB = (int)B, i_in_c = (int)in_c, i_out_c = (int)out_c;
    int i_in_h = (int)in_h, i_in_w = (int)in_w;
    int i_out_h = (int)out_h, i_out_w = (int)out_w;
    int ik = (int)k, i_stride = (int)stride, i_padding = (int)padding;

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &input_cache->buffer);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &grad_output->buffer);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_weights->buffer);
    clSetKernelArg(kernel, 3, sizeof(cl_mem), &d_bias->buffer);
    if (kernel == g_ctx.conv2d_backward_weights_3x3_ic1_kernel) {
        clSetKernelArg(kernel, 4, sizeof(int), &iB);
        clSetKernelArg(kernel, 5, sizeof(int), &i_out_c);
        clSetKernelArg(kernel, 6, sizeof(int), &i_in_h);
        clSetKernelArg(kernel, 7, sizeof(int), &i_in_w);
        clSetKernelArg(kernel, 8, sizeof(int), &i_out_h);
        clSetKernelArg(kernel, 9, sizeof(int), &i_out_w);
    } else if (kernel == g_ctx.conv2d_backward_weights_3x3_kernel) {
        clSetKernelArg(kernel, 4, sizeof(int), &iB);
        clSetKernelArg(kernel, 5, sizeof(int), &i_in_c);
        clSetKernelArg(kernel, 6, sizeof(int), &i_out_c);
        clSetKernelArg(kernel, 7, sizeof(int), &i_in_h);
        clSetKernelArg(kernel, 8, sizeof(int), &i_in_w);
        clSetKernelArg(kernel, 9, sizeof(int), &i_out_h);
        clSetKernelArg(kernel, 10, sizeof(int), &i_out_w);
    } else {
        clSetKernelArg(kernel, 4, sizeof(int), &iB);
        clSetKernelArg(kernel, 5, sizeof(int), &i_in_c);
        clSetKernelArg(kernel, 6, sizeof(int), &i_out_c);
        clSetKernelArg(kernel, 7, sizeof(int), &i_in_h);
        clSetKernelArg(kernel, 8, sizeof(int), &i_in_w);
        clSetKernelArg(kernel, 9, sizeof(int), &i_out_h);
        clSetKernelArg(kernel, 10, sizeof(int), &i_out_w);
        clSetKernelArg(kernel, 11, sizeof(int), &ik);
        clSetKernelArg(kernel, 12, sizeof(int), &i_stride);
        clSetKernelArg(kernel, 13, sizeof(int), &i_padding);
    }

    size_t total = (kernel == g_ctx.conv2d_backward_weights_3x3_ic1_kernel)
        ? (out_c * 9 + out_c)
        : (kernel == g_ctx.conv2d_backward_weights_3x3_kernel)
        ? (out_c * in_c * 9 + out_c)
        : (out_c * in_c * k * k + out_c);
    size_t global = round_up(total, 256);
    size_t local = 256;

    clEnqueueNDRangeKernel(g_ctx.queue, kernel, 1, NULL, &global, &local, 0, NULL, NULL);
}

void opencl_dense_backward_weights(GPUBuffer *input_cache, GPUBuffer *grad_output,
                                   GPUBuffer *d_weights, size_t B, size_t in_f, size_t out_f) {
    cl_kernel kernel = g_ctx.dense_backward_weights_kernel;

    int iB = (int)B, i_in_f = (int)in_f, i_out_f = (int)out_f;

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &input_cache->buffer);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &grad_output->buffer);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_weights->buffer);
    clSetKernelArg(kernel, 3, sizeof(int), &iB);
    clSetKernelArg(kernel, 4, sizeof(int), &i_in_f);
    clSetKernelArg(kernel, 5, sizeof(int), &i_out_f);

    size_t global = round_up(in_f * out_f, 256);
    size_t local = 256;

    clEnqueueNDRangeKernel(g_ctx.queue, kernel, 1, NULL, &global, &local, 0, NULL, NULL);
}

void opencl_dense_backward_bias(GPUBuffer *grad_output, GPUBuffer *d_bias,
                                size_t B, size_t out_f) {
    cl_kernel kernel = g_ctx.dense_backward_bias_kernel;

    int iB = (int)B, i_out_f = (int)out_f;

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &grad_output->buffer);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_bias->buffer);
    clSetKernelArg(kernel, 2, sizeof(int), &iB);
    clSetKernelArg(kernel, 3, sizeof(int), &i_out_f);

    size_t global = round_up(out_f, 256);
    size_t local = 256;

    clEnqueueNDRangeKernel(g_ctx.queue, kernel, 1, NULL, &global, &local, 0, NULL, NULL);
}

void opencl_adamw_update(GPUBuffer *weights, GPUBuffer *gradients, GPUBuffer *m, GPUBuffer *v,
                         float lr, float beta1, float beta2, float eps, float wd, float m_corr, float v_corr, size_t n) {
    cl_kernel kernel = g_ctx.adamw_update_kernel;

    int in = (int)n;
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &weights->buffer);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &gradients->buffer);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &m->buffer);
    clSetKernelArg(kernel, 3, sizeof(cl_mem), &v->buffer);
    clSetKernelArg(kernel, 4, sizeof(float), &lr);
    clSetKernelArg(kernel, 5, sizeof(float), &beta1);
    clSetKernelArg(kernel, 6, sizeof(float), &beta2);
    clSetKernelArg(kernel, 7, sizeof(float), &eps);
    clSetKernelArg(kernel, 8, sizeof(float), &wd);
    clSetKernelArg(kernel, 9, sizeof(float), &m_corr);
    clSetKernelArg(kernel, 10, sizeof(float), &v_corr);
    clSetKernelArg(kernel, 11, sizeof(int), &in);

    size_t global = round_up(n, 256);
    size_t local = 256;

    clEnqueueNDRangeKernel(g_ctx.queue, kernel, 1, NULL, &global, &local, 0, NULL, NULL);
}

void opencl_zero_buffer(GPUBuffer *buffer, size_t n) {
    cl_kernel kernel = g_ctx.zero_buffer_kernel;

    int in = (int)n;
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &buffer->buffer);
    clSetKernelArg(kernel, 1, sizeof(int), &in);

    size_t global = round_up(n, 256);
    size_t local = 256;

    clEnqueueNDRangeKernel(g_ctx.queue, kernel, 1, NULL, &global, &local, 0, NULL, NULL);
}

#endif
