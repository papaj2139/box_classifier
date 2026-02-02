//conv2D forward - tiled with local memory
//work groups: (out_w/TILE, out_h/TILE, B*out_c)
#define TILE_W 8
#define TILE_H 8

__kernel void conv2d_forward(
    __global const float *input,   //[B, in_c, in_h, in_w]
    __global const float *weights, //[out_c, in_c, k, k]
    __global const float *bias,    //[out_c]
    __global float *output,        //[B, out_c, out_h, out_w]
    const int B,
    const int in_c,
    const int out_c,
    const int in_h,
    const int in_w,
    const int out_h,
    const int out_w,
    const int k,
    const int stride,
    const int padding
) {
    int gid = get_global_id(0);
    int total = B * out_c * out_h * out_w;
    if (gid >= total) return;
    
    //decode indices
    int ow = gid % out_w;
    int temp = gid / out_w;
    int oh = temp % out_h;
    temp = temp / out_h;
    int oc = temp % out_c;
    int b = temp / out_c;
    
    float sum = bias[oc];
    
    //unroll small kernel sizes for performance
    #pragma unroll 4
    for (int ic = 0; ic < in_c; ic++) {
        int in_base = b * (in_c * in_h * in_w) + ic * (in_h * in_w);
        int w_base = oc * (in_c * k * k) + ic * (k * k);
        
        for (int kh = 0; kh < k; kh++) {
            int ih = oh * stride + kh - padding;
            if (ih >= 0 && ih < in_h) {
                for (int kw = 0; kw < k; kw++) {
                    int iw = ow * stride + kw - padding;
                    if (iw >= 0 && iw < in_w) {
                        sum += input[in_base + ih * in_w + iw] * 
                               weights[w_base + kh * k + kw];
                    }
                }
            }
        }
    }
    
    output[gid] = sum;
}

//dense forward - vectorized
__kernel void dense_forward(
    __global const float *input,   //[B, in_f]
    __global const float *weights, //[in_f, out_f]
    __global const float *bias,    //[out_f]
    __global float *output,        //[B, out_f]
    const int B,
    const int in_f,
    const int out_f
) {
    int gid = get_global_id(0);
    if (gid >= B * out_f) return;
    
    int j = gid % out_f;
    int b = gid / out_f;
    
    float sum = bias[j];
    
    //vectorized accumulation with manual unrolling
    int i = 0;
    float4 acc = (float4)(0.0f);
    
    //process 4 at a time
    for (; i + 3 < in_f; i += 4) {
        float4 inp = (float4)(
            input[b * in_f + i],
            input[b * in_f + i + 1],
            input[b * in_f + i + 2],
            input[b * in_f + i + 3]
        );
        float4 wgt = (float4)(
            weights[(i) * out_f + j],
            weights[(i + 1) * out_f + j],
            weights[(i + 2) * out_f + j],
            weights[(i + 3) * out_f + j]
        );
        acc += inp * wgt;
    }
    sum += acc.x + acc.y + acc.z + acc.w;
    
    //handle remainder
    for (; i < in_f; i++) {
        sum += input[b * in_f + i] * weights[i * out_f + j];
    }
    
    output[gid] = sum;
}

//relu forward
__kernel void relu_forward(
    __global const float *input,
    __global float *output,
    __global float *mask,
    const int n
) {
    int gid = get_global_id(0);
    if (gid >= n) return;
    
    float val = input[gid];
    float m = val > 0.0f ? 1.0f : 0.0f;
    output[gid] = val * m;
    mask[gid] = m;
}

//relu backward
__kernel void relu_backward(
    __global const float *grad_output,
    __global const float *mask,
    __global float *grad_input,
    const int n
) {
    int gid = get_global_id(0);
    if (gid >= n) return;
    
    grad_input[gid] = grad_output[gid] * mask[gid];
}

//sigmoid forward
__kernel void sigmoid_forward(
    __global const float *input,
    __global float *output,
    const int n
) {
    int gid = get_global_id(0);
    if (gid >= n) return;
    
    float x = input[gid];
    //numerically stable sigmoid
    if (x >= 0) {
        output[gid] = 1.0f / (1.0f + exp(-x));
    } else {
        float ex = exp(x);
        output[gid] = ex / (1.0f + ex);
    }
}

//sigmoid backward
__kernel void sigmoid_backward(
    __global const float *grad_output,
    __global const float *sigmoid_output,
    __global float *grad_input,
    const int n
) {
    int gid = get_global_id(0);
    if (gid >= n) return;
    
    float s = sigmoid_output[gid];
    grad_input[gid] = grad_output[gid] * s * (1.0f - s);
}

//maxpool forward
__kernel void maxpool_forward(
    __global const float *input,
    __global float *output,
    __global float *indices,
    const int B,
    const int C,
    const int in_h,
    const int in_w,
    const int out_h,
    const int out_w,
    const int pool_size,
    const int stride
) {
    int gid = get_global_id(0);
    int total = B * C * out_h * out_w;
    if (gid >= total) return;
    
    int ow = gid % out_w;
    int temp = gid / out_w;
    int oh = temp % out_h;
    temp = temp / out_h;
    int c = temp % C;
    int b = temp / C;
    
    float max_val = -1e30f;
    int max_idx = 0;
    
    int in_base = b * (C * in_h * in_w) + c * (in_h * in_w);
    
    for (int ph = 0; ph < pool_size; ph++) {
        int ih = oh * stride + ph;
        if (ih < in_h) {
            for (int pw = 0; pw < pool_size; pw++) {
                int iw = ow * stride + pw;
                if (iw < in_w) {
                    float val = input[in_base + ih * in_w + iw];
                    if (val > max_val) {
                        max_val = val;
                        max_idx = ph * pool_size + pw;
                    }
                }
            }
        }
    }
    
    output[gid] = max_val;
    indices[gid] = (float)max_idx;
}

//maxpool backward
__kernel void maxpool_backward(
    __global const float *grad_output,
    __global const float *indices,
    __global float *grad_input,
    const int B,
    const int C,
    const int in_h,
    const int in_w,
    const int out_h,
    const int out_w,
    const int pool_size,
    const int stride
) {
    int gid = get_global_id(0);
    int total = B * C * out_h * out_w;
    if (gid >= total) return;
    
    int ow = gid % out_w;
    int temp = gid / out_w;
    int oh = temp % out_h;
    temp = temp / out_h;
    int c = temp % C;
    int b = temp / C;
    
    float grad = grad_output[gid];
    int max_idx = (int)indices[gid];
    int ph = max_idx / pool_size;
    int pw = max_idx % pool_size;
    int ih = oh * stride + ph;
    int iw = ow * stride + pw;
    
    int gi_idx = b * (C * in_h * in_w) + c * (in_h * in_w) + ih * in_w + iw;
    
    //atomic add for potential overlaps (rare with stride >= pool_size)
    //for non-overlapping pooling, this is safe without atomic
    grad_input[gi_idx] += grad;
}

//conv2d backward - input gradient
__kernel void conv2d_backward_input(
    __global const float *grad_output,
    __global const float *weights,
    __global float *grad_input,
    const int B,
    const int in_c,
    const int out_c,
    const int in_h,
    const int in_w,
    const int out_h,
    const int out_w,
    const int k,
    const int stride,
    const int padding
) {
    int gid = get_global_id(0);
    int total = B * in_c * in_h * in_w;
    if (gid >= total) return;
    
    int iw = gid % in_w;
    int temp = gid / in_w;
    int ih = temp % in_h;
    temp = temp / in_h;
    int ic = temp % in_c;
    int b = temp / in_c;
    
    float sum = 0.0f;
    
    for (int oc = 0; oc < out_c; oc++) {
        int go_base = b * (out_c * out_h * out_w) + oc * (out_h * out_w);
        int w_base = oc * (in_c * k * k) + ic * (k * k);
        
        for (int kh = 0; kh < k; kh++) {
            int oh_unstrided = ih + padding - kh;
            if (oh_unstrided >= 0 && oh_unstrided % stride == 0) {
                int oh = oh_unstrided / stride;
                if (oh < out_h) {
                    for (int kw = 0; kw < k; kw++) {
                        int ow_unstrided = iw + padding - kw;
                        if (ow_unstrided >= 0 && ow_unstrided % stride == 0) {
                            int ow = ow_unstrided / stride;
                            if (ow < out_w) {
                                sum += grad_output[go_base + oh * out_w + ow] *
                                       weights[w_base + kh * k + kw];
                            }
                        }
                    }
                }
            }
        }
    }
    
    grad_input[gid] = sum;
}

//weight gradient kernel
__kernel void conv2d_backward_weights(
    __global const float *input_cache,   //[B, in_c, in_h, in_w]
    __global const float *grad_output,   //[B, out_c, out_h, out_w]
    __global float *d_weights,           //[out_c, in_c, k, k]
    __global float *d_bias,              //[out_c]
    const int B,
    const int in_c,
    const int out_c,
    const int in_h,
    const int in_w,
    const int out_h,
    const int out_w,
    const int k,
    const int stride,
    const int padding
) {
    //each work item computes one weight gradient d_W[oc, ic, kh, kw]
    int gid = get_global_id(0);
    int total_weights = out_c * in_c * k * k;
    
    if (gid < total_weights) {
        int kw = gid % k;
        int temp = gid / k;
        int kh = temp % k;
        temp = temp / k;
        int ic = temp % in_c;
        int oc = temp / in_c;
        
        float grad_sum = 0.0f;
        
        for (int b = 0; b < B; b++) {
            int in_base = b * (in_c * in_h * in_w) + ic * (in_h * in_w);
            int go_base = b * (out_c * out_h * out_w) + oc * (out_h * out_w);
            
            for (int oh = 0; oh < out_h; oh++) {
                int ih = oh * stride + kh - padding;
                if (ih >= 0 && ih < in_h) {
                    for (int ow = 0; ow < out_w; ow++) {
                        int iw = ow * stride + kw - padding;
                        if (iw >= 0 && iw < in_w) {
                            grad_sum += input_cache[in_base + ih * in_w + iw] *
                                       grad_output[go_base + oh * out_w + ow];
                        }
                    }
                }
            }
        }
        
        d_weights[gid] += grad_sum;
    }
    
    //bias gradient - one per output channel
    //use separate work items for bias
    int bias_gid = gid - total_weights;
    if (bias_gid >= 0 && bias_gid < out_c) {
        int oc = bias_gid;
        float bias_grad = 0.0f;
        
        for (int b = 0; b < B; b++) {
            int go_base = b * (out_c * out_h * out_w) + oc * (out_h * out_w);
            for (int oh = 0; oh < out_h; oh++) {
                for (int ow = 0; ow < out_w; ow++) {
                    bias_grad += grad_output[go_base + oh * out_w + ow];
                }
            }
        }
        
        d_bias[oc] += bias_grad;
    }
}

//dense backward - weight gradient
__kernel void dense_backward_weights(
    __global const float *input_cache,  //[B, in_f]
    __global const float *grad_output,  //[B, out_f]
    __global float *d_weights,          //[in_f, out_f]
    const int B,
    const int in_f,
    const int out_f
) {
    int gid = get_global_id(0);
    if (gid >= in_f * out_f) return;
    
    int j = gid % out_f;
    int i = gid / out_f;
    
    float sum = 0.0f;
    for (int b = 0; b < B; b++) {
        sum += input_cache[b * in_f + i] * grad_output[b * out_f + j];
    }
    
    d_weights[gid] += sum;
}

//dense backward - input gradient
__kernel void dense_backward_input(
    __global const float *grad_output,  //[B, out_f]
    __global const float *weights,      //[in_f, out_f]
    __global float *grad_input,         //[B, in_f]
    const int B,
    const int in_f,
    const int out_f
) {
    int gid = get_global_id(0);
    if (gid >= B * in_f) return;
    
    int i = gid % in_f;
    int b = gid / in_f;
    
    float sum = 0.0f;
    
    //vectorized
    int j = 0;
    float4 acc = (float4)(0.0f);
    for (; j + 3 < out_f; j += 4) {
        float4 go = (float4)(
            grad_output[b * out_f + j],
            grad_output[b * out_f + j + 1],
            grad_output[b * out_f + j + 2],
            grad_output[b * out_f + j + 3]
        );
        float4 wgt = (float4)(
            weights[i * out_f + j],
            weights[i * out_f + j + 1],
            weights[i * out_f + j + 2],
            weights[i * out_f + j + 3]
        );
        acc += go * wgt;
    }
    sum = acc.x + acc.y + acc.z + acc.w;
    
    for (; j < out_f; j++) {
        sum += grad_output[b * out_f + j] * weights[i * out_f + j];
    }
    
    grad_input[gid] = sum;
}

//dense backward - bias gradient
__kernel void dense_backward_bias(
    __global const float *grad_output,  //[B, out_f]
    __global float *d_bias,             //[out_f]
    const int B,
    const int out_f
) {
    int j = get_global_id(0);
    if (j >= out_f) return;
    
    float sum = 0.0f;
    for (int b = 0; b < B; b++) {
        sum += grad_output[b * out_f + j];
    }
    
    d_bias[j] += sum;
}

//sgd update
__kernel void sgd_update(
    __global float *weights,
    __global const float *gradients,
    const float lr,
    const int n
) {
    int gid = get_global_id(0);
    if (gid >= n) return;
    
    weights[gid] -= lr * gradients[gid];
}

//zero buffer
__kernel void zero_buffer(
    __global float *buffer,
    const int n
) {
    int gid = get_global_id(0);
    if (gid >= n) return;
    buffer[gid] = 0.0f;
}
