//conv2D forward - tiled with local memory
//work groups: (out_w/TILE, out_h/TILE, B*out_c)
#define TILE_W 8
#define TILE_H 8

static inline void winograd_input_transform(
    const float d[4][4],
    float v[4][4]
) {
    const float BT[4][4] = {
        {1.0f, 0.0f, -1.0f, 0.0f},
        {0.0f, 1.0f, 1.0f, 0.0f},
        {0.0f, -1.0f, 1.0f, 0.0f},
        {0.0f, 1.0f, 0.0f, -1.0f}
    };
    const float B[4][4] = {
        {1.0f, 0.0f, 0.0f, 0.0f},
        {0.0f, 1.0f, -1.0f, 1.0f},
        {-1.0f, 1.0f, 1.0f, 0.0f},
        {0.0f, 0.0f, 0.0f, -1.0f}
    };

    float tmp[4][4] = {0};
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            float sum = 0.0f;
            for (int k = 0; k < 4; k++) {
                sum += BT[i][k] * d[k][j];
            }
            tmp[i][j] = sum;
        }
    }

    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            float sum = 0.0f;
            for (int k = 0; k < 4; k++) {
                sum += tmp[i][k] * B[k][j];
            }
            v[i][j] = sum;
        }
    }
}

static inline void winograd_filter_transform(
    const float g[3][3],
    float u[4][4]
) {
    const float G[4][3] = {
        {1.0f, 0.0f, 0.0f},
        {0.5f, 0.5f, 0.5f},
        {0.5f, -0.5f, 0.5f},
        {0.0f, 0.0f, 1.0f}
    };

    float tmp[4][3] = {0};
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 3; j++) {
            float sum = 0.0f;
            for (int k = 0; k < 3; k++) {
                sum += G[i][k] * g[k][j];
            }
            tmp[i][j] = sum;
        }
    }

    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            float sum = 0.0f;
            for (int k = 0; k < 3; k++) {
                sum += tmp[i][k] * G[j][k];
            }
            u[i][j] = sum;
        }
    }
}

static inline void winograd_output_transform(
    const float m[4][4],
    float y[2][2]
) {
    const float AT[2][4] = {
        {1.0f, 1.0f, 1.0f, 0.0f},
        {0.0f, 1.0f, -1.0f, -1.0f}
    };
    const float A[4][2] = {
        {1.0f, 0.0f},
        {1.0f, 1.0f},
        {1.0f, -1.0f},
        {0.0f, -1.0f}
    };

    float tmp[2][4] = {0};
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 4; j++) {
            float sum = 0.0f;
            for (int k = 0; k < 4; k++) {
                sum += AT[i][k] * m[k][j];
            }
            tmp[i][j] = sum;
        }
    }

    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            float sum = 0.0f;
            for (int k = 0; k < 4; k++) {
                sum += tmp[i][k] * A[k][j];
            }
            y[i][j] = sum;
        }
    }
}

static inline void winograd_grad_output_transform(
    const float e[2][2],
    float m[4][4]
) {
    const float A[4][2] = {
        {1.0f, 0.0f},
        {1.0f, 1.0f},
        {1.0f, -1.0f},
        {0.0f, -1.0f}
    };
    const float AT[2][4] = {
        {1.0f, 1.0f, 1.0f, 0.0f},
        {0.0f, 1.0f, -1.0f, -1.0f}
    };

    float tmp[4][2] = {0};
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 2; j++) {
            float sum = 0.0f;
            for (int k = 0; k < 2; k++) {
                sum += A[i][k] * e[k][j];
            }
            tmp[i][j] = sum;
        }
    }

    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            float sum = 0.0f;
            for (int k = 0; k < 2; k++) {
                sum += tmp[i][k] * AT[k][j];
            }
            m[i][j] = sum;
        }
    }
}

__kernel void conv2d_transform_weights_winograd_3x3(
    __global const float *weights,
    __global float *transformed_weights,
    const int out_c,
    const int in_c,
    const int reverse
) {
    int gid = get_global_id(0);
    int total = out_c * in_c;
    if (gid >= total) return;

    int ic = gid % in_c;
    int oc = gid / in_c;
    int w_base = oc * (in_c * 9) + ic * 9;

    float g[3][3];
    for (int kh = 0; kh < 3; kh++) {
        for (int kw = 0; kw < 3; kw++) {
            int src_kh = reverse ? (2 - kh) : kh;
            int src_kw = reverse ? (2 - kw) : kw;
            g[kh][kw] = weights[w_base + src_kh * 3 + src_kw];
        }
    }

    float u[4][4];
    winograd_filter_transform(g, u);

    int t_base = gid * 16;
    int idx = 0;
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            transformed_weights[t_base + idx++] = u[i][j];
        }
    }
}

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

__kernel void conv2d_forward_winograd_3x3(
    __global const float *input,   //[B, in_c, in_h, in_w]
    __global const float *weights_t, //[out_c, in_c, 4, 4]
    __global const float *bias,    //[out_c]
    __global float *output,        //[B, out_c, out_h, out_w]
    const int B,
    const int in_c,
    const int out_c,
    const int in_h,
    const int in_w,
    const int out_h,
    const int out_w
) {
    int gid = get_global_id(0);
    int tile_h = (out_h + 1) / 2;
    int tile_w = (out_w + 1) / 2;
    int total = B * out_c * tile_h * tile_w;
    if (gid >= total) return;

    int tx = gid % tile_w;
    int temp = gid / tile_w;
    int ty = temp % tile_h;
    temp = temp / tile_h;
    int oc = temp % out_c;
    int b = temp / out_c;

    float m[4][4] = {0};
    for (int ic = 0; ic < in_c; ic++) {
        float d[4][4] = {0};

        int in_base = b * (in_c * in_h * in_w) + ic * (in_h * in_w);
        for (int i = 0; i < 4; i++) {
            int ih = ty * 2 + i - 1;
            for (int j = 0; j < 4; j++) {
                int iw = tx * 2 + j - 1;
                if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                    d[i][j] = input[in_base + ih * in_w + iw];
                }
            }
        }

        float u[4][4];
        float v[4][4];
        int t_base = (oc * in_c + ic) * 16;
        int idx = 0;
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                u[i][j] = weights_t[t_base + idx++];
            }
        }
        winograd_input_transform(d, v);

        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                m[i][j] += u[i][j] * v[i][j];
            }
        }
    }

    float y[2][2];
    winograd_output_transform(m, y);

    for (int i = 0; i < 2; i++) {
        int oh = ty * 2 + i;
        if (oh < out_h) {
            for (int j = 0; j < 2; j++) {
                int ow = tx * 2 + j;
                if (ow < out_w) {
                    int out_idx = b * (out_c * out_h * out_w) + oc * (out_h * out_w) + oh * out_w + ow;
                    output[out_idx] = y[i][j] + bias[oc];
                }
            }
        }
    }
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

static inline uint hash_u32(uint x) {
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

//dropout forward
__kernel void dropout_forward(
    __global const float *input,
    __global float *output,
    __global float *mask,
    const float keep_prob,
    const uint seed,
    const int n
) {
    int gid = get_global_id(0);
    if (gid >= n) return;

    if (keep_prob <= 0.0f) {
        output[gid] = 0.0f;
        mask[gid] = 0.0f;
        return;
    }

    uint state = hash_u32((uint)gid ^ seed);
    float r = (float)(state & 0x00ffffffu) * (1.0f / 16777216.0f);
    float m = r < keep_prob ? (1.0f / keep_prob) : 0.0f;
    output[gid] = input[gid] * m;
    mask[gid] = m;
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

__kernel void conv2d_backward_input_winograd_3x3(
    __global const float *grad_output,
    __global const float *weights_t,
    __global float *grad_input,
    const int B,
    const int in_c,
    const int out_c,
    const int in_h,
    const int in_w,
    const int out_h,
    const int out_w
) {
    int gid = get_global_id(0);
    int tile_h = (in_h + 1) / 2;
    int tile_w = (in_w + 1) / 2;
    int total = B * in_c * tile_h * tile_w;
    if (gid >= total) return;

    int tx = gid % tile_w;
    int temp = gid / tile_w;
    int ty = temp % tile_h;
    temp = temp / tile_h;
    int ic = temp % in_c;
    int b = temp / in_c;

    float m[4][4] = {0};
    for (int oc = 0; oc < out_c; oc++) {
        float d[4][4] = {0};

        int go_base = b * (out_c * out_h * out_w) + oc * (out_h * out_w);
        for (int i = 0; i < 4; i++) {
            int oh = ty * 2 + i - 1;
            for (int j = 0; j < 4; j++) {
                int ow = tx * 2 + j - 1;
                if (oh >= 0 && oh < out_h && ow >= 0 && ow < out_w) {
                    d[i][j] = grad_output[go_base + oh * out_w + ow];
                }
            }
        }

        float u[4][4];
        float v[4][4];
        int t_base = (oc * in_c + ic) * 16;
        int idx = 0;
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                u[i][j] = weights_t[t_base + idx++];
            }
        }
        winograd_input_transform(d, v);

        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                m[i][j] += u[i][j] * v[i][j];
            }
        }
    }

    float y[2][2];
    winograd_output_transform(m, y);

    for (int i = 0; i < 2; i++) {
        int ih = ty * 2 + i;
        if (ih < in_h) {
            for (int j = 0; j < 2; j++) {
                int iw = tx * 2 + j;
                if (iw < in_w) {
                    int out_idx = b * (in_c * in_h * in_w) + ic * (in_h * in_w) + ih * in_w + iw;
                    grad_input[out_idx] = y[i][j];
                }
            }
        }
    }
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

//specialized 3x3, stride=1, padding=1 weight gradient
__kernel void conv2d_backward_weights_3x3_s1p1(
    __global const float *input_cache,   //[B, in_c, in_h, in_w]
    __global const float *grad_output,   //[B, out_c, out_h, out_w]
    __global float *d_weights,           //[out_c, in_c, 3, 3]
    __global float *d_bias,              //[out_c]
    const int B,
    const int in_c,
    const int out_c,
    const int in_h,
    const int in_w,
    const int out_h,
    const int out_w
) {
    int gid = get_global_id(0);
    int total_weights = out_c * in_c * 9;

    if (gid < total_weights) {
        int kw = gid % 3;
        int temp = gid / 3;
        int kh = temp % 3;
        temp = temp / 3;
        int ic = temp % in_c;
        int oc = temp / in_c;

        int oh_start = (kh == 0) ? 1 : 0;
        int oh_end = (kh == 2) ? (out_h - 2) : (out_h - 1);
        int ow_start = (kw == 0) ? 1 : 0;
        int ow_end = (kw == 2) ? (out_w - 2) : (out_w - 1);

        float grad_sum = 0.0f;

        for (int b = 0; b < B; b++) {
            int in_base = b * (in_c * in_h * in_w) + ic * (in_h * in_w);
            int go_base = b * (out_c * out_h * out_w) + oc * (out_h * out_w);

            for (int oh = oh_start; oh <= oh_end; oh++) {
                int ih = oh + kh - 1;
                for (int ow = ow_start; ow <= ow_end; ow++) {
                    int iw = ow + kw - 1;
                    grad_sum += input_cache[in_base + ih * in_w + iw] *
                                grad_output[go_base + oh * out_w + ow];
                }
            }
        }

        d_weights[gid] += grad_sum;
        return;
    }

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

//specialized 3x3, stride=1, padding=1, in_c=1 weight gradient
__kernel void conv2d_backward_weights_3x3_ic1_s1p1(
    __global const float *input_cache,   //[B, 1, in_h, in_w]
    __global const float *grad_output,   //[B, out_c, out_h, out_w]
    __global float *d_weights,           //[out_c, 1, 3, 3]
    __global float *d_bias,              //[out_c]
    const int B,
    const int out_c,
    const int in_h,
    const int in_w,
    const int out_h,
    const int out_w
) {
    int gid = get_global_id(0);
    int total_weights = out_c * 9;

    if (gid < total_weights) {
        int kw = gid % 3;
        int temp = gid / 3;
        int kh = temp % 3;
        int oc = temp / 3;

        int oh_start = (kh == 0) ? 1 : 0;
        int oh_end = (kh == 2) ? (out_h - 2) : (out_h - 1);
        int ow_start = (kw == 0) ? 1 : 0;
        int ow_end = (kw == 2) ? (out_w - 2) : (out_w - 1);

        float grad_sum = 0.0f;

        for (int b = 0; b < B; b++) {
            int in_base = b * (in_h * in_w);
            int go_base = b * (out_c * out_h * out_w) + oc * (out_h * out_w);

            for (int oh = oh_start; oh <= oh_end; oh++) {
                int ih = oh + kh - 1;
                for (int ow = ow_start; ow <= ow_end; ow++) {
                    int iw = ow + kw - 1;
                    grad_sum += input_cache[in_base + ih * in_w + iw] *
                                grad_output[go_base + oh * out_w + ow];
                }
            }
        }

        d_weights[gid] += grad_sum;
        return;
    }

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

__kernel void conv2d_backward_weights_winograd_3x3(
    __global const float *input_cache,   //[B, in_c, in_h, in_w]
    __global const float *grad_output,   //[B, out_c, out_h, out_w]
    __global float *partial_weights,     //[groups, in_c * 9]
    __global float *partial_bias,        //[groups]
    const int B,
    const int in_c,
    const int out_c,
    const int in_h,
    const int in_w,
    const int out_h,
    const int out_w
) {
    int gid = get_global_id(0);
    int tiles_w = (out_w + 1) / 2;
    int tiles_h = (out_h + 1) / 2;
    int tiles_per_oc = B * tiles_h * tiles_w;
    int total = out_c * tiles_per_oc;
    if (gid >= total) return;

    int tile = gid % tiles_per_oc;
    int oc = gid / tiles_per_oc;
    int b = tile / (tiles_h * tiles_w);
    int rem = tile % (tiles_h * tiles_w);
    int ty = rem / tiles_w;
    int tx = rem % tiles_w;

    int out_spatial = out_h * out_w;
    int go_base = b * (out_c * out_spatial) + oc * out_spatial;
    float bias_sum = 0.0f;

    __global float *out_partial = partial_weights + gid * (in_c * 9);

    for (int ic = 0; ic < in_c; ic++) {
        int in_base = b * (in_c * in_h * in_w) + ic * (in_h * in_w);

        float d[4][4] = {0};
        float e2[2][2] = {0};
        for (int i = 0; i < 4; i++) {
            int oh = ty * 2 + i - 1;
            for (int j = 0; j < 4; j++) {
                int ow = tx * 2 + j - 1;
                if (oh >= 0 && oh < out_h && ow >= 0 && ow < out_w) {
                    d[i][j] = input_cache[in_base + oh * in_w + ow];
                }
            }
        }
        for (int i = 0; i < 2; i++) {
            int oh = ty * 2 + i;
            if (oh < out_h) {
                for (int j = 0; j < 2; j++) {
                    int ow = tx * 2 + j;
                    if (ow < out_w) {
                        float go = grad_output[go_base + oh * out_w + ow];
                        e2[i][j] = go;
                        bias_sum += go;
                    }
                }
            }
        }

        float v[4][4];
        float m[4][4];
        winograd_input_transform(d, v);
        winograd_grad_output_transform(e2, m);

        float dU[4][4];
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                dU[i][j] = v[i][j] * m[i][j];
            }
        }

        const float GT[3][4] = {
            {1.0f, 0.5f, 0.5f, 0.0f},
            {0.0f, 0.5f, -0.5f, 0.0f},
            {0.0f, 0.5f, 0.5f, 1.0f}
        };
        const float G[4][3] = {
            {1.0f, 0.0f, 0.0f},
            {0.5f, 0.5f, 0.5f},
            {0.5f, -0.5f, 0.5f},
            {0.0f, 0.0f, 1.0f}
        };

        float tmp[3][4] = {0};
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 4; j++) {
                float sum = 0.0f;
                for (int k = 0; k < 4; k++) {
                    sum += GT[i][k] * dU[k][j];
                }
                tmp[i][j] = sum;
            }
        }

        float delta[3][3];
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                float sum = 0.0f;
                for (int k = 0; k < 4; k++) {
                    sum += tmp[i][k] * G[k][j];
                }
                delta[i][j] = sum;
            }
        }

        int base = ic * 9;
        out_partial[base + 0] = delta[0][0];
        out_partial[base + 1] = delta[0][1];
        out_partial[base + 2] = delta[0][2];
        out_partial[base + 3] = delta[1][0];
        out_partial[base + 4] = delta[1][1];
        out_partial[base + 5] = delta[1][2];
        out_partial[base + 6] = delta[2][0];
        out_partial[base + 7] = delta[2][1];
        out_partial[base + 8] = delta[2][2];
    }

    partial_bias[gid] = bias_sum;
}

__kernel void conv2d_backward_weights_winograd_reduce(
    __global const float *partial_weights,
    __global const float *partial_bias,
    __global float *d_weights,
    __global float *d_bias,
    const int out_c,
    const int in_c,
    const int tiles_per_oc
) {
    int gid = get_global_id(0);
    int total_weights = out_c * in_c * 9;
    int weight_stride = in_c * 9;

    if (gid < total_weights) {
        int oc = gid / (in_c * 9);
        int rem = gid % (in_c * 9);
        int base = oc * tiles_per_oc * weight_stride + rem;
        float sum = 0.0f;
        for (int t = 0; t < tiles_per_oc; t++) {
            sum += partial_weights[base + t * weight_stride];
        }
        d_weights[gid] += sum;
    }

    if (gid < out_c) {
        float sum = 0.0f;
        for (int t = 0; t < tiles_per_oc; t++) {
            sum += partial_bias[gid * tiles_per_oc + t];
        }
        d_bias[gid] += sum;
    }
}

//tiled weight gradient kernel
//each work-group handles one output-channel tile and one spatial tile
//work-group layout:
//  group_id = oc * pos_tiles + tile
//  local_id  = spatial position within the tile
//local memory layout:
//  lane_weight_partials[local_size * weights_per_oc]
//  lane_bias_partials[local_size]
__kernel void conv2d_backward_weights_tiled(
    __global const float *input_cache,   //[B, in_c, in_h, in_w]
    __global const float *grad_output,   //[B, out_c, out_h, out_w]
    __global float *partial_weights,     //[groups, weights_per_oc]
    __global float *partial_bias,        //[groups]
    __local float *lane_weight_partials,
    __local float *lane_bias_partials,
    const int B,
    const int in_c,
    const int out_c,
    const int in_h,
    const int in_w,
    const int out_h,
    const int out_w,
    const int k,
    const int stride,
    const int padding,
    const int weights_per_oc,
    const int pos_tiles,
    const int total_positions
) {
    int lid = get_local_id(0);
    int lsize = get_local_size(0);
    int gid = get_group_id(0);
    int oc = gid / pos_tiles;
    int tile = gid % pos_tiles;

    __local float *my_weight_partials = lane_weight_partials + lid * weights_per_oc;

    for (int w = 0; w < weights_per_oc; w++) {
        my_weight_partials[w] = 0.0f;
    }
    lane_bias_partials[lid] = 0.0f;
    barrier(CLK_LOCAL_MEM_FENCE);

    int pos_idx = tile * lsize + lid;
    if (oc < out_c && pos_idx < total_positions) {
        int out_spatial = out_h * out_w;
        int b = pos_idx / out_spatial;
        int rem = pos_idx % out_spatial;
        int oh = rem / out_w;
        int ow = rem % out_w;

        int go_base = b * (out_c * out_h * out_w) + oc * out_spatial + oh * out_w + ow;
        float grad = grad_output[go_base];
        lane_bias_partials[lid] = grad;

        int w_base = 0;
        for (int ic = 0; ic < in_c; ic++) {
            int in_base = b * (in_c * in_h * in_w) + ic * (in_h * in_w);
            for (int kh = 0; kh < k; kh++) {
                int ih = oh * stride + kh - padding;
                if (ih >= 0 && ih < in_h) {
                    for (int kw = 0; kw < k; kw++) {
                        int iw = ow * stride + kw - padding;
                        if (iw >= 0 && iw < in_w) {
                            int w_idx = w_base + kh * k + kw;
                            my_weight_partials[w_idx] +=
                                input_cache[in_base + ih * in_w + iw] * grad;
                        }
                    }
                }
            }
            w_base += k * k;
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (lid == 0 && oc < out_c) {
        int group_base = gid * weights_per_oc;
        for (int w = 0; w < weights_per_oc; w++) {
            float sum = 0.0f;
            for (int lane = 0; lane < lsize; lane++) {
                sum += lane_weight_partials[lane * weights_per_oc + w];
            }
            partial_weights[group_base + w] = sum;
        }

        float bias_sum = 0.0f;
        for (int lane = 0; lane < lsize; lane++) {
            bias_sum += lane_bias_partials[lane];
        }
        partial_bias[gid] = bias_sum;
    }
}

//reduce tiled partials into final gradients
__kernel void conv2d_backward_weights_reduce(
    __global const float *partial_weights,
    __global const float *partial_bias,
    __global float *d_weights,
    __global float *d_bias,
    const int out_c,
    const int weights_per_oc,
    const int pos_tiles
) {
    int gid = get_global_id(0);
    int total_weights = out_c * weights_per_oc;

    if (gid < total_weights) {
        int oc = gid / weights_per_oc;
        int w = gid % weights_per_oc;
        float sum = 0.0f;
        int base = oc * pos_tiles * weights_per_oc + w;
        for (int tile = 0; tile < pos_tiles; tile++) {
            sum += partial_weights[base + tile * weights_per_oc];
        }
        d_weights[gid] += sum;
    }

    if (gid < out_c) {
        float sum = 0.0f;
        int base = gid * pos_tiles;
        for (int tile = 0; tile < pos_tiles; tile++) {
            sum += partial_bias[base + tile];
        }
        d_bias[gid] += sum;
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
