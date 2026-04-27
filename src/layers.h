#ifndef LAYERS_H
#define LAYERS_H

#include "tensor.h"

//layer types
typedef enum {
    LAYER_CONV2D,
    LAYER_MAXPOOL2D,
    LAYER_DENSE,
    LAYER_GLOBALAVGPOOL2D,
    LAYER_FLATTEN,
    LAYER_RELU,
    LAYER_SIGMOID,
    LAYER_DROPOUT,
} LayerType;

//layer structures (all support batched input)
//conv2D:input [B, Cin, H, W] -> output [B, Cout, Hout, Wout]
typedef struct {
    size_t in_channels;
    size_t out_channels;
    size_t kernel_size;
    size_t stride;
    size_t padding;
    
    Tensor *weights;    //[out_channels, in_channels, kH, kW]
    Tensor *bias;       //[out_channels]
    
    Tensor *d_weights;  //gradients
    Tensor *d_bias;
    
    Tensor *m_weights;  //adam moments
    Tensor *v_weights;
    Tensor *m_bias;
    Tensor *v_bias;
    
    Tensor *input_cache;  //[B, Cin, H, W]
} Conv2DLayer;

//maxpool2D: input [B, C, H, W] -> output [B, C, Hout, Wout]
typedef struct {
    size_t pool_size;
    size_t stride;
    
    Tensor *max_indices;  //[B, C, hout, wout]
    size_t batch_size, channels, input_h, input_w;
} MaxPool2DLayer;

//dense: input [B, in_features] -> output [B, out_features]
typedef struct {
    size_t in_features;
    size_t out_features;
    
    Tensor *weights;    //[in_features, out_features]
    Tensor *bias;       //[out_features]
    
    Tensor *d_weights;
    Tensor *d_bias;
    
    Tensor *m_weights;
    Tensor *v_weights;
    Tensor *m_bias;
    Tensor *v_bias;
    
    Tensor *input_cache;  //[B, in_features]
} DenseLayer;

//global average pool: input [B, C, H, W] -> output [B, C]
typedef struct {
    size_t batch_size;
    size_t channels;
    size_t input_h;
    size_t input_w;
} GlobalAvgPool2DLayer;

//flatten: input [B, C, H, W] -> output [B, C*H*W]
typedef struct {
    size_t batch_size;
    size_t original_shape[4];
    size_t original_ndim;
} FlattenLayer;

//relu: element-wise an preserves shape
typedef struct {
    Tensor *mask;
} ReLULayer;

//sigmoid: element-wise, preserves shape
typedef struct {
    Tensor *output_cache;
} SigmoidLayer;

//dropout
typedef struct {
    float p;
    Tensor *mask;
    unsigned int seed;
    int training;
} DropoutLayer;

//generic layer interface
typedef struct Layer {
    LayerType type;
    void *impl;
    
    //all forward/backward take batched tensors
    Tensor *(*forward)(struct Layer *self, Tensor *input, int training);
    Tensor *(*backward)(struct Layer *self, Tensor *grad_output);
    void (*update)(struct Layer *self, float learning_rate, size_t t);
    void (*zero_grad)(struct Layer *self);
    void (*destroy)(struct Layer *self);
} Layer;

//layer constructors
Layer *layer_conv2d_create(size_t in_channels, size_t out_channels, 
                            size_t kernel_size, size_t stride, size_t padding);
Layer *layer_maxpool2d_create(size_t pool_size, size_t stride);
Layer *layer_dense_create(size_t in_features, size_t out_features);
Layer *layer_globalavgpool2d_create(void);
Layer *layer_flatten_create(void);
Layer *layer_relu_create(void);
Layer *layer_sigmoid_create(void);
Layer *layer_dropout_create(float p);

//utility
size_t calc_output_size(size_t input_size, size_t kernel_size, 
                        size_t stride, size_t padding);

#endif
