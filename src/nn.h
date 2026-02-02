#ifndef NN_H
#define NN_H

#include "tensor.h"
#include "layers.h"

#define MAX_LAYERS 32

typedef struct {
    Layer *layers[MAX_LAYERS];
    size_t num_layers;
    Tensor *activations[MAX_LAYERS + 1];
} NeuralNetwork;

//model lifecycle
NeuralNetwork *nn_create(void);
void nn_add_layer(NeuralNetwork *nn, Layer *layer);
void nn_destroy(NeuralNetwork *nn);

//batched forward and backward
//input shape: [B, C, H, W] for conv layers and [B, features] for dense
Tensor *nn_forward(NeuralNetwork *nn, Tensor *input, int training);
void nn_backward(NeuralNetwork *nn, Tensor *grad_loss);
void nn_update(NeuralNetwork *nn, float learning_rate);
void nn_zero_gradients(NeuralNetwork *nn);

//utility
void nn_print_summary(NeuralNetwork *nn);
size_t nn_count_parameters(NeuralNetwork *nn);

//loss functions (batched)
//returns mean loss over batch, grad_out is [B, 1]
float loss_bce_batch(Tensor *predicted, Tensor *target, Tensor *grad_out);

//metrics
typedef struct {
    float loss;
    float accuracy;
    size_t correct;
    size_t total;
} TrainingMetrics;

void metrics_reset(TrainingMetrics *m);
void metrics_update_batch(TrainingMetrics *m, Tensor *pred, Tensor *target, float batch_loss);
void metrics_print(TrainingMetrics *m, const char *prefix);

//save/load
int nn_save(NeuralNetwork *nn, const char *filepath);
NeuralNetwork *nn_load(const char *filepath);

#endif
