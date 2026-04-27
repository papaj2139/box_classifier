#include "nn.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#ifdef USE_OPENCL
#include "opencl_backend.h"
#endif

static const char *layer_type_name(LayerType type);
static float sigmoid_scalar(float x);

//model lifecycle
NeuralNetwork *nn_create(void) {
    NeuralNetwork *nn = calloc(1, sizeof(NeuralNetwork));
    return nn;
}

void nn_add_layer(NeuralNetwork *nn, Layer *layer) {
    if (nn->num_layers >= MAX_LAYERS) {
        fprintf(stderr, "Error: Maximum layers exceeded\n");
        return;
    }
    nn->layers[nn->num_layers++] = layer;
}

void nn_destroy(NeuralNetwork *nn) {
    if (!nn) return;
    
    for (size_t i = 0; i < nn->num_layers; i++) {
        if (nn->layers[i] && nn->layers[i]->destroy) {
            nn->layers[i]->destroy(nn->layers[i]);
        }
    }
    
    for (size_t i = 0; i <= nn->num_layers; i++) {
        if (nn->activations[i]) {
            tensor_destroy(nn->activations[i]);
        }
    }
    
    free(nn);
}

//batched forward and backward
Tensor *nn_forward(NeuralNetwork *nn, Tensor *input, int training) {
    //clean up old activations
    for (size_t i = 0; i <= nn->num_layers; i++) {
        if (nn->activations[i]) {
            tensor_destroy(nn->activations[i]);
            nn->activations[i] = NULL;
        }
    }
    
    nn->activations[0] = tensor_clone(input);
    Tensor *current = nn->activations[0];
    
    for (size_t i = 0; i < nn->num_layers; i++) {
        Tensor *output = nn->layers[i]->forward(nn->layers[i], current, training);
        nn->activations[i + 1] = output;
        current = output;
    }
    
    return current;
}

void nn_backward(NeuralNetwork *nn, Tensor *grad_loss) {
    Tensor *current_grad = tensor_clone(grad_loss);
    
    for (int i = (int)nn->num_layers - 1; i >= 0; i--) {
        Tensor *next_grad = nn->layers[i]->backward(nn->layers[i], current_grad);
        tensor_destroy(current_grad);
        current_grad = next_grad;
    }
    
    tensor_destroy(current_grad);
}

void nn_update(NeuralNetwork *nn, float learning_rate) {
    nn->t++;
    for (size_t i = 0; i < nn->num_layers; i++) {
        if (nn->layers[i]->update) {
            nn->layers[i]->update(nn->layers[i], learning_rate, nn->t);
        }
    }
}

void nn_zero_gradients(NeuralNetwork *nn) {
    for (size_t i = 0; i < nn->num_layers; i++) {
        if (nn->layers[i]->zero_grad) {
            nn->layers[i]->zero_grad(nn->layers[i]);
        }
    }
}

//batched loss functions
float loss_bce_batch_with_metrics(Tensor *predicted, Tensor *target,
                                  Tensor *grad_out, TrainingMetrics *metrics);
float loss_bce_logits_batch_with_metrics(Tensor *logits, Tensor *target,
                                         Tensor *grad_out, TrainingMetrics *metrics);

float loss_bce_batch(Tensor *predicted, Tensor *target, Tensor *grad_out) {
    return loss_bce_batch_with_metrics(predicted, target, grad_out, NULL);
}

float loss_bce_logits_batch(Tensor *logits, Tensor *target, Tensor *grad_out) {
    return loss_bce_logits_batch_with_metrics(logits, target, grad_out, NULL);
}

float loss_bce_batch_with_metrics(Tensor *predicted, Tensor *target,
                                 Tensor *grad_out, TrainingMetrics *metrics) {
    size_t B = predicted->shape[0];
    float total_loss = 0.0f;
    float scale = 1.0f / (float)B;  //scale gradients to match averaged loss

#ifdef USE_OPENCL
    tensor_to_cpu(predicted);
#endif
    
    for (size_t b = 0; b < B; b++) {
        float p = predicted->data[b];
        float t = target->data[b];
        
        //clamp to avoid log(0)
        p = fmaxf(fminf(p, 1.0f - 1e-6f), 1e-6f);
        
        //bce loss
        float loss = -(t * logf(p) + (1.0f - t) * logf(1.0f - p));
        total_loss += loss;
        
        //gradient: d/dp BCE = (p - t) / (p * (1 - p)), scaled by 1/B
        //note: this cancels with the p*(1-p) in sigmoid backward, resulting in (p-t)/B
        if (grad_out) {
            float denom = p * (1.0f - p);
            if (denom < 1e-6f) denom = 1e-6f; //keep it stable
            grad_out->data[b] = scale * (p - t) / denom;
        }

        if (metrics) {
            int pred_class = p >= 0.5f ? 1 : 0;
            int true_class = t >= 0.5f ? 1 : 0;
            if (pred_class == true_class) {
                metrics->correct++;
            }
        }
    }

    if (metrics) {
        metrics->loss += total_loss;
        metrics->total += B;
        metrics->accuracy = metrics->total ? (float)metrics->correct / (float)metrics->total : 0.0f;
    }

    return total_loss / (float)B;
}

float loss_bce_logits_batch_with_metrics(Tensor *logits, Tensor *target,
                                         Tensor *grad_out, TrainingMetrics *metrics) {
    size_t B = logits->shape[0];
    float total_loss = 0.0f;
    float scale = 1.0f / (float)B;

#ifdef USE_OPENCL
    tensor_to_cpu(logits);
#endif

    for (size_t b = 0; b < B; b++) {
        float z = logits->data[b];
        float t = target->data[b];
        float p = sigmoid_scalar(z);
        float p_clamped = fmaxf(fminf(p, 1.0f - 1e-6f), 1e-6f);
        float loss = -(t * logf(p_clamped) + (1.0f - t) * logf(1.0f - p_clamped));
        total_loss += loss;

        if (grad_out) {
            grad_out->data[b] = scale * (p - t);
        }

        if (metrics) {
            int pred_class = p >= 0.5f ? 1 : 0;
            int true_class = t >= 0.5f ? 1 : 0;
            if (pred_class == true_class) {
                metrics->correct++;
            }
        }
    }

    if (metrics) {
        metrics->loss += total_loss;
        metrics->total += B;
        metrics->accuracy = metrics->total ? (float)metrics->correct / (float)metrics->total : 0.0f;
    }

    return total_loss / (float)B;
}

//metrics
void metrics_reset(TrainingMetrics *m) {
    memset(m, 0, sizeof(TrainingMetrics));
}

void metrics_update_batch(TrainingMetrics *m, Tensor *pred, Tensor *target, float batch_loss) {
    size_t B = pred->shape[0];

#ifdef USE_OPENCL
    tensor_to_cpu(pred);
#endif
    
    m->loss += batch_loss * (float)B;  //accumulate total loss
    m->total += B;
    
    for (size_t b = 0; b < B; b++) {
        int pred_class = pred->data[b] >= 0.5f ? 1 : 0;
        int true_class = target->data[b] >= 0.5f ? 1 : 0;
        if (pred_class == true_class) {
            m->correct++;
        }
    }
    
    m->accuracy = (float)m->correct / (float)m->total;
}

void metrics_print(TrainingMetrics *m, const char *prefix) {
    printf("%s Loss: %.4f | Accuracy: %.2f%% (%zu/%zu)\n",
           prefix, m->loss / (float)m->total, m->accuracy * 100.0f,
           m->correct, m->total);
}

//utility
static const char *layer_type_name(LayerType type) {
    switch (type) {
        case LAYER_CONV2D: return "Conv2D";
        case LAYER_MAXPOOL2D: return "MaxPool2D";
        case LAYER_DENSE: return "Dense";
        case LAYER_GLOBALAVGPOOL2D: return "GlobalAvgPool2D";
        case LAYER_FLATTEN: return "Flatten";
        case LAYER_RELU: return "ReLU";
        case LAYER_SIGMOID: return "Sigmoid";
        case LAYER_DROPOUT: return "Dropout";
        default: return "Unknown";
    }
}

void nn_print_summary(NeuralNetwork *nn) {
    printf("\n=== Model Summary ===\n");
    for (size_t i = 0; i < nn->num_layers; i++) {
        printf("Layer %zu: %s\n", i, layer_type_name(nn->layers[i]->type));
    }
    printf("Total parameters: %zu\n", nn_count_parameters(nn));
}

size_t nn_count_parameters(NeuralNetwork *nn) {
    size_t count = 0;
    for (size_t i = 0; i < nn->num_layers; i++) {
        Layer *l = nn->layers[i];
        if (l->type == LAYER_CONV2D) {
            Conv2DLayer *c = (Conv2DLayer *)l->impl;
            count += c->weights->size + c->bias->size;
        } else if (l->type == LAYER_DENSE) {
            DenseLayer *d = (DenseLayer *)l->impl;
            count += d->weights->size + d->bias->size;
        }
    }
    return count;
}

//save/load
int nn_save(NeuralNetwork *nn, const char *filepath) {
    FILE *f = fopen(filepath, "wb");
    if (!f) return -1;
    
    fwrite(&nn->num_layers, sizeof(size_t), 1, f);
    
    for (size_t i = 0; i < nn->num_layers; i++) {
        Layer *l = nn->layers[i];
        fwrite(&l->type, sizeof(LayerType), 1, f);
        
        if (l->type == LAYER_CONV2D) {
            Conv2DLayer *c = (Conv2DLayer *)l->impl;
#ifdef USE_OPENCL
            tensor_to_cpu(c->weights);
            tensor_to_cpu(c->bias);
#endif
            fwrite(&c->in_channels, sizeof(size_t), 1, f);
            fwrite(&c->out_channels, sizeof(size_t), 1, f);
            fwrite(&c->kernel_size, sizeof(size_t), 1, f);
            fwrite(&c->stride, sizeof(size_t), 1, f);
            fwrite(&c->padding, sizeof(size_t), 1, f);
            fwrite(c->weights->data, sizeof(float), c->weights->size, f);
            fwrite(c->bias->data, sizeof(float), c->bias->size, f);
        } else if (l->type == LAYER_DENSE) {
            DenseLayer *d = (DenseLayer *)l->impl;
#ifdef USE_OPENCL
            tensor_to_cpu(d->weights);
            tensor_to_cpu(d->bias);
#endif
            fwrite(&d->in_features, sizeof(size_t), 1, f);
            fwrite(&d->out_features, sizeof(size_t), 1, f);
            fwrite(d->weights->data, sizeof(float), d->weights->size, f);
            fwrite(d->bias->data, sizeof(float), d->bias->size, f);
        } else if (l->type == LAYER_MAXPOOL2D) {
            MaxPool2DLayer *m = (MaxPool2DLayer *)l->impl;
            fwrite(&m->pool_size, sizeof(size_t), 1, f);
            fwrite(&m->stride, sizeof(size_t), 1, f);
        } else if (l->type == LAYER_GLOBALAVGPOOL2D) {
            // stateless layer
        } else if (l->type == LAYER_DROPOUT) {
            DropoutLayer *d = (DropoutLayer *)l->impl;
            fwrite(&d->p, sizeof(float), 1, f);
        }
    }
    
    fclose(f);
    return 0;
}

NeuralNetwork *nn_load(const char *filepath) {
    FILE *f = fopen(filepath, "rb");
    if (!f) return NULL;
    
    NeuralNetwork *nn = nn_create();
    
    size_t num_layers;
    if (fread(&num_layers, sizeof(size_t), 1, f) != 1) {
        fclose(f);
        nn_destroy(nn);
        return NULL;
    }
    
    for (size_t i = 0; i < num_layers; i++) {
        LayerType type;
        if (fread(&type, sizeof(LayerType), 1, f) != 1) break;
        
        Layer *l = NULL;
        
        if (type == LAYER_CONV2D) {
            size_t in_c, out_c, k, s, p;
            fread(&in_c, sizeof(size_t), 1, f);
            fread(&out_c, sizeof(size_t), 1, f);
            fread(&k, sizeof(size_t), 1, f);
            fread(&s, sizeof(size_t), 1, f);
            fread(&p, sizeof(size_t), 1, f);
            l = layer_conv2d_create(in_c, out_c, k, s, p);
            Conv2DLayer *c = (Conv2DLayer *)l->impl;
            fread(c->weights->data, sizeof(float), c->weights->size, f);
            fread(c->bias->data, sizeof(float), c->bias->size, f);
#ifdef USE_OPENCL
            tensor_to_gpu(c->weights);
            tensor_to_gpu(c->bias);
#endif
        } else if (type == LAYER_DENSE) {
            size_t in_f, out_f;
            fread(&in_f, sizeof(size_t), 1, f);
            fread(&out_f, sizeof(size_t), 1, f);
            l = layer_dense_create(in_f, out_f);
            DenseLayer *d = (DenseLayer *)l->impl;
            fread(d->weights->data, sizeof(float), d->weights->size, f);
            fread(d->bias->data, sizeof(float), d->bias->size, f);
#ifdef USE_OPENCL
            tensor_to_gpu(d->weights);
            tensor_to_gpu(d->bias);
#endif
        } else if (type == LAYER_MAXPOOL2D) {
            size_t ps, s;
            fread(&ps, sizeof(size_t), 1, f);
            fread(&s, sizeof(size_t), 1, f);
            l = layer_maxpool2d_create(ps, s);
        } else if (type == LAYER_GLOBALAVGPOOL2D) {
            l = layer_globalavgpool2d_create();
        } else if (type == LAYER_FLATTEN) {
            l = layer_flatten_create();
        } else if (type == LAYER_RELU) {
            l = layer_relu_create();
        } else if (type == LAYER_SIGMOID) {
            l = layer_sigmoid_create();
        } else if (type == LAYER_DROPOUT) {
            float p;
            fread(&p, sizeof(float), 1, f);
            l = layer_dropout_create(p);
        }
        
        if (l) nn_add_layer(nn, l);
    }
    
    fclose(f);
    return nn;
}

static float sigmoid_scalar(float x) {
    if (x >= 0.0f) {
        float e = expf(-x);
        return 1.0f / (1.0f + e);
    }
    float e = expf(x);
    return e / (1.0f + e);
}
