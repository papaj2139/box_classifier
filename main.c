#include "src/nn.h"
#include "src/data.h"
#ifdef USE_OPENCL
#include "src/opencl_backend.h"
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define IMAGE_SIZE 32
#define LEARNING_RATE 0.001f
#define EPOCHS 10
#define BATCH_SIZE 32

//build the CNN model
NeuralNetwork *build_model(void) {
    NeuralNetwork *nn = nn_create();
    
    //input: [B, 1, 32, 32]
    
    //conv block 1: 1 -> 32 channels
    nn_add_layer(nn, layer_conv2d_create(1, 32, 3, 1, 1)); //-> [B, 32, 32, 32]
    nn_add_layer(nn, layer_relu_create());
    nn_add_layer(nn, layer_maxpool2d_create(2, 2)); //-> [B, 32, 16, 16]
    
    //conv block 2: 32 -> 128 channels
    nn_add_layer(nn, layer_conv2d_create(32, 128, 3, 1, 1)); //-> [B, 128, 16, 16]
    nn_add_layer(nn, layer_relu_create());
    nn_add_layer(nn, layer_maxpool2d_create(2, 2)); //-> [B, 128, 8, 8]
    
    //reduce the spatial dimensions before the classifier head
    nn_add_layer(nn, layer_globalavgpool2d_create()); //-> [B, 128]
    
    //dense classifier head
    nn_add_layer(nn, layer_dense_create(128, 512)); //-> [B, 512]
    nn_add_layer(nn, layer_relu_create());
    nn_add_layer(nn, layer_dropout_create(0.3f));
    
    nn_add_layer(nn, layer_dense_create(512, 1)); //-> [B, 1] logits
    
    return nn;
}

void train_epoch(NeuralNetwork *nn, Dataset *train, TrainingMetrics *metrics) {
    dataset_shuffle(train);
    metrics_reset(metrics);
    
    size_t batch_num = 0;
    size_t total_batches = (train->count + BATCH_SIZE - 1) / BATCH_SIZE;
    
    Tensor *images, *labels;
    size_t actual_batch;
    
    while ((actual_batch = dataset_get_batch(train, BATCH_SIZE, &images, &labels)) > 0) {
        batch_num++;

        //zero gradients
        nn_zero_gradients(nn);
        
        //forward pass with batch
        Tensor *output = nn_forward(nn, images, 1);
        //compute BCE-with-logits loss and gradient directly on logits
        Tensor *grad_reshaped = tensor_create_2d(actual_batch, 1);
        loss_bce_logits_batch_with_metrics(output, labels, grad_reshaped, metrics);
        
        nn_backward(nn, grad_reshaped);

        //update weights (gradients are averaged over batch inside loss function)
        nn_update(nn, LEARNING_RATE);
        
        //cleanup batch tensors
        tensor_destroy(images);
        tensor_destroy(labels);
        tensor_destroy(grad_reshaped);
        
        //progress
        if (batch_num % 10 == 0 || batch_num == total_batches) {
            printf("  Batch %zu/%zu | Loss: %.4f | Acc: %.1f%%\n", 
                   batch_num, total_batches,
                   metrics->loss / (float)metrics->total,
                   metrics->accuracy * 100.0f);
        }
    }
    printf("\n");
}

void evaluate(NeuralNetwork *nn, Dataset *test, TrainingMetrics *metrics) {
    metrics_reset(metrics);
    dataset_reset(test);
    
    Tensor *images, *labels;
    size_t actual_batch;
    
    while ((actual_batch = dataset_get_batch(test, BATCH_SIZE, &images, &labels)) > 0) {
        Tensor *output = nn_forward(nn, images, 0);
        
        loss_bce_logits_batch_with_metrics(output, labels, NULL, metrics);
        
        tensor_destroy(images);
        tensor_destroy(labels);
    }
}

int main(int argc, char **argv) {
    //set seed for reproducibiliy
    srand(1);
    
    const char *data_dir = "dataset/wild";
    const char *labels_file = "dataset/wild/labels.txt";
    const char *model_file = "model.bin";
    
#ifdef USE_OPENCL
    if (opencl_init() != 0) {
        fprintf(stderr, "Failed to initialize OpenCL, falling back to CPU\n");
    }
#endif
    
    int mode_train = 1;
    if (argc > 1 && strcmp(argv[1], "infer") == 0) {
        mode_train = 0;
    }
    
    if (mode_train) {
        printf("=== Box Classifier Training ===\n");
#ifdef USE_OPENCL
        printf("Backend: OpenCL GPU\n");
#else
        printf("Backend: CPU (OpenMP)\n");
#endif
        printf("Batch size: %d | Learning rate: %.4f | Epochs: %d\n\n", 
               BATCH_SIZE, LEARNING_RATE, EPOCHS);
        
        printf("Loading dataset...\n");
        clock_t start = clock();
        Dataset *full = dataset_load(labels_file, data_dir);
        if (!full) {
            fprintf(stderr, "Failed to load dataset\n");
            return 1;
        }
        printf("Loaded in %.2fs\n", (double)(clock() - start) / CLOCKS_PER_SEC);
        
        Dataset *train, *test;
        dataset_split(full, &train, &test, 0.8f);
        dataset_destroy(full);
        
        printf("\nBuilding model...\n");
        NeuralNetwork *nn = build_model();
        nn_print_summary(nn);
        
        TrainingMetrics train_metrics, test_metrics;
        
        for (int epoch = 0; epoch < EPOCHS; epoch++) {
            printf("\n--- Epoch %d/%d ---\n", epoch + 1, EPOCHS);
            
            start = clock();
            train_epoch(nn, train, &train_metrics);
            double epoch_time = (double)(clock() - start) / CLOCKS_PER_SEC;
            
            metrics_print(&train_metrics, "Train");
            
            evaluate(nn, test, &test_metrics);
            metrics_print(&test_metrics, "Test ");
            
            printf("Epoch time: %.2fs (%.1f samples/sec)\n", 
                   epoch_time, train->count / epoch_time);
        }
        
        printf("\nSaving model to %s...\n", model_file);
        if (nn_save(nn, model_file) == 0) {
            printf("Model saved successfully!\n");
        }
        
        nn_destroy(nn);
        dataset_destroy(train);
        dataset_destroy(test);
        
    } else {
        if (argc < 3) {
            printf("Usage: %s infer <image.png>\n", argv[0]);
            return 1;
        }
        
        printf("Loading model from %s...\n", model_file);
        NeuralNetwork *nn = nn_load(model_file);
        if (!nn) {
            fprintf(stderr, "Failed to load model\n");
            return 1;
        }
        
        printf("Loading image: %s\n", argv[2]);
        Tensor *img = load_png_grayscale(argv[2]);
        if (!img) {
            fprintf(stderr, "Failed to load image\n");
            nn_destroy(nn);
            return 1;
        }
        
        //create batch of 1: [1, 1, H, W]
        Tensor *batch = tensor_create_4d(1, 1, img->shape[1], img->shape[2]);
        memcpy(batch->data, img->data, img->size * sizeof(float));
        
        Tensor *output = nn_forward(nn, batch, 0);
#ifdef USE_OPENCL
        tensor_to_cpu(output);
#endif
        float logit = output->data[0];
        float pred = 1.0f / (1.0f + expf(-logit));
        
        printf("\nPrediction: %.4f\n", pred);
        printf("Class: %s (confidence: %.1f%%)\n", 
               pred >= 0.5f ? "FILLED" : "EMPTY",
               pred >= 0.5f ? pred * 100.0f : (1.0f - pred) * 100.0f);
        
        tensor_destroy(img);
        tensor_destroy(batch);
        nn_destroy(nn);
    }
    
    return 0;
}
