#include "src/nn.h"
#include "src/data.h"
#ifdef USE_OPENCL
#include "src/opencl_backend.h"
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MODEL_FILE "model.bin"

void print_usage(const char *prog) {
    printf("Box Classifier Inference\n");
    printf("Usage: %s --img <image.png>\n", prog);
    printf("       %s --help\n", prog);
    printf("\nClassifies whether a box in the image is filled or empty.\n");
    printf("Expects a 64x64 grayscale PNG image.\n");
}

int main(int argc, char **argv) {
    const char *image_path = NULL;
    
    //parse args
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--img") == 0 && i + 1 < argc) {
            image_path = argv[++i];
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            print_usage(argv[0]);
            return 0;
        }
    }
    
    if (!image_path) {
        print_usage(argv[0]);
        return 1;
    }
    
#ifdef USE_OPENCL
    if (opencl_init() != 0) {
        fprintf(stderr, "OpenCL init failed, using CPU\n");
    }
#endif
    
    //load model
    printf("Loading model from %s...\n", MODEL_FILE);
    NeuralNetwork *nn = nn_load(MODEL_FILE);
    if (!nn) {
        fprintf(stderr, "Error: Failed to load model. Train first with ./box_classifier\n");
        return 1;
    }
    printf("Model loaded (%zu parameters)\n\n", nn_count_parameters(nn));
    
    //load image
    printf("Loading image: %s\n", image_path);
    Tensor *img = load_png_grayscale(image_path);
    if (!img) {
        fprintf(stderr, "Error: Failed to load image\n");
        nn_destroy(nn);
        return 1;
    }
    
    //check dimensions
    if (img->shape[1] != 64 || img->shape[2] != 64) {
        fprintf(stderr, "Warning: Image is %zux%zu, expected 64x64\n",
                img->shape[2], img->shape[1]);
    }
    
    //create batch of 1: [1, 1, H, W]
    Tensor *batch = tensor_create_4d(1, 1, img->shape[1], img->shape[2]);
    memcpy(batch->data, img->data, img->size * sizeof(float));
    
    //run inference
    printf("Running inference...\n\n");
    Tensor *output = nn_forward(nn, batch, 0);
    float pred = output->data[0];
    
    //print result
    printf("  Result: %s\n", pred >= 0.5f ? "✓ FILLED" : "☐ EMPTY");
    printf("  Confidence: %.1f%%\n", 
           pred >= 0.5f ? pred * 100.0f : (1.0f - pred) * 100.0f);
    printf("  Raw score: %.4f\n", pred);
    
    //cleanup
    tensor_destroy(img);
    tensor_destroy(batch);
    nn_destroy(nn);
    
#ifdef USE_OPENCL
    opencl_cleanup();
#endif
    
    return 0;
}
