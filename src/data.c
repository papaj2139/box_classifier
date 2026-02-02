#include "data.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb_image_resize2.h"

#define TARGET_SIZE 64

//image loading (supports color, any size  and it converts to  64x64 grayscale)
Tensor *load_png_grayscale(const char *filepath) {
    int width, height, channels;
    
    //load image as-is (auto-detect channels)
    unsigned char *data = stbi_load(filepath, &width, &height, &channels, 0);
    
    if (!data) {
        fprintf(stderr, "Failed to load image: %s\n", filepath);
        return NULL;
    }
    
    //convert to grayscale if color
    unsigned char *gray;
    if (channels >= 3) {
        //RGB/RGBA → grayscale using luminance formula
        gray = malloc(width * height);
        for (int i = 0; i < width * height; i++) {
            int r = data[i * channels + 0];
            int g = data[i * channels + 1];
            int b = data[i * channels + 2];
            //Standard luminance: 0.299*R + 0.587*G + 0.114*B
            gray[i] = (unsigned char)(0.299f * r + 0.587f * g + 0.114f * b);
        }
        stbi_image_free(data);
    } else {
        //already grayscale
        gray = data;
    }
    
    //resize to TARGET_SIZE x TARGET_SIZE if needed
    unsigned char *resized;
    int final_w = TARGET_SIZE;
    int final_h = TARGET_SIZE;
    
    if (width != TARGET_SIZE || height != TARGET_SIZE) {
        resized = malloc(TARGET_SIZE * TARGET_SIZE);
        
        //use stb_image_resize2 for resize
        stbir_resize_uint8_linear(gray, width, height, 0,
                                   resized, TARGET_SIZE, TARGET_SIZE, 0,
                                   STBIR_1CHANNEL);
        
        if (gray != data) free(gray);
        else stbi_image_free(gray);
        gray = resized;
    }
    
    //create tensor [1, 64, 64]
    Tensor *t = tensor_create_3d(1, (size_t)final_h, (size_t)final_w);
    
    //normalize to [0, 1]
    for (int y = 0; y < final_h; y++) {
        for (int x = 0; x < final_w; x++) {
            float val = (float)gray[y * final_w + x] / 255.0f;
            tensor_set_3d(t, 0, (size_t)y, (size_t)x, val);
        }
    }
    
    if (gray != data) free(gray);
    else stbi_image_free(gray);
    
    return t;
}

Dataset *dataset_load(const char *labels_path, const char *base_dir) {
    FILE *f = fopen(labels_path, "r");
    if (!f) {
        fprintf(stderr, "Failed to open labels file: %s\n", labels_path);
        return NULL;
    }
    
    Dataset *ds = malloc(sizeof(Dataset));
    ds->count = 0;
    ds->capacity = 1024;
    ds->samples = malloc(ds->capacity * sizeof(Sample));
    ds->shuffle_indices = NULL;
    ds->current_idx = 0;
    
    char line[512];
    while (fgets(line, sizeof(line), f)) {
        char rel_path[256];
        int label;
        
        if (sscanf(line, "%255s %d", rel_path, &label) != 2) {
            continue;
        }
        
        char full_path[512];
        snprintf(full_path, sizeof(full_path), "%s/%s", base_dir, rel_path);
        
        Tensor *img = load_png_grayscale(full_path);
        if (!img) continue;
        
        if (ds->count >= ds->capacity) {
            ds->capacity *= 2;
            ds->samples = realloc(ds->samples, ds->capacity * sizeof(Sample));
        }
        
        ds->samples[ds->count].image = img;
        ds->samples[ds->count].label = (float)label;
        ds->count++;
        
        if (ds->count % 500 == 0) {
            printf("Loaded %zu samples...\n", ds->count);
        }
    }
    
    fclose(f);
    
    ds->shuffle_indices = malloc(ds->count * sizeof(size_t));
    for (size_t i = 0; i < ds->count; i++) {
        ds->shuffle_indices[i] = i;
    }
    
    printf("Dataset loaded: %zu samples\n", ds->count);
    return ds;
}

void dataset_destroy(Dataset *ds) {
    if (!ds) return;
    
    for (size_t i = 0; i < ds->count; i++) {
        tensor_destroy(ds->samples[i].image);
    }
    
    free(ds->samples);
    free(ds->shuffle_indices);
    free(ds);
}

//shuffling and iteration
void dataset_shuffle(Dataset *ds) {
    static int seeded = 0;
    if (!seeded) {
        srand((unsigned int)time(NULL));
        seeded = 1;
    }
    
    for (size_t i = ds->count - 1; i > 0; i--) {
        size_t j = (size_t)rand() % (i + 1);
        size_t tmp = ds->shuffle_indices[i];
        ds->shuffle_indices[i] = ds->shuffle_indices[j];
        ds->shuffle_indices[j] = tmp;
    }
    
    ds->current_idx = 0;
}

void dataset_reset(Dataset *ds) {
    ds->current_idx = 0;
}

//batched data loading
size_t dataset_get_batch(Dataset *ds, size_t batch_size, 
                         Tensor **images_out, Tensor **labels_out) {
    if (ds->current_idx >= ds->count) {
        return 0;
    }
    
    size_t remaining = ds->count - ds->current_idx;
    size_t actual_batch = (batch_size < remaining) ? batch_size : remaining;
    
    //all images are now 64x64
    size_t H = TARGET_SIZE;
    size_t W = TARGET_SIZE;
    
    Tensor *images = tensor_create_4d(actual_batch, 1, H, W);
    Tensor *labels = tensor_create_1d(actual_batch);
    
    for (size_t b = 0; b < actual_batch; b++) {
        size_t idx = ds->shuffle_indices[ds->current_idx + b];
        Sample *s = &ds->samples[idx];
        
        size_t img_offset = b * (1 * H * W);
        memcpy(images->data + img_offset, s->image->data, s->image->size * sizeof(float));
        labels->data[b] = s->label;
    }
    
    ds->current_idx += actual_batch;
    
    *images_out = images;
    *labels_out = labels;
    return actual_batch;
}

//train/test split
void dataset_split(Dataset *full, Dataset **train, Dataset **test, float train_ratio) {
    size_t train_count = (size_t)(full->count * train_ratio);
    size_t test_count = full->count - train_count;
    
    dataset_shuffle(full);
    
    *train = malloc(sizeof(Dataset));
    (*train)->count = train_count;
    (*train)->capacity = train_count;
    (*train)->samples = malloc(train_count * sizeof(Sample));
    (*train)->shuffle_indices = malloc(train_count * sizeof(size_t));
    (*train)->current_idx = 0;
    
    for (size_t i = 0; i < train_count; i++) {
        size_t idx = full->shuffle_indices[i];
        (*train)->samples[i].image = tensor_clone(full->samples[idx].image);
        (*train)->samples[i].label = full->samples[idx].label;
        (*train)->shuffle_indices[i] = i;
    }
    
    *test = malloc(sizeof(Dataset));
    (*test)->count = test_count;
    (*test)->capacity = test_count;
    (*test)->samples = malloc(test_count * sizeof(Sample));
    (*test)->shuffle_indices = malloc(test_count * sizeof(size_t));
    (*test)->current_idx = 0;
    
    for (size_t i = 0; i < test_count; i++) {
        size_t idx = full->shuffle_indices[train_count + i];
        (*test)->samples[i].image = tensor_clone(full->samples[idx].image);
        (*test)->samples[i].label = full->samples[idx].label;
        (*test)->shuffle_indices[i] = i;
    }
    
    printf("Split: %zu train, %zu test\n", train_count, test_count);
}
