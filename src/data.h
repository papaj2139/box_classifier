#ifndef DATA_H
#define DATA_H

#include "tensor.h"
#include <stddef.h>

typedef struct {
    Tensor *image;      //[1, H, W] grayscale
    float label;        //0 = empty 1 = filled
} Sample;

typedef struct {
    Sample *samples;
    size_t count;
    size_t capacity;
    size_t *shuffle_indices;
    size_t current_idx; //for batched iteration
} Dataset;

//load dataset from labels.txt file
Dataset *dataset_load(const char *labels_path, const char *base_dir);
void dataset_destroy(Dataset *ds);

//shuffling and iteration
void dataset_shuffle(Dataset *ds);
void dataset_reset(Dataset *ds);

//batched data loading - returns tensors [B, 1, H, W] and [B]
//returns actual batch size (may be less than requested at end of dataset)
size_t dataset_get_batch(Dataset *ds, size_t batch_size, 
                         Tensor **images_out, Tensor **labels_out);

//train/test split
void dataset_split(Dataset *full, Dataset **train, Dataset **test, float train_ratio);

//load single PNG as tensor [1, H, W]
Tensor *load_png_grayscale(const char *filepath);

#endif
