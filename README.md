# Box Classifier

a neural network trained from scratch in pure C to classify checkbox images as filled or empty
Features OpenCL GPU acceleration

Built to prove my friend that CNNs are better than standard OMR shit

## Architecture

Standard CNN for binary image classification:

```
Input (64×64 grayscale)
    ↓
Conv2D (32 filters, 3×3) → ReLU → MaxPool (2×2)
    ↓
Conv2D (64 filters, 3×3) → ReLU → MaxPool (2×2)
    ↓
Flatten → Dense (512) → ReLU → Dropout (30%)
    ↓
Dense (1) → Sigmoid → Output (0=empty, 1=filled)
```

## Quick Start

```bash
#Build with GPU support
make gpu

#Train on dataset
./box_classifier

#Inference on single image
./model --img path/to/checkbox.png
```

## Dataset

Training images are loaded from `dataset/wild/` (configured in `main.c`): (I don't include the dataset in the repo)
```
dataset/wild/
├── empty/          #empty checkbox images
├── filled/         #filled checkbox images
└── labels.txt      #image paths and labels
```

You can generate the dataset with the native C generator:

```bash
make wildgen
./generate_wild_dataset --out dataset/wild --empty 10000 --filled 10000
```

It writes compact 32x32 grayscale PGM files, which the loader already accepts and resizes to the model input size.

## Build

```bash
make              #CPU build (OpenMP)
make gpu          #GPU build (OpenCL)
make wildgen      #Build the native dataset generator
```

## Dependencies

- GCC with OpenMP
- OpenCL (optional for GPU)

## License

GPLv3
