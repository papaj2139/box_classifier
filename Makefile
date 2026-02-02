CC = gcc
CFLAGS = -Wall -Wextra -O3 -march=native -ffast-math -fopenmp -Isrc
LDFLAGS = -lm -fopenmp

# OpenCL support (enable with: make USE_OPENCL=1 or make gpu)
ifdef USE_OPENCL
    CFLAGS += -DUSE_OPENCL
    LDFLAGS += -lOpenCL
    SRC_OPENCL = src/opencl_backend.c
endif

# Core library sources
LIB_SRC = src/tensor.c src/layers.c src/nn.c src/data.c $(SRC_OPENCL)
LIB_OBJ = $(LIB_SRC:.c=.o)

# Targets
TRAINER = box_classifier
INFERENCER = model

.PHONY: all clean run gpu infer debug

all: $(TRAINER) $(INFERENCER)

# GPU build (with OpenCL)
gpu:
	$(MAKE) USE_OPENCL=1 all

# Training binary
$(TRAINER): main.o $(LIB_OBJ)
	$(CC) $^ -o $@ $(LDFLAGS)

# Inference binary
$(INFERENCER): infer.o $(LIB_OBJ)
	$(CC) $^ -o $@ $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

# Debug build
debug: CFLAGS = -Wall -Wextra -O0 -g -fsanitize=address -Isrc
debug: LDFLAGS = -lm -fsanitize=address
debug: clean all

# Run training
run: $(TRAINER)
	./$(TRAINER)

# Run inference (usage: make infer IMG=path/to/image.png)
infer: $(INFERENCER)
	./$(INFERENCER) --img $(IMG)

clean:
	rm -f *.o src/*.o $(TRAINER) $(INFERENCER)

# Dependencies
main.o: main.c src/nn.h src/data.h
infer.o: infer.c src/nn.h src/data.h
src/tensor.o: src/tensor.c src/tensor.h
src/layers.o: src/layers.c src/layers.h src/tensor.h
src/nn.o: src/nn.c src/nn.h src/layers.h src/tensor.h
src/data.o: src/data.c src/data.h src/tensor.h src/stb_image.h
src/opencl_backend.o: src/opencl_backend.c src/opencl_backend.h src/tensor.h
