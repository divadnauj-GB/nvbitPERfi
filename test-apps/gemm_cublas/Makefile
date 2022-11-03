TARGET = gemm

# CUDA
CUDA_PATH=/usr/local/cuda
NVCC = $(CUDA_PATH)/bin/nvcc

SMS = 70
$(foreach sm,$(SMS), $(eval ARCH += -gencode arch=compute_$(sm),code=sm_$(sm)))

INCLUDE_DIR = -I../common/inc

FLAGS = -Xptxas -v $(ARCH) -L$(CUDA_PATH)/lib64 -lcublas  $(INCLUDE_DIR)

all: clean $(TARGET) 

$(TARGET): 
	$(NVCC) -o $(TARGET) $(FLAGS) $(TARGET).cu

test:
	./$(TARGET) float

golden:
	./$(TARGET) float >golden_stdout.txt 2>golden_stderr.txt

clean:
	rm -f *.o *~ $(TARGET)

clobber: clean
	rm -f golden* *_regcount.p stdout* stderr*
