TARGET = gemm
PRECISION = 32

# CUDA
CUDA_PATH=/usr/local/cuda
NVCC = $(CUDA_PATH)/bin/nvcc

SMS = 70
$(foreach sm,$(SMS), $(eval ARCH += -gencode arch=compute_$(sm),code=sm_$(sm)))

INCLUDE_DIR = -I../include

FLAGS = -Xptxas -v $(ARCH) -DPRECISION=$(PRECISION) -L$(CUDA_PATH)/lib64 -lcublas  $(INCLUDE_DIR)

all: clean $(TARGET) 

$(TARGET): 
	$(NVCC) -o $(TARGET) $(FLAGS) $(TARGET).cu

test:
	./$(TARGET)

golden:
	./$(TARGET) >golden_stdout.txt 2>golden_stderr.txt

clean:
	rm -f *.o *~ $(TARGET)

clobber: clean
	rm -f golden* *_regcount.p stdout* stderr*
