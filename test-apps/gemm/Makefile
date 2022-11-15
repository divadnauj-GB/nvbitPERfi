DEBUG=0
BLOCK_SIZE=32
#for radiation setup
LOGS=1
OMP=1
ALPHA=1.0
BETA=0.0
SIZE?=4096
ITERATIONS=1
PRECISION=float
CUBLAS=0
TENSOR_CORES=0
DMR=none
BUILDRELATIVEERROR=0
CHECKBLOCK=1

DATADIR=.

OUTPUTPREFIX=$(PRECISION)_$(ALPHA)_$(BETA)_$(SIZE)_cublas_$(CUBLAS)_tensor_$(TENSOR_CORES).matrix

# NVCC optmizations flags
NVCCOPTFLAGS=

# This is what I use, uncomment if you know your arch and want to specify
# Kepler, Volta
SMS= 70 86

$(foreach sm,$(SMS), $(eval ARCH += -gencode arch=compute_$(sm),code=sm_$(sm)))

VPATH=./src/
TARGET=gemm
OBJDIR=./obj/
CXX=g++
CUDAPATH=/usr/local/cuda
NVCC=$(CUDAPATH)/bin/nvcc 
LDFLAGS=  -L$(CUDAPATH)/lib64 -lcudart -lcurand -lcublas
COMMON= --std=c++11 
CFLAGS= -O3 -Wall -Wfatal-errors

INCLUDE=-I$(CUDAPATH)/include -I../common  -I$(CUDAPATH)/include/

ifneq ($(BLOCK_SIZE), 32)
COMMON+= -DBLOCK_SIZE=$(BLOCK_SIZE)
endif

ifeq ($(CUBLAS), 1)
USECUBLAS= --use_cublas
endif

ifeq ($(TENSOR_CORES), 1)
USETENSORS= --tensor_cores
endif

ifeq ($(DEBUG), 1) 
CFLAGS+=-O0 -g -DDEBUG
NVCCFLAGS+= -g -G -DDEBUG
endif

ifeq ($(BUILDRELATIVEERROR), 1)
NVCCFLAGS+= -DBUILDRELATIVEERROR=1
endif

ifeq ($(OMP), 1) 
CFLAGS+= -DOMP -fopenmp
NVCCFLAGS+= -Xcompiler "-DOMP -fopenmp"
endif

ifeq ($(LOGS), 1)
INCLUDE+=-I$(LOG_HELPER_PATH)/include/
NVCCFLAGS+= -DLOGS
CFLAGS+= -DLOGS
LDFLAGS+= -L$(LOG_HELPER_PATH)/build -lLogHelper -DLOGS=1
endif

OBJ= main.o setup_gemm.o Parameters.o common.o setup_tensor_cores.o

OBJS = $(addprefix $(OBJDIR), $(OBJ))
DEPS = $(wildcard src/*.h) Makefile

# It is necessary to multi compiler analysis
PASSHOSTCC= -DNVCCOPTFLAGS="$(strip $(NVCCOPTFLAGS))"
NVCCFLAGS+= -Xptxas -v $(NVCCOPTFLAGS) --optimize 3 $(PASSHOSTCC)

all: mkdir $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(COMMON) $(CFLAGS) $^ $(INCLUDE) -o $@  $(LDFLAGS)  

$(OBJDIR)%.o: %.cpp $(DEPS)
	$(CXX) $(COMMON) $(CFLAGS) -c $< -o $@ $(INCLUDE) 

$(OBJDIR)%.o: %.cu $(DEPS)
	$(NVCC) -ccbin $(CXX) $(ARCH) $(COMMON) $(NVCCFLAGS) $(INCLUDE) -c $< -o $@ 

mkdir:
	mkdir -p obj
	mkdir -p $(DATADIR)

.PHONY: clean
clean:
	rm -rf $(OBJS) $(TARGET)

# GEMM options
generate:
	./$(TARGET) --size $(SIZE) --generate --verbose \
			--precision $(PRECISION) --dmr none \
			--iterations 1  \
			--alpha $(ALPHA) --beta $(BETA) \
			--input_a $(DATADIR)/a_$(OUTPUTPREFIX) \
			--input_b $(DATADIR)/b_$(OUTPUTPREFIX) \
			--input_c $(DATADIR)/c_$(OUTPUTPREFIX) \
			--gold  $(DATADIR)/g_$(OUTPUTPREFIX) $(USECUBLAS) $(USETENSORS)
	
test:
	./$(TARGET) --size $(SIZE) --verbose \
			--precision $(PRECISION) --dmr $(DMR) \
			--iterations $(ITERATIONS)  \
			--alpha $(ALPHA) --beta $(BETA) \
			--input_a $(DATADIR)/a_$(OUTPUTPREFIX) \
			--input_b $(DATADIR)/b_$(OUTPUTPREFIX) \
			--input_c $(DATADIR)/c_$(OUTPUTPREFIX) \
			--gold  $(DATADIR)/g_$(OUTPUTPREFIX)	\
			--opnum $(CHECKBLOCK) $(USECUBLAS) $(USETENSORS)
			

golden:
	./$(TARGET) --size $(SIZE) --verbose \
			--precision $(PRECISION) --dmr $(DMR) \
			--iterations $(ITERATIONS)  \
			--alpha $(ALPHA) --beta $(BETA) \
			--input_a $(DATADIR)/a_$(OUTPUTPREFIX) \
			--input_b $(DATADIR)/b_$(OUTPUTPREFIX) \
			--input_c $(DATADIR)/c_$(OUTPUTPREFIX) \
			--gold  $(DATADIR)/g_$(OUTPUTPREFIX)	\
			--opnum $(CHECKBLOCK) $(USECUBLAS) $(USETENSORS) > golden_stdout.txt 2>golden_stderr.txt
	sed -i '/Log file path/c\' golden_stdout.txt
	sed -i '/Time/c\' golden_stdout.txt
	sed -i '/time/c\' golden_stdout.txt
	sed -i '/^$$/d' golden_stdout.txt