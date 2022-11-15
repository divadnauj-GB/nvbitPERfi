CXX=g++
EXEC=nw
DEBUG=0
LOGS=1
OMP=1
BUILDPROFILER=0
SIZE=16384
ITERATIONS=1
PENALTY=10
DATADIR=.
INPUT=$(DATADIR)/input_$(SIZE)_$(PENALTY)
GOLD=$(DATADIR)/gold_$(SIZE)_$(PENALTY)

VPATH=./src/
CUDAPATH=/usr/local/cuda
NVCC=$(CUDAPATH)/bin/nvcc

NVCCFLAGS= -std=c++11 -Xptxas -v  -Wno-deprecated-gpu-targets
# NVCC optmizations flags
NVCCOPTFLAGS=
CXXFLAGS= -std=c++11 -o3  -fPIC -fopenmp -Wall -Wfatal-errors  -Wunknown-pragmas -Wunused-function

# This is what I use, uncomment if you know your arch and want to specify
# Kepler, Volta
SMS= 35 53 70 86
$(foreach sm,$(SMS), $(eval ARCH += -gencode arch=compute_$(sm),code=sm_$(sm)))

INCLUDE= -I./src -I$(LOG_HELPER_PATH)/include -I$(CUDAPATH)/samples/common/inc -I../common/include -I$(CUDAPATH)/include

OBJDIR=./obj/
OBJ= needle.o needle_kernel.o

ifeq ($(DEBUG), 1) 
CXXFLAGS+=-O0 -g
NVCCFLAGS+= -g -G
endif

ifeq ($(LOGS), 1)
CXXFLAGS+= -DLOGS
NVCCFLAGS+= -DLOGS
LDFLAGS+= -L$(LOG_HELPER_PATH)/build  -lLogHelper
endif

ifeq ($(OMP), 1) 
CFLAGS+= -DOMP -fopenmp
endif


ifeq ($(BUILDPROFILER), 1)
CXXFLAGS+= -DBUILDPROFILER
LDFLAGS+= -L../common/lib -lNVMLWrapper -L$(CUDAPATH)/lib64/stubs -lnvidia-ml 
endif

LDFLAGS+= -lm -pthread -lstdc++ -L$(CUDAPATH)/lib64  -lcudart  -lcurand -lcudadevrt  -Wno-deprecated-gpu-targets

OBJS = $(addprefix $(OBJDIR), $(OBJ))
DEPS = $(wildcard src/*.h) Makefile


# It is necessary to multi compiler analysis
PASSHOSTCC= -DNVCCOPTFLAGS="$(strip $(NVCCOPTFLAGS))"
NVCCFLAGS+= -Xptxas -v $(NVCCOPTFLAGS) --optimize 3 $(PASSHOSTCC)

all: mkdir $(EXEC)

$(EXEC): $(OBJS)  
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS) $(INCLUDE)

$(OBJDIR)%.o: %.cpp $(DEPS)
	$(CXX) $(CXXFLAGS) -c $< -o $@ $(INCLUDE)
	
$(OBJDIR)%.o: %.cu $(DEPS)
	$(NVCC) $(ARCH) $(NVCCFLAGS) -c $< -o $@ $(INCLUDE) 

mkdir:
	mkdir -p $(OBJDIR)
	mkdir -p $(DATADIR)

clean:
	rm -f $(OBJDIR)* $(EXEC)

generate:
	./$(EXEC) $(SIZE) $(PENALTY) $(INPUT) $(GOLD) 1 1

test:
	./$(EXEC) $(SIZE) $(PENALTY) $(INPUT) $(GOLD) $(ITERATIONS) 0


golden:
	./$(EXEC) $(SIZE) $(PENALTY) $(INPUT) $(GOLD) $(ITERATIONS) 0 > golden_stdout.txt 2>golden_stderr.txt
	sed -i '/kernel time/c\REPLACED.' golden_stdout.txt 
	sed -i '/read../c\REPLACED.' golden_stdout.txt 

