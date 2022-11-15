CXX=g++
EXEC=cudaLUD
LOGS=1
BUILDPROFILER=0
ITERATIONS=1
SIZE=2048
DATADIR=.
GOLD=$(DATADIR)/gold_$(SIZE).data
INPUT=$(DATADIR)/input_$(SIZE).data

VPATH=./src/
CUDAPATH=/usr/local/cuda
NVCC=$(CUDAPATH)/bin/nvcc

NVCCFLAGS= -std=c++11 -Xptxas -v
# NVCC optmizations flags
NVCCOPTFLAGS=
CXXFLAGS= -std=c++11 -o3  -fPIC -fopenmp 

# This is what I use, uncomment if you know your arch and want to specify
# Kepler, Volta
SMS= 70 86
$(foreach sm,$(SMS), $(eval ARCH += -gencode arch=compute_$(sm),code=sm_$(sm)))

INCLUDE= -I./src -I$(LOG_HELPER_PATH)/include -I$(CUDAPATH)/samples/common/inc -I../common/include -I$(CUDAPATH)/include

OBJDIR=./obj/
OBJ=  lud_kernel.o  main.o Parameters.o

ifeq ($(DEBUG), 1) 
CXXFLAGS+=-O0 -g
NVCCFLAGS+= -g -G
endif

ifeq ($(LOGS), 1)
CXXFLAGS+= -DLOGS
NVCCFLAGS+= -DLOGS
LDFLAGS+= -L$(LOG_HELPER_PATH)/build  -lLogHelper
endif

ifeq ($(BUILDPROFILER), 1)
CXXFLAGS+= -DBUILDPROFILER
LDFLAGS+= -L$(RADDIR)/src/cuda/common/lib -lNVMLWrapper -L$(CUDAPATH)/lib64/stubs -lnvidia-ml 
endif

LDFLAGS+= -L$(CUDAPATH)/lib64  -lcudart  -lcurand -lcudadevrt 

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
	./$(EXEC) --size $(SIZE) --input $(INPUT) --gold $(GOLD) --iterations 1 --verbose --generate
	
test:
	./$(EXEC) --size $(SIZE) --input $(INPUT) --gold $(GOLD) --iterations $(ITERATIONS) --verbose

golden:
	./$(EXEC) --size $(SIZE) --input $(INPUT) --gold $(GOLD) --iterations 1 --verbose > golden_stdout.txt 2>golden_stderr.txt
	sed -i '/Array set time/c\REPLACED.' golden_stdout.txt 
	sed -i '/Gold check time/c\REPLACED.' golden_stdout.txt 
	sed -i '/OUTPUT/c\REPLACED.' golden_stdout.txt 
	sed -i '/Iteration 0 overall time/c\REPLACED.' golden_stdout.txt 
	sed -i '/Total kernel time/c\REPLACED.' golden_stdout.txt 
	sed -i '/Average kernel time/c\REPLACED.' golden_stdout.txt 
	sed -i '/Device kernel time/c\REPLACED.' golden_stdout.txt 
	sed -i '/Copy time/c\REPLACED.' golden_stdout.txt 

