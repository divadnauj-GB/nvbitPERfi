CXX=g++
EXEC=mergesort
LOGS=1
BUILDPROFILER=0
DEBUG=0
#SIZE=134217728
SIZE=1048576
ITERATIONS=1

VPATH=./
CUDAPATH=/usr/local/cuda
NVCC=$(CUDAPATH)/bin/nvcc

NVCCFLAGS= -std=c++11 -Xptxas -v -rdc=true -Xcompiler -fopenmp -Xcompiler -fPIC  -Xcompiler  -pthread
CXXFLAGS= -std=c++11 -O3 -fPIC -pthread -fopenmp

DATADIR=.
INCLUDE= -I./src -I$(LOG_HELPER_PATH)/include -I$(CUDAPATH)/samples/common/inc -I../common/include -I$(CUDAPATH)/include
# Kepler, Volta
SMS= 70 86
$(foreach sm,$(SMS),$(eval ARCH += -gencode arch=compute_$(sm),code=sm_$(sm)))

OBJDIR=./obj/
OBJ= bitonic.o  main.o  mergeSort.o  mergeSort_host.o  mergeSort_validate.o

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
NVCCFLAGS+= -DBUILDPROFILER
LDFLAGS+= -L$(RADDIR)/src/cuda/common/lib -lNVMLWrapper -L$(CUDAPATH)/lib64/stubs -lnvidia-ml 
endif

LDFLAGS+= -L$(CUDAPATH)/lib64  -lcudart  -lcurand

OBJS = $(addprefix $(OBJDIR), $(OBJ))
DEPS = $(wildcard src/*.h) Makefile

# It is necessary to multi compiler analysis
PASSHOSTCC= -DNVCCOPTFLAGS="$(strip $(NVCCOPTFLAGS))"
NVCCFLAGS+= -Xptxas -v $(NVCCOPTFLAGS) --optimize 3 $(PASSHOSTCC)

all: mkdir $(EXEC)


$(EXEC): $(OBJS)  
	$(NVCC) $(ARCH) $^ -o $@ $(LDFLAGS) $(INCLUDE) $(NVCCFLAGS)

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
	 ./$(EXEC) -size=$(SIZE) -generate -verbose \
	 		   -input=$(DATADIR)/input_$(SIZE) \
	 		   -gold=$(DATADIR)/gold_$(SIZE) \
	 		   -iterations=1 
	 		   #-noinputensurance
	 		   	
test:
	 ./$(EXEC) -size=$(SIZE) -verbose \
	 		   -input=$(DATADIR)/input_$(SIZE) \
	 		   -gold=$(DATADIR)/gold_$(SIZE) \
	 		   -iterations=$(ITERATIONS)
	 		   #-noinputensurance

golden:
	  ./$(EXEC) -size=$(SIZE) -verbose \
	 		   -input=$(DATADIR)/input_$(SIZE) \
	 		   -gold=$(DATADIR)/gold_$(SIZE) \
	 		   -iterations=1 > golden_stdout.txt 2>golden_stderr.txt
	sed -i '/Time/c\' golden_stdout.txt 
	sed -i '/time/c\' golden_stdout.txt 
	sed -i '/Perf/c\' golden_stdout.txt
	sed -i '/Starting/c\' golden_stdout.txt
