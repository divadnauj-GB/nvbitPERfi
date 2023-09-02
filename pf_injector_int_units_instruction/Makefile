NVCC=nvcc -ccbin=`which gcc` -D_FORCE_INLINES

NVCC_VER_REQ=10.1
NVCC_VER=$(shell $(NVCC) --version | grep release | cut -f2 -d, | cut -f3 -d' ')
NVCC_VER_CHECK=$(shell echo "${NVCC_VER} >= $(NVCC_VER_REQ)" | bc)

NVBIT_PATH=../../../core
INCLUDES=-I$(NVBIT_PATH)
INCLUDES+=-I../common

LIBS=-L$(NVBIT_PATH) -lnvbit
NVCC_PATH=-L$(subst bin/nvcc,lib64,$(shell which nvcc | tr -s /))

# No error will be injected if this flag is set. It is a good method to test
# the tool and ensure that no crashes/output corruptions occur with this flag
DUMMY=0

SOURCES=$(wildcard *.cu)
OBJECTS=$(SOURCES:.cu=.o)
ARCH= -gencode arch=compute_70,code=[sm_70,compute_70]
ARCH+= -gencode arch=compute_35,code=[sm_35,compute_35]
ARCH+= -gencode arch=compute_86,code=[sm_86,compute_86]
ARCH+= -gencode arch=compute_53,code=[sm_53,compute_53]

NVBIT_TOOL=pf_injector.so

all: $(NVBIT_TOOL)

$(NVBIT_TOOL): $(OBJECTS) $(NVBIT_PATH)/libnvbit.a
	$(NVCC) $(ARCH) -O3 $(OBJECTS) $(LIBS) $(NVCC_PATH) -lcuda -lcudart_static -shared -o $@

%.o: %.cu
	$(NVCC) -dc -c -std=c++11 $(INCLUDES) -Xptxas -cloning=no -Xcompiler -Wall $(ARCH) -O3 -Xcompiler -fPIC $< -o $@

inject_funcs.o: inject_funcs.cu
	$(NVCC) $(INCLUDES) -maxrregcount=16 -Xptxas -astoolspatch --keep-device-functions $(ARCH) -DDUMMY=$(DUMMY) -Xcompiler -Wall -Xcompiler -fPIC -c $< -o $@
clean:
	rm -f *.so *.o

