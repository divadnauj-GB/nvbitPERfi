NVCC=nvcc -ccbin=`which gcc` -D_FORCE_INLINES

NVCC_VER_REQ=11.3
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
SMS= 70 86
$(foreach sm,$(SMS),$(eval ARCH += -gencode arch=compute_$(sm),code=sm_$(sm)))

NVBIT_TOOL=pf_injector.so

all: $(NVBIT_TOOL)

$(NVBIT_TOOL): $(OBJECTS) $(NVBIT_PATH)/libnvbit.a
	$(NVCC) $(ARCH) -O3 $(OBJECTS) $(LIBS) $(NVCC_PATH) -lcuda -lcudart_static -shared -o $@

%.o: %.cu pf_injector.h
	$(NVCC) -dc -c -std=c++17 $(INCLUDES) -Xptxas -cloning=no -Xcompiler -Wall $(ARCH) -O3 -Xcompiler -fPIC $< -o $@

inject_funcs.o: inject_funcs.cu pf_injector.h
	$(NVCC) $(INCLUDES) -maxrregcount=24 -Xptxas -astoolspatch --keep-device-functions $(ARCH) -DDUMMY=$(DUMMY) -Xcompiler -Wall -Xcompiler -fPIC -c $< -o $@

clean:
	rm -f *.so *.o

