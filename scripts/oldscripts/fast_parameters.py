# set parameters for fi
# it is an easier way to set all parameters for SASSIFI, it is the same as setting it on specific_param.py

from os import environ

benchmark = environ['BENCHMARK']
NVBITFI_HOME = environ['NVBITFI_HOME']
THRESHOLD_JOBS = int(environ['FAULTS'])
ADDITIONAL_PARAMETERS = environ['ADDITIONAL_PARAMETERS']

all_apps = {
    'simple_add': [
        NVBITFI_HOME + '/test-apps/simple_add',  # workload directory
        'simple_add',  # binary name
        NVBITFI_HOME + '/test-apps/simple_add/',  # path to the binary file
        2,  # expected runtime
        ADDITIONAL_PARAMETERS  # additional parameters to the run.sh
    ],

    'lava_mp': [
        NVBITFI_HOME + '/test-apps/lava_mp',  # workload directory
        'lava_mp',  # binary name
        NVBITFI_HOME + '/test-apps/lava_mp/',  # path to the binary file
        5,  # expected runtime
        ADDITIONAL_PARAMETERS  # additional parameters to the run.sh
    ],

    'gemm': [
        NVBITFI_HOME + '/test-apps/gemm',  # workload directory
        'gemm',  # binary name
        NVBITFI_HOME + '/test-apps/gemm/',  # path to the binary file
        5,  # expected runtime
        ADDITIONAL_PARAMETERS  # additional parameters to the run.sh
    ],

    'bfs': [
        NVBITFI_HOME + '/test-apps/bfs',  # workload directory
        'cudaBFS',  # binary name
        NVBITFI_HOME + '/test-apps/bfs/',  # path to the binary file
        3,  # expected runtime
        ADDITIONAL_PARAMETERS  # additional parameters to the run.sh
    ],

    'accl': [
        NVBITFI_HOME + '/test-apps/accl',  # workload directory
        'cudaACCL',  # binary name
        NVBITFI_HOME + '/test-apps/accl/',  # path to the binary file
        1,  # expected runtime
        ADDITIONAL_PARAMETERS  # additional parameters to the run.sh
    ],

    'mergesort': [
        NVBITFI_HOME + '/test-apps/mergesort',  # workload directory
        'mergesort',  # binary name
        NVBITFI_HOME + '/test-apps/mergesort/',  # path to the binary file
        5,  # expected runtime
        ADDITIONAL_PARAMETERS  # additional parameters to the run.sh
    ],

    'quicksort': [
        NVBITFI_HOME + '/test-apps/quicksort',  # workload directory
        'quicksort',  # binary name
        NVBITFI_HOME + '/test-apps/quicksort/',  # path to the binary file
        5,  # expected runtime
        ADDITIONAL_PARAMETERS  # additional parameters to the run.sh
    ],

    'hotspot': [
        NVBITFI_HOME + '/test-apps/hotspot',  # workload directory
        'hotspot',  # binary name
        NVBITFI_HOME + '/test-apps/hotspot/',  # path to the binary file
        3,  # expected runtime
        ADDITIONAL_PARAMETERS  # additional parameters to the run.sh
    ],

    'darknet_v2': [
        NVBITFI_HOME + '/test-apps/darknet_v2',  # workload directory
        'darknet_v2',  # binary name
        NVBITFI_HOME + '/test-apps/darknet_v2/',  # path to the binary file
        5,  # expected runtime
        ADDITIONAL_PARAMETERS  # additional parameters to the run.sh
    ],

    'darknet_v3': [
        NVBITFI_HOME + '/test-apps/darknet_v3',  # workload directory
        'darknet_v3_single',  # binary name
        NVBITFI_HOME + '/test-apps/darknet_v3/',  # path to the binary file
        5,  # expected runtime
        ADDITIONAL_PARAMETERS  # additional parameters to the run.sh
    ],

    'darknet_rubens': [
        NVBITFI_HOME + '/test-apps/darknet_rubens',  # workload directory
        'darknet',  # binary name
        NVBITFI_HOME + '/test-apps/darknet_rubens/',  # path to the binary file
        5,  # expected runtime
        ADDITIONAL_PARAMETERS  # additional parameters to the run.sh
    ],

    'gaussian': [
        NVBITFI_HOME + '/test-apps/gaussian',  # workload directory
        'cudaGaussian',  # binary name
        NVBITFI_HOME + '/test-apps/gaussian/',  # path to the binary file
        3,  # expected runtime
        ADDITIONAL_PARAMETERS  # additional parameters to the run.sh
    ],

    'lud': [
        NVBITFI_HOME + '/test-apps/lud',  # workload directory
        'cudaLUD',  # binary name
        NVBITFI_HOME + '/test-apps/lud/',  # path to the binary file
        3,  # expected runtime
        ADDITIONAL_PARAMETERS  # additional parameters to the run.sh
    ],
    'nw': [
        NVBITFI_HOME + '/test-apps/nw',  # workload directory
        'nw',  # binary name
        NVBITFI_HOME + '/test-apps/nw/',  # path to the binary file
        3,  # expected runtime
        ADDITIONAL_PARAMETERS  # additional parameters to the run.sh
    ],
    'cfd': [
        NVBITFI_HOME + '/test-apps/cfd',  # workload directory
        'cfd',  # binary name
        NVBITFI_HOME + '/test-apps/cfd/',  # path to the binary file
        3,  # expected runtime
        ADDITIONAL_PARAMETERS  # additional parameters to the run.sh
    ],
    'trip_hotspot': [
        NVBITFI_HOME + '/test-apps/trip_hotspot',  # workload directory
        'trip_hotspot',  # binary name
        NVBITFI_HOME + '/test-apps/trip_hotspot/',  # path to the binary file
        3,  # expected runtime
        ADDITIONAL_PARAMETERS  # additional parameters to the run.sh
    ],

    'trip_mxm': [
        NVBITFI_HOME + '/test-apps/trip_mxm',  # workload directory
        'trip_mxm',  # binary name
        NVBITFI_HOME + '/test-apps/trip_mxm/',  # path to the binary file
        3,  # expected runtime
        ADDITIONAL_PARAMETERS  # additional parameters to the run.sh
    ],

    'trip_lava': [
        NVBITFI_HOME + '/test-apps/trip_lava',  # workload directory
        'trip_lava',  # binary name
        NVBITFI_HOME + '/test-apps/trip_lava/',  # path to the binary file
        3,  # expected runtime
        ADDITIONAL_PARAMETERS  # additional parameters to the run.sh
    ],

    'darknet_lenet': [
        NVBITFI_HOME + '/test-apps/darknet_lenet',  # workload directory
        'darknet',  # binary name
        NVBITFI_HOME + '/test-apps/darknet_lenet/',  # path to the binary file
        3,  # expected runtime
        ADDITIONAL_PARAMETERS  # additional parameters to the run.sh
    ],

    'py_faster_rcnn': [
        NVBITFI_HOME + '/test-apps/py_faster_rcnn',  # workload directory
        'py_faster_rcnn.py',  # binary name
        '/home/carol/radiation-benchmarks/src/cuda/py-faster-rcnn/',  # path to the binary file
        5,  # expected runtime
        ADDITIONAL_PARAMETERS,
    ],

    'trip_micro': [
        NVBITFI_HOME + '/test-apps/trip_micro',  # workload directory
        # 'cuda_micro-add_single', # binary name
        # 'cuda_micro-mul_single', # binary name
        'cuda_micro-fma_single',  # binary name
        NVBITFI_HOME + '/test-apps/trip_micro/',  # path to the binary file
        20,  # expected runtime
        ADDITIONAL_PARAMETERS  # additional parameters to the run.sh
    ],
}

apps = {benchmark: all_apps[benchmark]}
