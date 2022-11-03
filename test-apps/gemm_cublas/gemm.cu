
#include <iostream>
#include <exception>
#include <vector>

#include "cuda_fp16.h"
//#include <cublas.h>
#include <cublas_v2.h>

#include "cuda_utils.h"
#include "device_vector.h"

void call_gemm(size_t dim, const half *v1_dev, const half *v2_dev, half *output_dev, cublasHandle_t blas_handle,
               half &alpha, half &beta) {
    checkCublasErrors(cublasHgemm(blas_handle, CUBLAS_OP_N, CUBLAS_OP_N, dim, dim, dim,
                                  &alpha, v1_dev, dim, v2_dev, dim, &beta, output_dev, dim));
}

void call_gemm(size_t dim, const float *v1_dev, const float *v2_dev, float *output_dev, cublasHandle_t blas_handle,
               float &alpha, float &beta) {
    checkCublasErrors(cublasSgemm(blas_handle, CUBLAS_OP_N, CUBLAS_OP_N, dim, dim, dim,
                                  &alpha, v1_dev, dim, v2_dev, dim, &beta, output_dev, dim));
}

void call_gemm(size_t dim, const double *v1_dev, const double *v2_dev, double *output_dev, cublasHandle_t blas_handle,
               double &alpha, double &beta) {
    checkCublasErrors(cublasDgemm(blas_handle, CUBLAS_OP_N, CUBLAS_OP_N, dim, dim, dim,
                                  &alpha, v1_dev, dim, v2_dev, dim, &beta, output_dev, dim));
}

template<typename real_t>
void perform_gemm(size_t dim, real_t val_v1, real_t val_v2) {
    auto size = dim * dim;
    std::cout << "Size of the array:" << size << std::endl;
    std::vector<real_t> v1(size, val_v1), v2(size, val_v2);
    DeviceVector<real_t> v1_dev = v1;
    DeviceVector<real_t> v2_dev = v2;
    DeviceVector<real_t> output_dev(size, 0);


    cudaDeviceSynchronize();
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        throw std::runtime_error("Error: kernel failed " + std::string(cudaGetErrorString(error)));
    }
    cublasHandle_t blas_handle;
    checkCublasErrors(cublasCreate(&blas_handle));
    checkCublasErrors(cublasSetMathMode(blas_handle, CUBLAS_DEFAULT_MATH));

    real_t alpha = 1.0f, beta = 0.0f;
    call_gemm(dim, v1_dev.data(), v2_dev.data(), output_dev.data(), blas_handle, alpha, beta);

    auto output = output_dev.to_vector();
    auto expected_val = (double) dim;
    for (auto i = 0; i < output.size(); i++) {
        auto out = double(output[i]);
        if (out != expected_val) {
            std::cout << "POS:" << i << " " << out << std::endl;
        }
    }

    std::cout << "Finished computation\n";
}

int main(int argc, char *argv[]) {
    setbuf(stdout, nullptr); // Disable stdout buffering
    if (argc < 2) {
        std::cerr << "You should enter " << argv[0] << " [half|float|double]\n";
    }
    auto precision = std::string(argv[1]);
    auto size = 256;
    if (precision == "half") {
        perform_gemm<half>(size, 1.0f, 1.0f);
    } else if (precision == "float") {
        perform_gemm<float>(size, 1.0f, 1.0f);
    } else if (precision == "double") {
        perform_gemm<double>(size, 1.0f, 1.0f);
    }

    return 0;
}
