
#include <iostream>
#include <exception>
#include <vector>
#include <random>
#include <fstream>

#include "cuda_fp16.h"
#include <cublas_v2.h>

#include "cuda_utils.h"
#include "device_vector.h"

#define CHAR_CAST(x) (reinterpret_cast<char*>(x))
#define GENERATOR_MAXABSVALUE_GEMM 10
#define GENERATOR_MINABSVALUE_GEMM -GENERATOR_MAXABSVALUE_GEMM

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

template<typename T>
bool write_to_file(std::string &path, std::vector<T> &array) {
    std::ofstream output(path, std::ios::binary);
    if (output.good()) {
        output.write(CHAR_CAST(array.data()), array.size() * sizeof(T));
        return true;
    }
    return false;
}

template<typename real_t>
bool read_from_file(const std::string &file_path, std::vector<real_t> &vector) {
    std::ifstream file(file_path, std::ios::binary);
    if (file.good()) {
        file.read(CHAR_CAST(vector.data()), vector.size() * sizeof(real_t));
        return true;
    }
    return false;
}


template<typename real_t>
void generate_inputs(size_t size, std::vector<real_t> &a_vector,
                     std::vector<real_t> &b_vector, std::string &a_file_path, std::string &b_file_path) {
    std::random_device rd; //Will be used to obtain a seed for the random number engine
    //Standard mersenne_twister_engine seeded with rd()
    std::mt19937 generator((std::mt19937(rd())));

    std::uniform_real_distribution<real_t> dis(GENERATOR_MINABSVALUE_GEMM, GENERATOR_MAXABSVALUE_GEMM);
    for (auto i = 0; i < size; i++) {
        a_vector[i] = real_t(dis(generator));
        b_vector[i] = real_t(dis(generator));
    }
    if (!write_to_file(a_file_path, a_vector) || !write_to_file(b_file_path, b_vector)) {
        std::throw_with_nested(std::runtime_error("Couldn't open " + a_file_path + " or " + b_file_path));
    }
}

template<typename real_t>
void perform_gemm(size_t dim, std::string &a_path, std::string &b_path, std::string &golden_file_path, bool generate) {
    auto size = dim * dim;
    std::cout << "Size of the array:" << size << std::endl;
    std::vector<real_t> v1(size), v2(size), golden_vector(size);
    if (generate) {
        generate_inputs(size, v1, v2, a_path, b_path);
    } else {
        if (!read_from_file(a_path, v1) ||
            !read_from_file(b_path, v2) ||
            !read_from_file(golden_file_path, golden_vector)) {
            std::throw_with_nested(
                    std::runtime_error("Couldn't open " + a_path + ", " + golden_file_path + ", or " + b_path));
        }
    }
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
    if (!generate) {
        for (auto i = 0; i < output.size(); i++) {
            auto out = output[i];
            auto gold = golden_vector[i];
            if (out != gold) {
                std::cout << "diff:" << i << " o:" << out << " g:" << gold << std::endl;
            }
        }
    } else {
        if (!write_to_file(golden_file_path, output)) {
            std::throw_with_nested(std::runtime_error("Couldn't open " + golden_file_path));
        }
    }

    std::cout << "Finished computation\n";
}

int main(int argc, char *argv[]) {
    setbuf(stdout, nullptr); // Disable stdout buffering
    if (argc < 6) {
        std::cerr << "You should enter " << argv[0]
                  << " [half|float|double] <a file path> <b file path> <golden file path> <generate>\n";
    }
    auto precision = std::string(argv[1]);
    auto a_path = std::string(argv[2]);
    auto b_path = std::string(argv[3]);
    auto c_path = std::string(argv[4]);
    auto generate = bool(std::stoi(argv[5]));
    std::cout << "Precision:" << precision << std::endl
              << "A path:" << a_path << std::endl
              << "B path:" << b_path << std::endl
              << "Golden path:" << c_path << std::endl
              << "Generate:" << generate << std::endl;


    auto size = 1024;
//    if (precision == "half") {
//        perform_gemm<half>(size, a_path, b_path, c_path, generate);
//    } else
    if (precision == "float") {
        perform_gemm<float>(size, a_path, b_path, c_path, generate);
    } else if (precision == "double") {
        perform_gemm<double>(size, a_path, b_path, c_path, generate);
    }

    return 0;
}
