/*
 * cuda_utils.h
 *
 *  Created on: 17/06/2019
 *      Author: fernando
 */

#ifndef CUDA_UTILS_H_
#define CUDA_UTILS_H_

#include <sys/time.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cublas_v2.h>

#define ERROR_STRING_SIZE 1024

static void _checkFrameworkErrors(cudaError_t error, int line, const char *file) { // NOLINT(bugprone-reserved-identifier)
	if (error == cudaSuccess) {
		return;
	}
	char errorDescription[ERROR_STRING_SIZE];
	snprintf(errorDescription, ERROR_STRING_SIZE, "CUDA Framework error: %s. Error code %d.",
			cudaGetErrorString(error), (int) error);
	printf("%s - Line: %d at %s\n", errorDescription, line, file);
	exit(EXIT_FAILURE);
}

#define checkFrameworkErrors(error) _checkFrameworkErrors(error, __LINE__, __FILE__);

static void _checkCublasErrors(cublasStatus_t error, int line, const char *file) { // NOLINT(bugprone-reserved-identifier)
	if (error == CUBLAS_STATUS_SUCCESS) {
		return;
	}
	char errorDescription[ERROR_STRING_SIZE];
	snprintf(errorDescription, 250, "CUDA CUBLAS error: %d. Bailing.", (error));
	printf("%s - Line: %d at %s\n", errorDescription, line, file);
	exit(EXIT_FAILURE);
}

#define checkCublasErrors(error) _checkCublasErrors(error, __LINE__, __FILE__);

static cudaDeviceProp get_device() {
//================== Retrieve and set the default CUDA device
	cudaDeviceProp prop = cudaDevicePropDontCare;
	checkFrameworkErrors(cudaSetDevice(0));
	checkFrameworkErrors(cudaGetDeviceProperties(&prop, 0));
	return prop;
}

#endif /* CUDA_UTILS_H_ */
