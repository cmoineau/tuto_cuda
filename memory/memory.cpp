#include <stdio.h>        // printf
#include <cuda_runtime.h> // cudaError_t

int main() {
    int a = 1;
    int* dev_a;
    int b = 0;

    printf("Before adding value to GPU : a = %d, c = %d\n", a,b);
    cudaError_t err = cudaMalloc(&dev_a, sizeof(int));
    if (err != cudaSuccess) {
        printf("cudaMalloc error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    err = cudaMemcpy(&a, dev_a, sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        printf("cudaMemcpy error: %s\n", cudaGetErrorString(err));
        return 1;
    }
    printf("Adter adding value to GPU : a = %d, c = %d\n", a,b);

    err = cudaMemcpy(&b, dev_a, sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        printf("cudaMemcpy error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    printf("Adter retrieving value from GPU : a = %d, c = %d\n", a,b);

    cudaFree(dev_a);

    return 0;
}
