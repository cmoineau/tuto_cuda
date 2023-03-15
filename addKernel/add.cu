#include <stdio.h>
#include <cuda_runtime.h>

__global__ void add_kernel(int a, int b, int* result) {
    *result = a + b;
}

int main() {
    int a = 3;
    int b = 5;
    int c;
    int* dev_c;
    
    cudaError_t err = cudaMalloc(&dev_c, sizeof(int));
    if (err != cudaSuccess) {
        printf("cudaMalloc error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    add_kernel<<<1, 1>>>(a, b, dev_c);
    err = cudaDeviceSynchronize(); // Wait for kernel to finish
    if (err != cudaSuccess) {
        printf("cudaDeviceSynchronize error: %s\n", cudaGetErrorString(err));
        return 1;
    }
    err = cudaMemcpy(&c, dev_c, sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        printf("cudaMemcpy error: %s\n", cudaGetErrorString(err));
        return 1;
    }
    printf("%d + %d = %d\n", a, b, c);

    cudaFree(dev_c);

    return 0;
}
