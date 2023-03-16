#include <stdio.h>        // printf
#include <cuda_runtime.h> // cudaError_t


int main() {
    int a = 1;
    int* dev_a; // pointer to the device variable
    int b = 0;
    printf("Before sending value to GPU : a = %d, b = %d\n", a,b);

    // Alocate memory on GPU
    cudaMalloc(&dev_a, sizeof(int));

    // Copy data on GPU
    cudaMemcpy(dev_a, &a, sizeof(int), cudaMemcpyHostToDevice);

    printf("After sending value to GPU : a = %d, b = %d\n", a,b);

    // Copy data from GPU
    cudaMemcpy(&b, dev_a, sizeof(int),cudaMemcpyDeviceToHost);
    
    printf("After retrieving value from GPU : a = %d, b = %d\n", a,b);

    cudaFree(dev_a);


    int cpu_value = 1; // Value initialized on the host
    // Trying to copy data in a non allocated memory !
    cudaError_t err = cudaMemcpy(&cpu_value, &a, sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        printf("\n----\ncudaMemcpy error: %s\n", cudaGetErrorString(err));
        return 1;
    }


    return 0;
}
