#include <stdio.h>

__global__ void add_kernel(int *a, int *b, int* result) {
    *result = *a + *b;
}

int main() {
    int a = 3;
    int b = 5;
    int c;
    int *dev_a, *dev_b, *dev_c;

    // Allocate space for device copied of a, b, c
    cudaMalloc((void**) &dev_a, sizeof(int));
    cudaMalloc((void**) &dev_b, sizeof(int));
    cudaMalloc((void**) &dev_c, sizeof(int));
    

    cudaMemcpy(dev_a, &a, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, &b, sizeof(int), cudaMemcpyHostToDevice);


    add_kernel<<<1, 1>>>(dev_a, dev_b, dev_c);
    
    cudaMemcpy(&c, dev_c, sizeof(int), cudaMemcpyDeviceToHost);

    printf("%d + %d = %d\n", a, b, c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);

    return 0;
}
