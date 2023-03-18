#include <stdio.h>  /* printf*/
#include <stdlib.h> /* time */
#include <chrono>

__global__ void add_kernel(int *a, int *b, int* result) {
    result[threadIdx.x] = a[threadIdx.x] + b[threadIdx.x];
}

# define N 512


int main() {
    int tensor_size = N * sizeof(int);
    int *a = (int *)malloc(tensor_size);
    int *b = (int *)malloc(tensor_size);
    int *c = (int *)malloc(tensor_size);
    int *dev_a, *dev_b, *dev_c;
    

    for(int i=0; i<N; ++i){
        a[i] = rand()%500;
        b[i] = rand()%500;
    }
    
    // Allocate space for device copied of a, b, c
    cudaMalloc((void**) &dev_a, tensor_size);
    cudaMalloc((void**) &dev_b, tensor_size);
    cudaMalloc((void**) &dev_c, tensor_size);
    

    cudaMemcpy(dev_a, a, tensor_size, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, tensor_size, cudaMemcpyHostToDevice);

    add_kernel<<<1, N>>>(dev_a, dev_b, dev_c);

    cudaMemcpy(c, dev_c, tensor_size, cudaMemcpyDeviceToHost);

    
    for(int i=0; i<N; ++i){
        if(a[i] + b[i] != c[i]){
            printf("The test fail on index : %d !\n", i);
            printf("%d, %d, %d\n", a[i], b[i], c[i]);
            return -1;
        }
    }
    printf("Ok !\n");
    free(a); free(b); free(c);
    cudaFree(dev_a); cudaFree(dev_b); cudaFree(dev_c);

    return 0;
}
