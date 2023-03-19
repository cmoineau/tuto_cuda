#include <stdio.h>  /* printf*/
#include <cuda_runtime_api.h>

# define N 10
# define NB_BLOCKS 2
# define BLOCK_SIZE 5
# define RADIUS 2

__global__ void stencil(int *in, int *out) {
    printf("[%d][%d] Stencil !\n", blockIdx.x, threadIdx.x);
    // Create a __shared__ variable that store the data and a Halo at each boundary
    // Note : we can not use blockDim.x instead of BLOCK_SIZE as "a variable length array cannot have static storage duration"
    __shared__ int temp[BLOCK_SIZE + 2*RADIUS];
    
    int inputIdx = threadIdx.x + blockIdx.x * blockDim.x;
    int tmpIdx = threadIdx.x + RADIUS;

    // Read input into shared memory
    temp[tmpIdx] = in[inputIdx];
    if (threadIdx.x < RADIUS) {
        if (inputIdx >= RADIUS && inputIdx < N-RADIUS){
            temp[tmpIdx - RADIUS] = in[inputIdx - RADIUS];
            temp[tmpIdx + BLOCK_SIZE] = in[inputIdx + BLOCK_SIZE];
        }
    }

    // Synchronize to make sure every thread had the time to write in temp
    __syncthreads();

    // Compute output
    int result = 0;
    if (inputIdx >= RADIUS && inputIdx < N-RADIUS){
        for (int offset = -RADIUS; offset <= RADIUS; offset ++)
                result += temp[inputIdx + offset];
    }
    printf("[%d][%d] Writing : %d\n", blockIdx.x, threadIdx.x, result);
    out[inputIdx] = result;
}



int main() {
    int tensor_size = N * sizeof(int);
    int *a = (int *)malloc(tensor_size);
    int *b = (int *)malloc(tensor_size);
    int *dev_a, *dev_b;
    
    // Fill a with random values
    for(int i=0; i<N; ++i){
        a[i] = rand()%500;
    }
    
    // Allocate space for device copied of a and b
    cudaMalloc((void**) &dev_a, tensor_size);
    cudaMalloc((void**) &dev_b, tensor_size);
    
    // Send data to GPU
    cudaMemcpy(dev_a, a, tensor_size, cudaMemcpyHostToDevice);
    stencil<<<NB_BLOCKS, BLOCK_SIZE>>>(dev_a, dev_b);

    cudaMemcpy(b, dev_b, tensor_size, cudaMemcpyDeviceToHost);


    printf("[");
    for(int i=0; i<N; ++i){
        printf("%d ", a[i]);
    }
    printf("]\n");
    printf("[");
    for(int i=0; i<N; ++i){
        printf("%d ", b[i]);
    }
    printf("]\n");

    free(a); free(b);
    cudaFree(dev_a); cudaFree(dev_b);

    return 0;
}
