#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <cuda_runtime.h>

#include <curand.h>
#include <curand_kernel.h>

#define NUM_SAMPLES_PER_THREAD 10000
#define THREADS_PER_BLOCK 256

__global__ void monte_carlo_pi(int* d_num_points_inside_circle, curandState* d_state) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;

    curand_init(clock64(), index, 0, &d_state[index]);

    int num_points_inside_circle = 0;
    for (int i = 0; i < NUM_SAMPLES_PER_THREAD; i++) {
        float x = curand_uniform(&d_state[index]);
        float y = curand_uniform(&d_state[index]);
        // Checkif random point is in the circle
        if (x * x + y * y <= 1.0f) {
            num_points_inside_circle++;
        }
    }
    // Use of atomicAdd to avoid race condition
    atomicAdd(d_num_points_inside_circle, num_points_inside_circle);
}

__global__ void setup_kernel(curandState* d_state, unsigned long long seed) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    curand_init(seed, tid, 0, &d_state[tid]);
}

int main() {
    int num_streams = 4;
    int block_size = THREADS_PER_BLOCK;
    int num_blocks = 10;

    int* d_result[num_streams];
    int h_result[num_streams];
    cudaStream_t stream[num_streams];
    curandState* d_state[num_streams];
    

    for (int i = 0; i < num_streams; i++) {
        cudaMalloc(&d_state[i], sizeof(curandState) * block_size * num_blocks);
        cudaStreamCreate(&stream[i]);
        cudaMalloc(&d_result[i], sizeof(int));
        cudaMemset(d_result[i], 0, sizeof(int));

        setup_kernel<<<num_blocks, block_size, 0, stream[i]>>>(d_state[i], time(NULL));
        monte_carlo_pi<<<num_blocks, block_size, 0, stream[i]>>>(d_result[i], d_state[i]);
    }

    for (int i = 0; i < num_streams; i++) {
        cudaMemcpyAsync(&h_result[i], d_result[i], sizeof(float), cudaMemcpyDeviceToHost, stream[i]);
        cudaStreamSynchronize(stream[i]);
    }

    int nb_in_circle = 0;
    for (int i = 0; i < num_streams; i++) {
        nb_in_circle += h_result[i];
        printf("Steam %d : %f\n", i, 4.0f * h_result[i] / (num_blocks *block_size* NUM_SAMPLES_PER_THREAD));
        cudaFree(d_result[i]);
        cudaStreamDestroy(stream[i]);
    }

    float pi_estimate = 4.0f * nb_in_circle / (num_streams * num_blocks *block_size* NUM_SAMPLES_PER_THREAD);

    std::cout << "Pi estimate: " << pi_estimate << std::endl;

    return 0;
}
