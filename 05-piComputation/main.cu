#include <stdio.h>
#include <curand.h>
#include <curand_kernel.h>

#define NUM_SAMPLES_PER_THREAD 10000
#define THREADS_PER_BLOCK 256

__global__ void monte_carlo_pi(int* d_num_points_inside_circle, int* d_num_points_total, curandState* d_state) {
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
    atomicAdd(d_num_points_total, NUM_SAMPLES_PER_THREAD);
}

__global__ void setup_kernel(curandState* d_state, unsigned long long seed) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    curand_init(seed, tid, 0, &d_state[tid]);
}

int main() {
    int num_blocks = 1;
    int num_threads = THREADS_PER_BLOCK;
    int num_samples = NUM_SAMPLES_PER_THREAD * num_threads * num_blocks;

    // Allocate device memory for the random state and results
    curandState* d_state;
    cudaMalloc(&d_state, sizeof(curandState) * num_threads * num_blocks);
    int* d_num_points_inside_circle;
    cudaMalloc(&d_num_points_inside_circle, sizeof(int));
    int* d_num_points_total;
    cudaMalloc(&d_num_points_total, sizeof(int));

    // Initialize the random state for each thread
    setup_kernel<<<num_blocks, num_threads>>>(d_state, time(NULL));

    // Launch the kernel
    monte_carlo_pi<<<num_blocks, num_threads>>>(d_num_points_inside_circle, d_num_points_total, d_state);

    // Copy the results back to the host
    int num_points_inside_circle, num_points_total;
    cudaMemcpy(&num_points_inside_circle, d_num_points_inside_circle, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&num_points_total, d_num_points_total, sizeof(int), cudaMemcpyDeviceToHost);

    // Compute the value of pi
    float pi = 4.0f * num_points_inside_circle / (float)num_points_total;

    // Print the result
    printf("Estimated value of pi: %f\n", pi);

    // Free the device memory
    cudaFree(d_num_points_inside_circle);
    cudaFree(d_num_points_total);
    cudaFree(d_state);

    return 0;
}
