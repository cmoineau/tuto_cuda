# Memory transfert 

In this example we will see how to transfert data (an integer) between the CPU and the GPU.

We will also tackle the handling of CUDA error.

## Malloc

Just as in C, you need to allocate memory for your variable.

This is done by the [cudaMalloc](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1g37d37965bfb4803b6d4e59ff26856356) function : 

```c++
int* dev_a;
cudaMalloc(&dev_a, sizeof(int));
```

Do not forget to free the memory to avoid memory leak ! 

```c++
cudaFree(dev_a);
```

## Send data to the GPU

```c++
// Copy data on GPU
cudaMemcpy(dev_a, &a, sizeof(int), cudaMemcpyHostToDevice);
```


## Retrieve data from the GPU

```c++
// Copy data from GPU
cudaMemcpy(&b, dev_a, sizeof(int),cudaMemcpyDeviceToHost);
```

## Handling error

If you mix a pointer to a CPU data with  apointer to a GPU data, CUDA will raise an error to catch it you must get the return value of the function :

```c++
int cpu_value = 1; // Value initialized on the host
// Trying to copy data in a non allocated memory !
cudaError_t err = cudaMemcpy(&cpu_value, &a, sizeof(int), cudaMemcpyHostToDevice);
if (err != cudaSuccess) {
    printf("\n----\ncudaMemcpy error: %s\n", cudaGetErrorString(err));
    return 1;
}
```

Every function seen here return ``cudaError_t`` values.

> Note : You need to include the header ``#include <cuda_runtime.h>`` to use ``cudaError_t``.

## Compile and expected output

```
nvcc -o mem memory.cu
```

The output should look like this : 

```
Before sending value to GPU : a = 1, b = 0
After sending value to GPU : a = 1, b = 0
After retrieving value from GPU : a = 1, b = 1

----
cudaMemcpy error: invalid argument
```
