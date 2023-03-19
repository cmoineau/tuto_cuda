# Add Int Kernel

In this example we will see how to create a CUDA kernel that enable parallelism.

## Creating the kernel

In this example we create a kernel to add two numbers multiple time

```c++
__global__ void add_kernel(int *a, int *b, int* result) {
    result[blockIdx.x] = a[blockIdx.x] + b[blockIdx.x];
}
```
The ``blockIdx.x`` variable show where we are in the block.
Each **block** is a **parallel invocation** of the kernel. A **set of block** is a **grid**.

The kernel will be called as follow :

```cpp
add_kernel<<<N, 1>>>(dev_a, dev_b, dev_c);
```

(see ``addTensor_block.cu``)

But instead of running N blocks and 1 thread we could run 1 block and N threads.

To do this we juste need to change ``blockIdx`` with ``threadIdx`` in the kernel :

```cpp
__global__ void add_kernel(int *a, int *b, int* result) {
    result[threadIdx.x] = a[threadIdx.x] + b[threadIdx.x];
}
```

And call the kernel using :

```cpp
add_kernel<<<1, N>>>(dev_a, dev_b, dev_c);
```
(see ``addTensor_thread.cu``)

We can also combine both approaches and use thread and blocks :


```cpp
__global__ void add_kernel(int *a, int *b, int* result) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    result[idx] = a[idx] + b[idx];
}
```

> Note : ``blockDim.x`` give the number of thread per block !

which is called as follow :

```cpp
# define N 2048*2048
# define THREAD_PER_BLOCKS 512
add_kernel<<<N/THREAD_PER_BLOCKS, THREAD_PER_BLOCKS>>>(dev_a, dev_b, dev_c);
```

(see ``addTensor_blockthread.cu``)
