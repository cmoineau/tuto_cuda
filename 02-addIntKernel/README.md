# Add Int Kernel

In this example we will see how to create a CUDA kernel and call it.
For the sake of the example we will create a simple kernel that add 2 integers.

## Creating a kernel

In this example we create a kernel to add two numbers :

```c++
__global__ void add_kernel(int a, int b, int* result) {
    *result = a + b;
}
```

Different qualificatif possible :

- `__global__` : run on GPUcalled by CPU ;
- `__device__` : run and called by GPU ;
- `__host__` : run and called by CPU.

## Calling a kernel
Calling the kernel is done as follow : 

```cpp
add_kernel<<<1, 1>>>(dev_a, dev_b, dev_c);
```

The synthax <<<X, Y>>>, where X is the number of blocks and Y the number of threads.

More on that on an other example.

## Compile and expected output

```
nvcc -o add add.cu
```

The output should look like this : 

```
3 + 5 = 8
```
