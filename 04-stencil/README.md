# 1D Stencil usecase

The problem we will solve in this example is the following :

We have a 1D array of N elements.
We want to apply a 1D stecil to this array.
The output element is the sum of of input elements within a radius.

Each thread will process one output element. But each input element will be read multiple time (2.radius+1).


## Shared memory

In a block threads can share data via **shared memory**, however it will be out of scope for other blocks.

To declare a shared data we use the keyword ```__shared__``.

Shared memory need to have a static size and is limited in size.