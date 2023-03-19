# Pi estimation with a monte carlo method

-atomicAdd : Perform an atomic addition
-atomicCAS: Performs an atomic compare-and-swap operation on a single 32-bit integer or pointer value. It compares the value at the given address with an expected value, and if they are equal, replaces the value with a new value. The return value indicates whether -the swap was successful.
-atomicExch: Performs an atomic exchange operation on a single 32-bit integer or pointer value. It stores the given value at the given address, and returns the old value at that address.
-atomicMin and atomicMax: Performs an atomic minimum or maximum operation on a single 32-bit integer or float value. It compares the value at the given address with a given value, and updates the value at the address with the minimum or maximum of the two values.
-atomicAnd, atomicOr, and atomicXor: Performs an atomic logical operation on a single 32-bit integer value. It applies the given logical operation between the value at the given address and a given value, and updates the value at the address with the result.