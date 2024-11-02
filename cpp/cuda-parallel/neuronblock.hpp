#pragma once
#include <cuda_runtime.h>
#include <iostream>

class NeuronBlock {
public:
    int potential;
    int *d_potential;  // Device pointer for potential

    NeuronBlock(int potential_init = 0, int neuron_numbers = 256);
    ~NeuronBlock();  // Destructor to clean up device memory

    // Leak method to adjust potential with a leak value
    __host__ void Leak(int index, int leak_value);

    // Integrate method for combining axon inputs and weights into the neuron potential
    __host__ void Integrate(int* d_axons, int* d_weights, int neuron_index, int axon_index, int axons_size);

    // Fire method to determine if a neuron should fire based on potential and threshold
    __host__ void Fire(int* d_neurons, int index, int threshold, int reset_value);

    // Print method to output NeuronBlock details
    __host__ __device__ void print() const; 
};
