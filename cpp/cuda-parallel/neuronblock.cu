#include "neuronblock.hpp"
#include <cuda_runtime.h>

// Constructor implementation
NeuronBlock::NeuronBlock(int potential_init, int neuron_numbers) {
    potential = potential_init;

    // Allocate memory for potential on the GPU
    cudaMallocManaged(&d_potential, sizeof(int));
    *d_potential = potential_init;
}

// Destructor implementation
NeuronBlock::~NeuronBlock() {
    cudaFree(d_potential);
}

// Leak method implementation
__host__ void NeuronBlock::Leak(int index, int leak_value) {
    potential += leak_value;  // Apply leak to the potential
    *d_potential = potential; // Update the device-side potential
}

// Integrate method implementation
__host__ void NeuronBlock::Integrate(
    int* d_axons, int* d_weights, int neuron_index, int axon_index, int axons_size) 
{
    // Calculate the integration value based on axon input and corresponding weight
    int integration_value = d_axons[axon_index] * d_weights[neuron_index * axons_size + axon_index];
    potential += integration_value; // Update the potential
    *d_potential = potential;       // Update the device-side potential
}

// Fire method implementation
__host__ void NeuronBlock::Fire(int* d_neurons, int index, int threshold, int reset_value) {
    // Determine if the neuron should fire based on the threshold
    d_neurons[index] = (potential >= threshold) ? 1 : 0;
    potential = reset_value; // Reset the potential after firing
    *d_potential = potential; // Update the device-side potential
}

// Print method to output NeuronBlock details
__host__ __device__ void NeuronBlock::print() const {
    printf("NeuronBlock potential: %d\n", potential);
}
