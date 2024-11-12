#include "core.hpp"
#include <cuda_runtime.h>
#include <iostream>

// Constructor with CUDA adaptation for Core properties
Core::Core(int x, int y, int axons_size, int neurons_size, int* weight, int* output) 
    : x_coordinate(x), y_coordinate(y), axons_size(axons_size), neurons_size(neurons_size), NextCore(nullptr), is_used(true), is_output_bus(false) {
    
    // Allocate memory for axons, queue, neurons, and connections on the GPU
    cudaMallocManaged(&d_axons, axons_size * sizeof(int));
    cudaMallocManaged(&d_queue, axons_size * sizeof(int));
    cudaMallocManaged(&d_neurons, neurons_size * sizeof(int));
    cudaMallocManaged(&d_core_output, neurons_size * sizeof(int));
    cudaMallocManaged(&d_connections, neurons_size * axons_size * sizeof(int));

    // Copy weight data to device memory (flattened)
    cudaMemcpy(d_connections, weight, neurons_size * axons_size * sizeof(int), cudaMemcpyHostToDevice);

    // Copy output data to device memory
    cudaMemcpy(d_core_output, output, neurons_size * sizeof(int), cudaMemcpyHostToDevice);

    // Create the neuron block
    NB = new NeuronBlock(0, axons_size);
}

// Constructor for output bus
Core::Core(int x, int y, int axons_size, int neurons_size) 
    : x_coordinate(x), y_coordinate(y), axons_size(axons_size), neurons_size(neurons_size), NextCore(nullptr), is_used(true), is_output_bus(true) {
    
    // Allocate memory for axons, queue, neurons on the GPU
    cudaMallocManaged(&d_axons, axons_size * sizeof(int));
    cudaMallocManaged(&d_queue, axons_size * sizeof(int));
    cudaMallocManaged(&d_neurons, neurons_size * sizeof(int));
    
    // Allocate and initialize connections as an identity matrix on the GPU
    cudaMallocManaged(&d_connections, neurons_size * axons_size * sizeof(int));
    for (int i = 0; i < neurons_size; ++i) {
        d_connections[i * axons_size + i] = 1;
    }

    // Create the neuron block
    NB = new NeuronBlock(0, axons_size);
}

// Destructor to free GPU memory
Core::~Core() {
    cudaFree(d_axons);
    cudaFree(d_queue);
    cudaFree(d_neurons);
    cudaFree(d_core_output);
    cudaFree(d_connections);
    delete NB;
}

// Kernel for NeuronIntegrate to be called on the GPU
__global__ void NeuronIntegrateKernel(int* d_axon_list, int* d_weight_list, int* d_neuron_potentials, int axons_size, int neurons_size) {
    int neuron_index = blockIdx.x * blockDim.x + threadIdx.x;
    int axon_index = blockIdx.y * blockDim.y + threadIdx.y;

    if (neuron_index < neurons_size && axon_index < axons_size) {
        int integration_value = d_axon_list[axon_index] * d_weight_list[neuron_index * axons_size + axon_index];
        atomicAdd(&d_neuron_potentials[neuron_index], integration_value);  // Accumulate potential for each neuron
    }
}

// Method to launch NeuronIntegrate kernel
__host__ void Core::NeuronIntegrate(int* d_axon_list, int* d_weight_list) {
    dim3 threadsPerBlock(32, 32);
    dim3 numBlocks((neurons_size + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (axons_size + threadsPerBlock.y - 1) / threadsPerBlock.y);

    int* d_neuron_potentials;
    cudaMallocManaged(&d_neuron_potentials, neurons_size * sizeof(int));
    cudaMemset(d_neuron_potentials, 0, neurons_size * sizeof(int));  // Initialize potentials to zero

    NeuronIntegrateKernel<<<numBlocks, threadsPerBlock>>>(d_axon_list, d_weight_list, d_neuron_potentials, axons_size, neurons_size);
    
    // Wait for GPU to finish
    cudaDeviceSynchronize();

    // Update potentials in the neuron block
    for (int i = 0; i < neurons_size; ++i) {
        NB->potential = d_neuron_potentials[i];
    }

    cudaFree(d_neuron_potentials);
}

// Implementation of NeuronLeak
__host__ void Core::NeuronLeak(int index, int leak_value) {
    NB->Leak(index, leak_value);
}

// Implementation of NeuronFire
__host__ void Core::NeuronFire(int* d_neuron_list, int index, int threshold, int reset_value) {
    d_neuron_list[index] = (NB->potential >= threshold) ? 1 : 0;
    NB->potential = reset_value;
}

// Method to load data from queue
__host__ void Core::loadFromQueue() {
    cudaMemcpy(d_axons, d_queue, axons_size * sizeof(int), cudaMemcpyDeviceToDevice);
}

// Method to transfer data to the next core
__host__ void Core::toNextCore() {
    if (NextCore) {
        for (int i = 0; i < neurons_size; ++i) {
            NextCore->d_queue[d_core_output[i]] = d_neurons[i];
        }
    }
}

// Print method to output Core details
__host__ __device__ void Core::print() const {
    printf("Core at (%d, %d) - Used: %d, Output Bus: %d\n", x_coordinate, y_coordinate, is_used, is_output_bus);
}
