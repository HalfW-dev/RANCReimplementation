#include "core.hpp"
#include <cuda_runtime.h>

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

// Implementation of NeuronLeak
__host__ void Core::NeuronLeak(int index, int leak_value) {
    NB->Leak(index, leak_value);
}

// Implementation of NeuronIntegrate
__host__ void Core::NeuronIntegrate(int* d_axon_list, int* d_weight_list, int neuron_index, int axon_index) {
    // Integrate axon and weight data into neuron potential using CUDA
    int integration_value = d_axon_list[axon_index] * d_weight_list[neuron_index * axons_size + axon_index];
    NB->potential += integration_value;
}

// Implementation of NeuronFire
__host__ void Core::NeuronFire(int* d_neuron_list, int index, int threshold, int reset_value) {
    d_neuron_list[index] = (NB->potential >= threshold) ? 1 : 0;
    NB->potential = reset_value;
}

// Method to load data from queue
__host__ void Core::loadFromQueue() {
    // Copy the queue to the axons
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
