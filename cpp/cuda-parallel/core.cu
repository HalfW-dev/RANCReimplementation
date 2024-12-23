#include <cuda_runtime.h>
#include "neuronblock.hpp"
#include "core.hpp"  // Ensure the corresponding header file is included

__global__ void NeuronIntegrateKernel(
    int* d_axons, int* d_connections, int* d_potential, int neurons_size, int axons_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < neurons_size * axons_size) {
        int neuron_idx = idx / axons_size;  // Compute neuron index
        int axon_idx = idx % axons_size;   // Compute axon index
        atomicAdd(&d_potential[neuron_idx], 
                  d_axons[axon_idx] * d_connections[neuron_idx * axons_size + axon_idx]);
    }
}

// Constructor with full initialization
__host__ Core::Core(int x, int y, int axons_size, int neurons_size, int* weight, int* output) 
    : x_coordinate(x), y_coordinate(y), axons_size(axons_size), neurons_size(neurons_size), 
      is_used(true), is_output_bus(false), NextCore(nullptr) {

    cudaMallocManaged(&d_axons, axons_size * sizeof(int));
    cudaMallocManaged(&d_queue, axons_size * sizeof(int));
    cudaMallocManaged(&d_neurons, neurons_size * sizeof(int));
    cudaMallocManaged(&d_potential, neurons_size * sizeof(int));
    cudaMallocManaged(&d_connections, neurons_size * axons_size * sizeof(int));
    cudaMallocManaged(&d_core_output, neurons_size * sizeof(int));

    cudaMemcpy(d_connections, weight, neurons_size * axons_size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_core_output, output, neurons_size * sizeof(int), cudaMemcpyHostToDevice);

    cudaMemset(d_potential, 0, neurons_size * sizeof(int));  // Initialize potentials to 0
}

// Constructor for output bus cores
__host__ Core::Core(int x, int y, int axons_size, int neurons_size) 
    : x_coordinate(x), y_coordinate(y), axons_size(axons_size), neurons_size(neurons_size), 
      is_used(true), is_output_bus(true), NextCore(nullptr) {

    cudaMallocManaged(&d_axons, axons_size * sizeof(int));
    cudaMallocManaged(&d_queue, axons_size * sizeof(int));
    cudaMallocManaged(&d_neurons, neurons_size * sizeof(int));
    cudaMallocManaged(&d_potential, neurons_size * sizeof(int));
    cudaMallocManaged(&d_connections, neurons_size * axons_size * sizeof(int));

    // Initialize connections as identity
    for (int i = 0; i < neurons_size; ++i) {
        for (int j = 0; j < axons_size; ++j) {
            d_connections[i * axons_size + j] = (i == j) ? 1 : 0;
        }
    }

    cudaMemset(d_potential, 0, neurons_size * sizeof(int));  // Initialize potentials to 0
}

// Destructor to free allocated memory
__host__ Core::~Core() {
    cudaFree(d_axons);
    cudaFree(d_queue);
    cudaFree(d_neurons);
    cudaFree(d_potential);
    cudaFree(d_connections);
    cudaFree(d_core_output);
}

// NeuronIntegrate method (launches the kernel)
__host__ void Core::NeuronIntegrate(dim3 blocksPerGrid, dim3 threadsPerBlock) {
    NeuronIntegrateKernel<<<blocksPerGrid, threadsPerBlock>>>(
        d_axons, d_connections, d_potential, neurons_size, axons_size);
    cudaDeviceSynchronize();  // Ensure kernel execution completes
}

// NeuronLeak method
__host__ void Core::NeuronLeak(int index, int leak_value) {
    d_potential[index] -= leak_value;  // Subtract the leak value from the potential
}

// NeuronFire method
__host__ void Core::NeuronFire(int* d_neurons, int index, int threshold, int reset_value) {
    d_neurons[index] = (d_potential[index] >= threshold) ? 1 : 0;  // Determine if neuron fires
    if (d_potential[index] >= threshold) {
        d_potential[index] = reset_value;  // Reset potential after firing
    }
}

// loadFromQueue method
__host__ void Core::loadFromQueue() {
    cudaMemcpy(d_axons, d_queue, axons_size * sizeof(int), cudaMemcpyDeviceToDevice);  // Load queue to axons
}

// toNextCore method
__host__ void Core::toNextCore() {
    if (NextCore) {
        for (int i = 0; i < neurons_size; ++i) {
            NextCore->d_queue[d_core_output[i]] = d_neurons[i];  // Pass neuron outputs to next core
        }
    }
}

// print method
__host__ __device__ void Core::print() const {
    printf("Core at (%d, %d) - Used: %d, Output Bus: %d\n", x_coordinate, y_coordinate, is_used, is_output_bus);
}
