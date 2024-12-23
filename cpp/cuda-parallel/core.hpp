#pragma once
#include "neuronblock.hpp"
#include <cuda_runtime.h>
#include <iostream>

class Core {
public:
    int x_coordinate;
    int y_coordinate;
    Core* NextCore;
    int* d_queue;         // Device pointer for queue
    int* d_axons;         // Device pointer for axons
    int* d_neurons;       // Device pointer for neurons
    int* d_core_output;   // Device pointer for core output
    int* d_connections;   // Device pointer for connections (flattened 2D array)
    int* d_potential;     // Device pointer for neuron potentials

    int queue_size;
    int axons_size;
    int neurons_size;
    int connections_size;

    NeuronBlock* NB;      // Neuron block

    bool is_used;
    bool is_output_bus;

    __host__ Core();  // Default constructor

    // Constructor with CUDA adaptation for Core properties
    __host__ Core(int x, int y, int axons_size, int neurons_size, int* weight, int* output);

    // Constructor for output bus
    __host__ Core(int x, int y, int axons_size, int neurons_size);

    // Destructor to free GPU memory
    __host__ ~Core();

    // Leak method to adjust potential with a leak value using CUDA
    __host__ void NeuronLeak(int index, int leak_value);

    // Integrate method for combining axon inputs and weights into the neuron potential using CUDA
    __host__ void NeuronIntegrate(dim3 blocksPerGrid, dim3 threadsPerBlock);

    // Fire method to determine if a neuron should fire based on potential and threshold using CUDA
    __host__ void NeuronFire(int* d_neuron_list, int index, int threshold, int reset_value);

    // Method to load data from queue using CUDA
    __host__ void loadFromQueue();

    // Method to transfer data to the next core using CUDA
    __host__ void toNextCore();

    // Print method to output Core details
    __host__ __device__ void print() const;
};

// CUDA kernel declaration for neuron integration
__global__ void NeuronIntegrateKernel(
    int* d_axons, int* d_connections, int* d_potential, int neurons_size, int axons_size);
