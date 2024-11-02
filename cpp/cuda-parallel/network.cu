#include "network.hpp"
#include <cuda_runtime.h>

// CUDA kernel to flatten the weight matrix for each core on the GPU
__global__ void FlattenWeightMatrixKernel(int *weight_list_2D, int *flat_weights, int rows, int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows && col < cols) {
        int index_2D = row * cols + col;
        flat_weights[index_2D] = weight_list_2D[index_2D];
    }
}

// Constructor
Network::Network(int x, int y) : x_size(x), y_size(y) {
    // Allocate memory for the 2D array of Core objects using unified memory
    cudaMallocManaged(&d_RANC_network, x_size * y_size * sizeof(Core));
}

// Destructor to free GPU memory
Network::~Network() {
    cudaFree(d_RANC_network);
}

void Network::initializeNetwork() {
    // This function doesn't need to do anything since we're using unified memory in the constructor
}

void Network::initializeCores(
        const std::vector<std::vector<int>>& topo_2d_vector, 
        const std::vector<std::vector<int>>& weight_2d_vector,
        const std::vector<std::vector<int>>& output_2d_vector
) 
{
    for (size_t idx = 0; idx < topo_2d_vector.size(); ++idx) {
        const std::vector<int>& topo = topo_2d_vector[idx];
        
        int i = topo[0];
        int j = topo[1];
        int param1 = topo[2];
        int param2 = topo[3];

        int index = i * y_size + j;  // Flatten 2D indexing to 1D

        if (param1 != 0) { // not an output bus
            const std::vector<int>& weight = weight_2d_vector[idx];
            const std::vector<int>& output = output_2d_vector[idx];

            // Flatten weight on CPU
            int rows = param2;
            int cols = param1;
            std::vector<int> weights_2D_host(rows * cols);

            // Copy 2D weight matrix to 1D host vector
            for (int r = 0; r < rows; ++r) {
                std::copy(weight.begin() + r * cols, weight.begin() + (r + 1) * cols, weights_2D_host.begin() + r * cols);
            }

            // Allocate and copy weights to GPU memory
            int *d_flat_weights;
            cudaMallocManaged(&d_flat_weights, sizeof(int) * rows * cols);

            // Copy the weights directly to the flattened memory
            cudaMemcpy(d_flat_weights, weights_2D_host.data(), sizeof(int) * rows * cols, cudaMemcpyHostToDevice);

            // Allocate and copy the output data to GPU memory
            int* d_output;
            cudaMallocManaged(&d_output, sizeof(int) * output.size());
            cudaMemcpy(d_output, output.data(), sizeof(int) * output.size(), cudaMemcpyHostToDevice);

            // Initialize the core with the flattened weights and output connections
            new (&d_RANC_network[index]) Core(i, j, param1, param2, d_flat_weights, d_output);

        } else { // is output bus
            new (&d_RANC_network[index]) Core(i, j, param2, param2);
        }
    }
}

void Network::setNextCores(const std::vector<std::vector<int>>& topo_2d_vector) {
    for (const std::vector<int>& topo : topo_2d_vector) {
        int i = topo[0];
        int j = topo[1];

        int index = i * y_size + j;  // Flatten 2D indexing to 1D

        if (i < x_size && j + 1 < y_size) {
            if (topo[2] != 0) {
                int next_core_x = topo[4];
                d_RANC_network[index].NextCore = &d_RANC_network[next_core_x * y_size + (j + 1)];
            } else {
                d_RANC_network[index].NextCore = nullptr;
            }
        }
    }
}

// Print method to output Network details
__host__ void Network::print() const {
    std::cout << "Network size: " << x_size << " x " << y_size << "\n";
    for (int i = 0; i < x_size; ++i) {
        for (int j = 0; j < y_size; ++j) {
            int index = i * y_size + j;  // Flatten 2D indexing to 1D
            d_RANC_network[index].print();
        }
    }
}
