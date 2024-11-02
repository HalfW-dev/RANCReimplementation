#pragma once
#include "core.hpp"
#include <vector>  // Include for std::vector

class Network {
public:
    int x_size;
    int y_size;
    Core* d_RANC_network;  // Device pointer to a flat array of Core objects

    Network(int x, int y);
    ~Network();  // Destructor to free GPU memory

    void initializeNetwork();

    void initializeCores(const std::vector<std::vector<int>>& topo_2d_vector, 
                         const std::vector<std::vector<int>>& weight_2d_vector, 
                         const std::vector<std::vector<int>>& output_2d_vector);
                         
    void setNextCores(const std::vector<std::vector<int>>& topo_2d_vector);

    __host__ void print() const;  // Host-only print method
};
