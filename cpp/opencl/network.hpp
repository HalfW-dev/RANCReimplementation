#pragma once
#include "core.hpp"
#include <vector>

class Network {
public:
    int x_size;
    int y_size;
    std::vector<std::vector<Core>> RANC_network;

    Network(int x, int y);

    void initializeNetwork();

    void initializeCores(const std::vector<std::vector<int>>& topo_2d_vector, const std::vector<std::vector<int>>& weight_2d_vector, const std::vector<std::vector<int>>& output_2d_vector);
    void setNextCores(const std::vector<std::vector<int>>& topo_2d_vector);

    void print() const;
};
