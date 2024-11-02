#include "network.hpp"

Network::Network(int x, int y) : x_size(x), y_size(y) {
    // Constructor does minimal initialization
}

void Network::initializeNetwork() {
    RANC_network.resize(x_size);
    for (int i = 0; i < x_size; ++i) {
        RANC_network[i].resize(y_size, Core());
    }
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

        if (param1 != 0) { //not output bus
            const std::vector<int>& weight = weight_2d_vector[idx];
            const std::vector<int>& output = output_2d_vector[idx];
            RANC_network[i][j] = Core(i, j, param1, param2, weight, output);
        }
        else { //is output bus
            //Add a 1:1 connection list here
            RANC_network[i][j] = Core(i, j, param2, param2);
        }
        
    }
}

void Network::setNextCores(const std::vector<std::vector<int>>& topo_2d_vector) {
    for (const std::vector<int>& topo : topo_2d_vector) {
        int i = topo[0];
        int j = topo[1];
        

        if (i < x_size && j + 1 < y_size) {
            if (topo[2] != 0) {
                int next_core_x = topo[4];
                RANC_network[i][j].NextCore = &RANC_network[next_core_x][j + 1];
            }
            else RANC_network[i][j].NextCore = nullptr;
        }
        else {
            // Handle error: index out of bounds
        }
    }
}

void Network::print() const {
    std::cout << "Network size: " << x_size << " x " << y_size << "\n";
    for (const std::vector<Core>& row : RANC_network) {
        for (const Core& core : row) {
            core.print();
        }
    }
}
