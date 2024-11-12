#include "neuronblock.hpp"
#include "core.hpp"
#include "network.hpp"

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cuda_runtime.h>
#include <chrono>   // For measuring execution time

int main() {
    auto start = std::chrono::high_resolution_clock::now(); // Start timer

    int RANC_x, RANC_y;
    int RANC_reset = 0, RANC_leak = 0, RANC_threshold = 0;
    std::vector<std::vector<int>> RANC_packets;

    // Load the network topology
    std::string networktopo_path = "./txt/network_topo.txt";
    std::ifstream networktopo_file(networktopo_path);
    if (!networktopo_file.is_open()) {
        std::cerr << "Unable to open network topology file: " << networktopo_path << std::endl;
        return 1;
    }

    std::vector<std::vector<int>> topo_line_2d_vector;
    std::string topo_line;
    while (std::getline(networktopo_file, topo_line)) {
        std::stringstream topo_ss(topo_line);
        std::vector<int> topo_line_vector;
        int topo_number;
        while (topo_ss >> topo_number) {
            topo_line_vector.push_back(topo_number);
        }
        topo_line_2d_vector.push_back(topo_line_vector);
    }
    networktopo_file.close();

    // Load core output configuration
    std::string core_output_path = "./txt/core_output.txt";
    std::ifstream core_output_file(core_output_path);
    if (!core_output_file.is_open()) {
        std::cerr << "Unable to open core output file: " << core_output_path << std::endl;
        return 1;
    }

    std::vector<std::vector<int>> output_2d_vector;
    std::string core_output_line;
    while (std::getline(core_output_file, core_output_line)) {
        std::vector<int> line_vector;
        std::stringstream ss(core_output_line);
        int number;
        while (ss >> number) {
            line_vector.push_back(number);
        }
        output_2d_vector.push_back(line_vector);
    }
    core_output_file.close();

    // Load configuration
    std::string config_path = "./txt/config.txt";
    std::ifstream config_file(config_path);
    if (!config_file.is_open()) {
        std::cerr << "Unable to open config file: " << config_path << std::endl;
        return 1;
    }
    std::string config_line;
    if (std::getline(config_file, config_line)) RANC_x = std::stoi(config_line);
    if (std::getline(config_file, config_line)) RANC_y = std::stoi(config_line);
    config_file.close();

    // Load packet data
    std::string packet_path = "./txt/packets.txt";
    std::ifstream packet_file(packet_path);
    if (!packet_file.is_open()) {
        std::cerr << "Unable to open packet file: " << packet_path << std::endl;
        return 1;
    }

    std::string packet_line;
    while (std::getline(packet_file, packet_line)) {
        std::vector<int> line_vector;
        std::stringstream ss(packet_line);
        int number;
        while (ss >> number) {
            line_vector.push_back(number);
        }
        RANC_packets.push_back(line_vector);
    }
    packet_file.close();

    // Load weights
    std::string csram_path = "./txt/csram.txt";
    std::ifstream csram_file(csram_path);
    if (!csram_file.is_open()) {
        std::cerr << "Unable to open weight file: " << csram_path << std::endl;
        return 1;
    }

    std::vector<std::vector<int>> weight_line_2d_vector;
    std::string weight_line;
    while (std::getline(csram_file, weight_line)) {
        std::vector<int> weight_line_vector;
        std::stringstream weight_ss(weight_line);
        int weight_number;
        while (weight_ss >> weight_number) {
            weight_line_vector.push_back(weight_number);
        }
        weight_line_2d_vector.push_back(weight_line_vector);
    }
    csram_file.close();

    // Initialize the network
    auto initialization_start = std::chrono::high_resolution_clock::now();
    Network* RANC = new Network(RANC_x, RANC_y);
    RANC->initializeNetwork();
    RANC->initializeCores(topo_line_2d_vector, weight_line_2d_vector, output_2d_vector);
    RANC->setNextCores(topo_line_2d_vector);
    auto initialization_end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> initialization_duration = initialization_end - initialization_start;
    std::cout << "Initialization time: " << initialization_duration.count() << " seconds" << std::endl;

    int output_bus_x, output_bus_y;
    for (int y = 0; y < RANC->y_size; y++) {
        for (int x = 0; x < RANC->x_size; x++) {
            int index = x * RANC->y_size + y;  // Linear indexing
            if (RANC->d_RANC_network[index].is_output_bus) {
                output_bus_x = x;
                output_bus_y = y;
            }
        }
    }

    std::ofstream output_file("./output/cpp_output.txt");
    if (!output_file.is_open()) {
        std::cerr << "Failed to open the output file: " << std::endl;
        return -1;
    }

    // Main processing loop
    auto processing_start = std::chrono::high_resolution_clock::now();
    for (int pak_idx = 0; pak_idx < RANC_packets.size(); pak_idx++) {
        std::cout << "Packet number " << pak_idx + 1 << std::endl;
        std::vector<int> packet = RANC_packets[pak_idx];
        int spk_idx = 0;

        // Load each packet into the network
        for (int x = 0; x < RANC->x_size; x++) {
            int index = x * RANC->y_size;
            for (int i = 0; i < RANC->d_RANC_network[index].axons_size; i++) {
                RANC->d_RANC_network[index].d_queue[i] = packet[spk_idx++];
            }
            RANC->d_RANC_network[index].loadFromQueue();
        }

        // Process each core in the network
        for (int y = 0; y < RANC->y_size; y++) {
            for (int x = 0; x < RANC->x_size; x++) {
                int index = x * RANC->y_size + y;
                if (!RANC->d_RANC_network[index].is_used) continue;

                RANC->d_RANC_network[index].loadFromQueue();

                if (!RANC->d_RANC_network[index].is_output_bus) {
                    // Launch NeuronIntegrate in parallel for the entire core
                    RANC->d_RANC_network[index].NeuronIntegrate(
                        RANC->d_RANC_network[index].d_axons,
                        RANC->d_RANC_network[index].d_connections
                    );

                    // Apply leak and fire operations sequentially
                    for (int neuron_idx = 0; neuron_idx < RANC->d_RANC_network[index].neurons_size; neuron_idx++) {
                        RANC->d_RANC_network[index].NeuronLeak(neuron_idx, RANC_leak);
                        RANC->d_RANC_network[index].NeuronFire(RANC->d_RANC_network[index].d_neurons, neuron_idx, RANC_threshold, RANC_reset);
                    }

                    // Transfer data to the next core
                    RANC->d_RANC_network[index].toNextCore();
                } else {
                    // Handle output bus core by copying axons to neurons
                    for (int i = 0; i < RANC->d_RANC_network[index].neurons_size; i++) {
                        int output_index = output_bus_x * RANC->y_size + output_bus_y;
                        RANC->d_RANC_network[output_index].d_neurons[i] = RANC->d_RANC_network[output_index].d_axons[i];
                    }
                }
            }
        }

        // Output results
        int output_index = output_bus_x * RANC->y_size + output_bus_y;
        for (int i = 0; i < RANC->d_RANC_network[output_index].neurons_size; i++) {
            std::cout << RANC->d_RANC_network[output_index].d_neurons[i] << " ";
        }
        std::cout << std::endl;

        for (int i = 0; i < RANC->d_RANC_network[output_index].neurons_size; i++) {
            output_file << RANC->d_RANC_network[output_index].d_neurons[i] << " ";
        }
        output_file << std::endl;
    }
    auto processing_end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> processing_duration = processing_end - processing_start;
    std::chrono::duration<double> total_duration = processing_end - start;
    std::cout << "Processing time: " << processing_duration.count() << " seconds" << std::endl;
    std::cout << "Total execution time: " << total_duration.count() << " seconds" << std::endl;

    cudaFree(RANC->d_RANC_network);
    output_file.close();
    delete RANC;
    return 0;
}
