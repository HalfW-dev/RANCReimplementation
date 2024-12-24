#include "neuronblock.hpp"
#include "core.hpp"
#include "network.hpp"

#include <iostream>
#include <fstream>  // For packet_file stream operations
#include <sstream>  // For string stream operations
#include <vector>   // For using std::vector
#include <string>
#include <chrono>   // For measuring execution time

int main() {
    
    auto start = std::chrono::high_resolution_clock::now(); // Start timer

    int RANC_x;
    int RANC_y;
    int RANC_reset = 0;
    int RANC_leak = 0;
    int RANC_threshold = 0;

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

    int output_bus_x, output_bus_y;
    for (int y = 0; y < RANC->y_size; y++) {
        for (int x = 0; x < RANC->x_size; x++) {
            if (RANC->RANC_network[x][y].is_output_bus) {
                output_bus_x = x;
                output_bus_y = y;
            }
        }
    }

    std::ofstream output_file("./txt/cpp_output.txt");
    if (!output_file.is_open()) {
        std::cerr << "Failed to open the output file: " << std::endl;
        return -1;
    }

    // Variables for timing Neuron functions
    double integrate_total_time = 0;
    double fire_total_time = 0;
    double leak_total_time = 0;
    int integrate_count = 0;
    int fire_count = 0;
    int leak_count = 0;

    // Main processing loop
    auto processing_start = std::chrono::high_resolution_clock::now();
    for (int pak_idx = 0; pak_idx < RANC_packets.size(); pak_idx++) {
        //std::cout << "Processing packet number " << pak_idx + 1 << std::endl;
        std::vector<int> packet = RANC_packets[pak_idx];
        int spk_idx = 0;
        for (int x = 0; x < RANC->x_size; x++) {
            for (int i = 0; i < RANC->RANC_network[x][0].axons.size(); i++) {
                RANC->RANC_network[x][0].queue[i] = packet[spk_idx++];
            }
            RANC->RANC_network[x][0].loadFromQueue();
        }

        for (int y = 0; y < RANC->y_size; y++) {
            for (int x = 0; x < RANC->x_size; x++) {
                if (!RANC->RANC_network[x][y].is_used) continue;

                RANC->RANC_network[x][y].loadFromQueue();
                if (!RANC->RANC_network[x][y].is_output_bus) {
                    #pragma omp_set_dynamic(0);
                    #pragma omp parallel for collapse(1) reduction(+:integrate_total_time, integrate_count) num_threads(9) 
                    for (int neuron_idx = 0; neuron_idx < RANC->RANC_network[x][y].neurons.size(); neuron_idx++) {
                        for (int axon_idx = 0; axon_idx < RANC->RANC_network[x][y].axons.size(); axon_idx++) {
                            auto integrate_start = std::chrono::high_resolution_clock::now();
                            RANC->RANC_network[x][y].NeuronIntegrate(
                                RANC->RANC_network[x][y].axons,
                                RANC->RANC_network[x][y].connections,
                                neuron_idx,
                                axon_idx
                            );
                            auto integrate_end = std::chrono::high_resolution_clock::now();
                            
                            #pragma omp atomic
                            integrate_total_time += std::chrono::duration<double>(integrate_end - integrate_start).count();
                            
                            #pragma omp atomic
                            integrate_count++;
                        }

                        RANC->RANC_network[x][y].NeuronLeak(neuron_idx, RANC_leak);
                        RANC->RANC_network[x][y].NeuronFire(RANC->RANC_network[x][y].neurons, neuron_idx, RANC_threshold, RANC_reset);
                        
                    }
                    RANC->RANC_network[x][y].toNextCore();
                }
                else {
                    for (int i = 0; i < RANC->RANC_network[x][y].neurons.size(); i++) {
                        RANC->RANC_network[output_bus_x][output_bus_y].neurons[i] = RANC->RANC_network[output_bus_x][output_bus_y].axons[i];
                    }
                }
            }
        }

        for (int output : RANC->RANC_network[output_bus_x][output_bus_y].neurons) {
            output_file << output << " ";
        }
        output_file << std::endl;
    }
    auto processing_end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> initialization_duration = initialization_end - initialization_start;
    std::cout << "Initialization time: " << initialization_duration.count() << " seconds" << std::endl;

    std::chrono::duration<double> processing_duration = processing_end - processing_start;
    std::chrono::duration<double> total_duration = processing_end - start;
    std::cout << "Processing time: " << processing_duration.count() << " seconds" << std::endl;
    std::cout << "Total execution time: " << total_duration.count() << " seconds" << std::endl;

    output_file.close();
    delete RANC;
    return 0;
}
