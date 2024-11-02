#pragma once
#include "neuronblock.hpp"
#include <iostream>
#include <vector>

class Core {
public:
    int x_coordinate;
    int y_coordinate;
    Core* NextCore;
    std::vector<int> queue;
    std::vector<int> axons;
    std::vector<int> neurons;
    std::vector<int> core_output;
    std::vector<std::vector<int>> connections;
    NeuronBlock* NB;

    bool is_used;
    bool is_output_bus;

    Core() : x_coordinate(0), y_coordinate(0), NextCore(nullptr), NB(nullptr), is_used(false), is_output_bus(false) {}
    Core(int x, int y, int axons_size, int neurons_size, std::vector<int> weight, std::vector<int> output);
    Core(int x, int y, int axons_size, int neurons_size);

    void NeuronLeak(int index, int leak_value);
    void NeuronIntegrate(
        std::vector<int> axon_list,
        std::vector<std::vector<int>> weight_list,
        int neuron_index,
        int axon_index
    );
    void NeuronFire(std::vector<int>& neuron_list, int index, int threshold, int reset_value);

    void loadFromQueue();
    void toNextCore();
    void print() const; // Add this method to print Core details

};
