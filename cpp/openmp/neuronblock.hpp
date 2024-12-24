#pragma once
#include <iostream>
#include <vector>

class NeuronBlock {
    //std::vector<int>& neuron_list;
    //std::vector<int> axon_list;
    //std::vector<std::vector<int>> connection_list;
    //std::vector<std::vector<int>> weight_list;

    public:
        int potential;

        NeuronBlock(int potential_init = 0, int neuron_numbers = 256);

        void Leak(int index, int leak_value);
        void Integrate(std::vector<int> axon_list, std::vector<std::vector<int>> weight_list, int neuron_index, int axon_index);
        void Fire(std::vector<int>& neuron_list, int index, int threshold, int reset_value);

        void print() const; // Add this method to print NeuronBlock details
};