#include "neuronblock.hpp"

// Constructor implementation
NeuronBlock::NeuronBlock(int potential_init, int neuron_numbers) {
    //std::vector<int> vec(neuron_numbers, potential_init);
    this->potential = 0;
}

// Leak method implementation
void NeuronBlock::Leak(int index, int leak_value) {
    this->potential += leak_value; // Leak
    //std::cout << "Applying leak value " << leak_value << std::endl;
}

// Integrate method implementation
void NeuronBlock::Integrate(
    std::vector<int> axon_list,
    std::vector<std::vector<int>> weight_list,
    int neuron_index,
    int axon_index
) 
{
    int integration_value = axon_list[axon_index] * weight_list[neuron_index][axon_index];
    potential += integration_value; // Integrate

    //std::cout << "//////Axon value: " << axon_list[axon_index] << std::endl;
    //std::cout << "//////Weight: " << weight_list[neuron_index][axon_index] << std::endl;
    //std::cout << "//////Integration value: " << integration_value << std::endl;
}

// Fire method implementation
void NeuronBlock::Fire(std::vector<int>& neuron_list, int index, int threshold, int reset_value) {
    neuron_list[index] = (this->potential >= threshold) ? 1 : 0; // Fire
    //std::cout << "Did it fire? " << (this->potential >= threshold) << std::endl;
    this->potential = reset_value;
}

void NeuronBlock::print() const {
    std::cout << "NeuronBlock potential: " << potential << std::endl;
}
