#include "core.hpp"

Core::Core(int x, int y, int axons_size, int neurons_size, std::vector<int> weight, std::vector<int> output)
{
    this->is_used = true;
    this->is_output_bus = false;

    this->x_coordinate = x;
    this->y_coordinate = y;

    std::vector<int> axon_empty(axons_size, 0);
    std::vector<int> queue_empty(axons_size, 0);
    std::vector<int> neuron_empty(neurons_size, 0);

    std::vector<std::vector<int>> connections_buffer(neurons_size, std::vector<int>(axons_size, 0));

    this->axons = axon_empty;
    this->neurons = neuron_empty;
    this->queue = queue_empty;

    for (int i = 0; i < neurons_size; ++i) {
        for (int j = 0; j < axons_size; ++j) {
            connections_buffer[i][j] = weight[i * axons_size + j];
        }
    }

    this->connections = connections_buffer;

    this->core_output = output;

    NB = new NeuronBlock(0, axons.size());

};

Core::Core(int x, int y, int axons_size, int neurons_size) //for output_bus
{
    this->is_used = true;
    this->is_output_bus = true;

    this->x_coordinate = x;
    this->y_coordinate = y;

    std::vector<int> axon_empty (axons_size, 0);
    std::vector<int> queue_empty(axons_size, 0);
    std::vector<int> neuron_empty (neurons_size, 0);

    this->axons = axon_empty;
    this->neurons = neuron_empty;
    this->queue = queue_empty;

    std::vector<std::vector<int>> connections_buffer(neurons_size, std::vector<int>(axons_size, 0));

    for (int i = 0; i < neurons_size; ++i) {
        connections_buffer[i][i] = 1;
    }

    this->connections = connections_buffer;

    NB = new NeuronBlock(0, axons.size());

}
void Core::NeuronLeak(int index, int leak_value)
{
    this->NB->Leak(index, leak_value);
}

void Core::NeuronIntegrate(std::vector<int> axon_list, std::vector<std::vector<int>> weight_list, int neuron_index)
{
    this->NB->Integrate(axon_list, weight_list, neuron_index);
}

void Core::NeuronFire(std::vector<int>& neuron_list, int index, int threshold, int reset_value)
{
    this->NB->Fire(neuron_list, index, threshold, reset_value);
};

void Core::loadFromQueue() {
    this->axons = this->queue;
};

void Core::toNextCore() {
    for (int i = 0; i < this->neurons.size(); i++) {
        this->NextCore->queue[core_output[i]] = this->neurons[i];
    }
};

void Core::print() const {
    //std::cout << "Core at (" << x_coordinate << ", " << y_coordinate << ")\n";
    //std::cout << "Axons size: " << axons.size() << ", Neurons size: " << neurons.size() << "\n";
    /*if (NB) {
        NB->print();
    }*/
    if (NextCore) {
        //std::cout << "NextCore at (" << NextCore->x_coordinate << ", " << NextCore->y_coordinate << ")\n";
    }
    for (const auto& row : connections) {
        for (int val : row) {
            std::cout << val << " ";
        }
        std::cout << std::endl;
    }

    std::cout << std::endl;
}