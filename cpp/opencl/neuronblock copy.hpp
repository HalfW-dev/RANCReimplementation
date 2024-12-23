#pragma once

#include <CL/cl.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>

class NeuronBlock {
    //std::vector<int>& neuron_list;
    //std::vector<int> axon_list;
    //std::vector<std::vector<int>> connection_list;
    //std::vector<std::vector<int>> weight_list;

    public:
        int potential;

        cl_int err;

        cl_platform_id platform;
        cl_device_id device;

        cl_context context;
        cl_command_queue queue;
        cl_program program;
        cl_kernel kernel;

        cl_mem buffer_axon;
        cl_mem buffer_weight;
        cl_mem buffer_integration;

        NeuronBlock(int potential_init = 0, int neuron_numbers = 256);

        NeuronBlock::~NeuronBlock();

        void Leak(int index, int leak_value);
        void Integrate(
            std::vector<int> axon_list,
            std::vector<std::vector<int>> weight_list, 
            int neuron_index 
            /*int axon_index*/
        );
        void Fire(std::vector<int>& neuron_list, int index, int threshold, int reset_value);

        void print() const; // Add this method to print NeuronBlock details
};