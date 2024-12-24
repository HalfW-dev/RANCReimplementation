#include "neuronblock.hpp"

// Constructor implementation
NeuronBlock::NeuronBlock(int potential_init, int neuron_numbers) {
    //std::vector<int> vec(neuron_numbers, potential_init);
    //this->potential = 0;

    std::vector<int> integration_empty(neuron_numbers, 0);

    integration_list = integration_empty;

    // Initialize OpenCL
    clGetPlatformIDs(1, &platform, nullptr);
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr);
    context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
    queue = clCreateCommandQueue(context, device, 0, &err);

    const char* kernelSource = R"(
    __kernel void elementwiseMultiply(
        __global const int* A, // Axons
        __global const int* B, // Flattened weights
        __global int* C,       // Integration results
        const int numAxons,    // Number of axons per neuron
        const int numNeurons   // Total number of neurons
    ) {
        int global_id = get_global_id(0); // Neuron index
        if (global_id >= numNeurons) return;

        int integration_value = 0;
        for (int i = 0; i < numAxons; i++) {
            integration_value += A[i] * B[global_id * numAxons + i];
        }
        C[global_id] = integration_value;
    }
    )";

    program = clCreateProgramWithSource(context, 1, &kernelSource, nullptr, &err);
    clBuildProgram(program, 1, &device, "-g", nullptr, nullptr);
    kernel = clCreateKernel(program, "elementwiseMultiply", &err);

    // Allocate persistent GPU buffers
    buffer_axon = nullptr;
    buffer_weight = nullptr;
    buffer_potential = nullptr;
}

NeuronBlock::~NeuronBlock() {
    if (buffer_axon) clReleaseMemObject(buffer_axon);
    if (buffer_weight) clReleaseMemObject(buffer_weight);
    if (buffer_potential) clReleaseMemObject(buffer_potential);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
}

// Leak method implementation
void NeuronBlock::Leak(int index, int leak_value) {
    for(int i=0; i<integration_list.size(); i++) {
        integration_list[i] += leak_value;
    }
    //this->potential += leak_value; // Leak
    //std::cout << "Applying leak value " << leak_value << std::endl;
}

// Integrate method implementation
void NeuronBlock::Integrate(
    std::vector<int> axon_list,
    std::vector<std::vector<int>> weight_list,
    std::vector<int>& neuron_list // New parameter
) 
{
    std::vector<int> flattened_weights;
    for (const auto& row : weight_list) {
        flattened_weights.insert(flattened_weights.end(), row.begin(), row.end());
    }

    // Number of neurons and axons
    int numNeurons = neuron_list.size();
    int numAxons = axon_list.size();

    // Allocate GPU buffers if not already created
    if (!buffer_axon) {
        buffer_axon = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(int) * numAxons, nullptr, &err);
    }
    if (!buffer_weight) {
        buffer_weight = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(int) * flattened_weights.size(), nullptr, &err);
    }
    if (!buffer_potential) {
        buffer_potential = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(int) * numNeurons, nullptr, &err);
    }

    // Update axon_list and weight_list data on the GPU
    clEnqueueWriteBuffer(queue, buffer_axon, CL_TRUE, 0, sizeof(int) * numAxons, axon_list.data(), 0, nullptr, nullptr);
    clEnqueueWriteBuffer(queue, buffer_weight, CL_TRUE, 0, sizeof(int) * flattened_weights.size(), flattened_weights.data(), 0, nullptr, nullptr);

    size_t localSize = (numAxons * numNeurons) / 256; // Define work-items per group
    size_t globalSize = numAxons * numNeurons; // Align globalSize to a multiple of localSize
    
    // Set kernel arguments
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &buffer_axon);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &buffer_weight);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &buffer_potential);
    clSetKernelArg(kernel, 3, sizeof(int), &numAxons);
    clSetKernelArg(kernel, 4, sizeof(int), &numNeurons);

    // Execute the kernel
    clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &globalSize, &localSize, 0, nullptr, nullptr);

    // Read back the results
    clEnqueueReadBuffer(queue, buffer_potential, CL_TRUE, 0, sizeof(int) * numNeurons, integration_list.data(), 0, nullptr, nullptr);

    // Update neuron potentials and process firing
    // for (int i = 0; i < numNeurons; i++) {
    //     neuron_list[i] = (integration_array[i] >= 0) ? 1 : 0; // Determine if the neuron fires
    // }
}

// Fire method implementation
void NeuronBlock::Fire(
    std::vector<int>& neuron_list, 
    int threshold, 
    int reset_value
) {
    for (int i = 0; i < neuron_list.size(); i++) {
        neuron_list[i] = (integration_list[i] >= threshold) ? 1 : 0; // Fire
        integration_list[i] = reset_value; // Reset potential
    }
}

// void NeuronBlock::print() const {
//     std::cout << "NeuronBlock potential: " << potential << std::endl;
// }
