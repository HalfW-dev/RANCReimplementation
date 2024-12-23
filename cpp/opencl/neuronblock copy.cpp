#include "neuronblock.hpp"

// Constructor implementation
NeuronBlock::NeuronBlock(int potential_init, int neuron_numbers) {
    //std::vector<int> vec(neuron_numbers, potential_init);
    this->potential = 0;

    // Initialize OpenCL
    clGetPlatformIDs(1, &platform, nullptr);
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr);
    context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
    queue = clCreateCommandQueue(context, device, 0, &err);

    const char* kernelSource = R"(
    __kernel void elementwiseMultiply(
        __global const int* A,
        __global const int* B,
        __global int* C,
        const int param1,
        const int param2) {
        int id = get_global_id(0);
        int offset = param1 * param2; // Precompute offset for memory access
        C[id] = A[id] * B[offset + id];
    }
    )";

    program = clCreateProgramWithSource(context, 1, &kernelSource, nullptr, &err);
    clBuildProgram(program, 1, &device, "-g", nullptr, nullptr);
    kernel = clCreateKernel(program, "elementwiseMultiply", &err);

    // Allocate persistent GPU buffers
    buffer_axon = nullptr;
    buffer_weight = nullptr;
    buffer_integration = nullptr;
}

NeuronBlock::~NeuronBlock() {
    if (buffer_axon) clReleaseMemObject(buffer_axon);
    if (buffer_weight) clReleaseMemObject(buffer_weight);
    if (buffer_integration) clReleaseMemObject(buffer_integration);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
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
    int neuron_index
    //int axon_index
) 
{
    std::vector<int> flattened_weights;
    for (const auto& row : weight_list) {
        flattened_weights.insert(flattened_weights.end(), row.begin(), row.end());
    }

    // Allocate GPU buffers if not already created
    if (!buffer_axon) {
        buffer_axon = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(int) * axon_list.size(), nullptr, &err);
    }
    if (!buffer_weight) {
        buffer_weight = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(int) * flattened_weights.size(), nullptr, &err);
    }
    if (!buffer_integration) {
        buffer_integration = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(int) * axon_list.size(), nullptr, &err);
    }

    // Update axon_list and weight_list data on the GPU
    clEnqueueWriteBuffer(queue, buffer_axon, CL_TRUE, 0, sizeof(int) * axon_list.size(), axon_list.data(), 0, nullptr, nullptr);
    clEnqueueWriteBuffer(queue, buffer_weight, CL_TRUE, 0, sizeof(int) * flattened_weights.size(), flattened_weights.data(), 0, nullptr, nullptr);

    size_t globalSize = ((axon_list.size() + 63) / 64) * 64; // Align globalSize to a multiple of localSize
    size_t localSize = 64; // Define work-items per group
    int width = weight_list[0].size(); // Width of the weight_list matrix

    // Set kernel arguments
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &buffer_axon);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &buffer_weight);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &buffer_integration);
    clSetKernelArg(kernel, 3, sizeof(int), &neuron_index);
    clSetKernelArg(kernel, 4, sizeof(int), &width);

    // Execute the kernel
    clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &globalSize, &localSize, 0, nullptr, nullptr);

    std::vector<int> integration_array(axon_list.size(), 0);

    // Read back the result
    clEnqueueReadBuffer(queue, buffer_integration, CL_TRUE, 0, sizeof(int) * axon_list.size(), integration_array.data(), 0, nullptr, nullptr);

    //int integration_value = axon_list[axon_index] * weight_list[neuron_index][axon_index];

    for(int i=0; i<integration_array.size(); i++) {
        potential += integration_array[i];
    }
    
    //potential += integration_value; // Integrate

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
