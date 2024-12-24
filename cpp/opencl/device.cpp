#include <CL/cl.h>
#include <iostream>
#include <vector>

int main() {
    cl_int err;
    cl_platform_id platform;
    cl_device_id device;

    // Get the first platform
    err = clGetPlatformIDs(1, &platform, nullptr);
    if (err != CL_SUCCESS) {
        std::cerr << "Error: Unable to get platform ID.\n";
        return 1;
    }

    // Get the first device of the platform
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 1, &device, nullptr);
    if (err != CL_SUCCESS) {
        std::cerr << "Error: Unable to get device ID.\n";
        return 1;
    }

    // Query device limits
    size_t maxWorkItemSizes[3];
    size_t maxWorkGroupSize;

    clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(maxWorkItemSizes), maxWorkItemSizes, nullptr);
    clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(maxWorkGroupSize), &maxWorkGroupSize, nullptr);

    // Display maximum limits
    std::cout << "Max work-item sizes: "
              << maxWorkItemSizes[0] << " x "
              << maxWorkItemSizes[1] << " x "
              << maxWorkItemSizes[2] << std::endl;

    std::cout << "Max work-group size: " << maxWorkGroupSize << std::endl;

    // Example calculation of maximum global size
    size_t localSize = 64; // Define a local size
    if (localSize > maxWorkGroupSize) {
        localSize = maxWorkGroupSize; // Adjust if localSize exceeds max work-group size
    }

    size_t globalSize = maxWorkItemSizes[0]; // Assume 1D kernel
    globalSize = ((globalSize + localSize - 1) / localSize) * localSize; // Align global size
    std::cout << "Adjusted global size: " << globalSize << std::endl;

    return 0;
}
