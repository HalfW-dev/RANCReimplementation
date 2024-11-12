#include <cuda_runtime.h>
#include <iostream>

int main() {
    int deviceCount;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    if (err != cudaSuccess) {
        std::cerr << "Failed to get device count: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }

    if (deviceCount == 0) {
        std::cerr << "No CUDA-capable devices detected." << std::endl;
        return 1;
    }

    for (int device = 0; device < deviceCount; ++device) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, device);

        std::cout << "Device " << device << ": " << prop.name << std::endl;
        std::cout << "  Compute capability: " << prop.major << "." << prop.minor << std::endl;
        std::cout << "  Total global memory: " << prop.totalGlobalMem / (1024 * 1024) << " MB" << std::endl;
        std::cout << "  Max threads per block: " << prop.maxThreadsPerBlock << std::endl;
        std::cout << "  Max threads per multiprocessor: " << prop.maxThreadsPerMultiProcessor << std::endl;
        std::cout << "  Number of multiprocessors: " << prop.multiProcessorCount << std::endl;
        std::cout << "  Max threads dimensions: (" << prop.maxThreadsDim[0] << ", "
                  << prop.maxThreadsDim[1] << ", " << prop.maxThreadsDim[2] << ")" << std::endl;
        std::cout << "  Max grid dimensions: (" << prop.maxGridSize[0] << ", "
                  << prop.maxGridSize[1] << ", " << prop.maxGridSize[2] << ")" << std::endl;
        std::cout << "  Warp size: " << prop.warpSize << std::endl;
        std::cout << "  Shared memory per block: " << prop.sharedMemPerBlock / 1024 << " KB" << std::endl;
        std::cout << "  Clock rate: " << prop.clockRate / 1000 << " MHz" << std::endl;
        std::cout << "  Memory Clock rate: " << prop.memoryClockRate / 1000 << " MHz" << std::endl;
        std::cout << "  Memory Bus Width: " << prop.memoryBusWidth << " bits" << std::endl;
        std::cout << "  Total constant memory: " << prop.totalConstMem / 1024 << " KB" << std::endl;
        std::cout << "  Multiprocessor count: " << prop.multiProcessorCount << std::endl;
        std::cout << "  Max pitch: " << prop.memPitch << std::endl;
        std::cout << "  Texture alignment: " << prop.textureAlignment << std::endl;
        std::cout << "  Device overlap: " << (prop.deviceOverlap ? "Enabled" : "Disabled") << std::endl;
        std::cout << "  Concurrent Kernels: " << (prop.concurrentKernels ? "Yes" : "No") << std::endl;
        std::cout << "  ECC Enabled: " << (prop.ECCEnabled ? "Yes" : "No") << std::endl;
        std::cout << "  Memory Bus Width: " << prop.memoryBusWidth << " bits" << std::endl;
        std::cout << "  Peak Memory Bandwidth (GB/s): "
                  << 2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6 << std::endl;
        std::cout << std::endl;
    }

    return 0;
}
