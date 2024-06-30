#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdlib.h>

struct NextCity {
    int cityIndex;
    double probability;
    NextCity* next;
};

__global__ void addKernel(int* c, const int* a, const int* b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i] * 2;
}

__global__ void scaleMapKernel(double* scaledMap, double* baseMap, double scaler, unsigned int size) {
    int i = threadIdx.x;
    if (i < size) {
        if (baseMap[i] > 0) {
            scaledMap[i] = scaler / baseMap[i];
        }
        else {
            scaledMap[i] = 0;
        }
    }
}


__global__ void evaporateFermoneKernel(double* fermoneMap1D, unsigned int size, double fermoneEvaporation) {
    int i = threadIdx.x; 
    if (i < size) {
        fermoneMap1D[i] *= fermoneEvaporation;
    }
}

__global__ void moveAnt(double* cityMap1D, double* fermoneMap1D, unsigned int mapSize, unsigned int citiesCount, double fermoneImportance, double distanceImportance) {
    int ant = threadIdx.x; // TODO consider different block/thread structure
    char visited [100]; // TODO figure out proper memory allocation technique within device to share accross all threads (should be citiesCount)
    int citySequence[100]; // TODO figure out proper memory allocation technique within device to share accross all threads (should be citiesCount)
    NextCity* nextCityProbabilities;

    int currentCity = ant;
    for (int i = 0; i < citiesCount; i++) {
        visited[currentCity] = 1;
        citySequence[i] = currentCity;
        nextCityProbabilities = calculatePathsSelectionProbabilies(cityMap1D, fermoneMap1D, fermoneImportance, distanceImportance, currentCity, visited, citiesCount);


    }

}

NextCity* calculatePathsSelectionProbabilies(double* cityMap1D, double* fermoneMap1D, double fermoneImportance, double distanceImportance, int currentCity, char* visited, int citiesCount) {
    NextCity * nc;

    double* distances = (cityMap1D + (citiesCount * currentCity));
    double totalProbabilty = 0;
    for (int i = 0; i < citiesCount; i++) {
        if (distances[i] > 0 && !visited[i]) {
            nc = (NextCity*)malloc(sizeof(struct NextCity));
            nc->cityIndex = i;
            nc-> // calculatePathSelectionProbalitity
        }
    }

    return nc
}

void evaporateFermone(double* fermoneMap1D, unsigned int size, double fermoneEvaporation) {
    double* dev_fermone_map = 0;

    cudaError_t cudaStatus;

    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_fermone_map, size * sizeof(double));
    if (cudaStatus != cudaSuccess) {
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_fermone_map, fermoneMap1D, size * sizeof(double), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        goto Error;
    }

    evaporateFermoneKernel << <1, size >> > (dev_fermone_map, size, fermoneEvaporation);

    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        goto Error;
    }

    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(fermoneMap1D, dev_fermone_map, size * sizeof(double), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        goto Error;
    }


    Error:
        cudaFree(dev_fermone_map);
}

void scaleMap(double* scaledMap, const double* baseMap, double scaler, unsigned int size) {
    double* dev_scaled_map = 0;
    double* dev_base_map = 0;

    cudaError_t cudaStatus;

    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_scaled_map, size * sizeof(double));
    if (cudaStatus != cudaSuccess) {
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_base_map, size * sizeof(double));
    if (cudaStatus != cudaSuccess) {
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_base_map, baseMap, size * sizeof(double), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        goto Error;
    }

    scaleMapKernel << <1, size >> > (dev_scaled_map, dev_base_map, scaler, size);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        goto Error;
    }

    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(scaledMap, dev_scaled_map, size * sizeof(double), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        goto Error;
    }

    Error:
        cudaFree(dev_scaled_map);
        cudaFree(dev_base_map);
}

// Helper function for using CUDA to add vectors in parallel.
void addcuda(int* c, const int* a, const int* b, unsigned int size)
{
    int* dev_a = 0;
    int* dev_b = 0;
    int* dev_c = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    addKernel << <1, size >> > (dev_c, dev_a, dev_b);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        goto Error;
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
}

extern "C" {
    void add_ant_wpr(int* c, const int* a, const int* b, int size) {
        addcuda(c, a, b, size);
    }

    void scale_city_matrix_wrp(double* flatScaledCityMap, const double* flatCityMap, unsigned int size, double distanceScaler) {
        scaleMap(flatScaledCityMap, flatCityMap, distanceScaler, size);
    }

    void evaporate_fermone_wrp(double* fermoneMap1D, unsigned int size, double fermoneEvaporation) {
        evaporateFermone(fermoneMap1D, size, fermoneEvaporation);
    }
}