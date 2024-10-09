#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "curand_kernel.h"

#include <stdlib.h>
#include <math.h>
#include <time.h>
//#include <stdio.h>

struct NextCity {
    int cityIndex;
    double probability;
};

__global__ void setupKernel(curandState* state) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    /* Each thread gets same seed, a different sequence
       number, no offset */
    curand_init(1234, id, 0, &state[id]);
}

__global__ void leaveFermone(double* fermoneMap1D, int* citySequences, double * distances, double fermoneIncrease, int cityCount) {
    int i = blockIdx.x;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    double leftFermone = fermoneIncrease / distances[i];
    if (j > 0) {
        fermoneMap1D[citySequences[j - 1] * cityCount + citySequences[j]] += leftFermone;
        fermoneMap1D[citySequences[j] * cityCount + citySequences[j-1]] += leftFermone;
    }

    if (j == 0) {
        fermoneMap1D[citySequences[i * cityCount] * cityCount + citySequences[i * cityCount + cityCount - 1]] += leftFermone;
        fermoneMap1D[citySequences[i * cityCount + cityCount - 1] * cityCount + citySequences[i * cityCount]] += leftFermone;
    }

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

__device__ double calculatePathSelectionProbalitity(double distance, double distanceImportance, double fermone, double fermoneImportance) {
    return pow(distance, distanceImportance) * pow(fermone, fermoneImportance);
}

__device__ void calculatePathsSelectionProbabilies(double* cityMap1D, double* fermoneMap1D, double fermoneImportance, double distanceImportance, int currentCity, char* visited, int citiesCount, int * nextCitiesIndex, double * nextCityProbability) {
   //  NextCity* nc;

    double* distances = (cityMap1D + (citiesCount * currentCity));
    double* fermones = (fermoneMap1D + (citiesCount * currentCity));
    double totalProbabilty = 0;
    //nc = (NextCity*)malloc(citiesCount * sizeof(struct NextCity));

    for (int i = 0; i < citiesCount; i++) {
        nextCityProbability[i] = -1000;
        nextCitiesIndex[i]= -1;
    }

    int position = 0;
    for (int i = 0; i < citiesCount; i++) {
        if (distances[i] > 0 && !visited[i]) {
            nextCitiesIndex[position] = i;
            nextCityProbability[position] = calculatePathSelectionProbalitity(distances[i], distanceImportance, fermones[i], fermoneImportance);
            totalProbabilty += nextCityProbability[position];
            //nc->next = (NextCity*)malloc(sizeof(struct NextCity));
            //nc = nc->next;
            //nc->next = 0;
            //nc->probability = -100;
            position++;
        }
    }

    
    nextCityProbability[0] /= totalProbabilty;

    for (int i = 1; i < position; i++) {
        nextCityProbability[i] /= totalProbabilty;
        nextCityProbability[i] += nextCityProbability[i - 1];
    }

    //while (nc->next && nc->next->probability > 0) {
    //    nc->next->probability /= totalProbabilty;
    //    nc->next->probability += nc->probability;
    //    nc = nc->next;
    //}

    //return nc;
}

__device__ int selectNexyCity(int* nextCitiesIndex, double* nextCityProbability, double randomSelector, int citiesCount) {
    //for (int i = 0; i < citiesCount; i++) {
    //    if (randomSelector < nc[i].probability) {

    //    }
    //}


    // return nc[0].cityIndex;
    int pos = 0;
    for (int i = 0; i < citiesCount; i++) {
        if (nextCitiesIndex[i] < 0) {
            break;
        }

        if (randomSelector < nextCityProbability[i]) {
            return nextCitiesIndex[i];
        }
        pos++;
    }
    return nextCitiesIndex[pos-1];

    //while (nc) {
    //    if (randomSelector < nc->probability) {
    //        return nc->cityIndex;
    //    }
    //    if (!nc->next) {
    //        return nc->cityIndex;
    //    }
    //    nc = nc->next;
    //}
    //return nc->cityIndex;
}

__device__ double calculatePathDistance(int* citySequence, unsigned int citiesCount, double* distanceMap1D) {
    double distance = 0;

    for (int i = 1; i < citiesCount; i++) {
        distance += *(distanceMap1D + (*(citySequence + i - 1) * citiesCount) + *(citySequence + i));
    }
    distance += *(distanceMap1D + (*(citySequence + citiesCount - 1) * citiesCount) + *(citySequence));

    return distance;
}

__global__ void moveAnt(double* cityMap1D, double* distanceMap1D, double* fermoneMap1D, unsigned int mapSize, unsigned int citiesCount, double fermoneImportance, double distanceImportance, double* distances, int* citySequences, curandState* state) {
    int ant = threadIdx.x; // TODO consider different block/thread structure
    char visited[4];
    int citySequence[4];
    // NextCity nextCityProbabilities[4]; //= (NextCity**)malloc(sizeof(struct NextCity));
    int nextCityProbabilitiesIndex[4];
    double nextCityProbabilitiesProbability[4];


    int currentCity = ant;
    curandState localState = state[ant];
    double r;
    for (int i = 0; i < citiesCount; i++) {
        visited[currentCity] = 1;
        citySequence[i] = currentCity;
        calculatePathsSelectionProbabilies(cityMap1D, fermoneMap1D, fermoneImportance, distanceImportance, currentCity, visited, citiesCount, nextCityProbabilitiesIndex, nextCityProbabilitiesProbability);
        r = curand_uniform(&localState);
        currentCity = selectNexyCity(nextCityProbabilitiesIndex, nextCityProbabilitiesProbability, r, citiesCount);
    }

    distances[ant] = calculatePathDistance(citySequence, citiesCount, distanceMap1D);

    for (int i = 0; i < citiesCount; i++) {
        citySequences[ant * citiesCount + i] = citySequence[i];
    }

    //NextCity* tmp = nextCityProbabilities;

    //int i = 0;
    //while (tmp && i < citiesCount) {
    //    free(tmp);
    //    tmp = (nextCityProbabilities + ++i);
    //}

    //while (nextCityProbabilities->next) {
    //    tmp = nextCityProbabilities->next;
    //    free(nextCityProbabilities);
    //    nextCityProbabilities = tmp;
    //}

    // free(nextCityProbabilities);
}

//__global__ void moveAnt_old(double* cityMap1D, double* fermoneMap1D, unsigned int mapSize, unsigned int citiesCount, double fermoneImportance, double distanceImportance, double* distances, int *citySequences, curandState* state) {
//    int ant = threadIdx.x; // TODO consider different block/thread structure
//    char visited [100]; // TODO figure out proper memory allocation technique within device to share accross all threads (should be citiesCount)
//    int citySequence[100]; // TODO figure out proper memory allocation technique within device to share accross all threads (should be citiesCount)
//    NextCity* nextCityProbabilities=0;
//
//    int currentCity = ant;
//    curandState localState = state[ant];
//    for (int i = 0; i < citiesCount; i++) {
//        visited[currentCity] = 1;
//        citySequence[i] = currentCity;
//        nextCityProbabilities = calculatePathsSelectionProbabilies(cityMap1D, fermoneMap1D, fermoneImportance, distanceImportance, currentCity, visited, citiesCount, nextCityProbabilities);
//        double r = (double)curand_uniform(&localState);
//        currentCity = selectNexyCity(nextCityProbabilities, r, citiesCount);
//    }
//    double distance = calculatePathDistance(citySequence, citiesCount, cityMap1D);
//
//    distances[ant] = distance;
//    
//    //TODO transer for loop to parallel ??
//    for (int i = 0; i < citiesCount; i++) {
//        citySequences[ant * citiesCount + i] = citySequence[i];
//    }
//}

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

void moveAnts(double* cityMap1D, double* distanceMap1D, double* fermoneMap1D, unsigned int mapSize, unsigned int citiesCount, double fermoneImportance, double distanceImportance, double* distances, int* citySequences) {
    double* dev_city_map = 0;
    double* dev_distance_map = 0;
    double* dev_fermone_map = 0;
    double* dev_distances = 0;
    int* dev_city_sequences = 0;
    cudaError_t cudaStatus;

    curandState* devStates=0;

    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        goto Error;
    }


    cudaStatus = cudaMalloc((void**)&dev_city_map, mapSize * sizeof(double));
    if (cudaStatus != cudaSuccess) {
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_distance_map, mapSize * sizeof(double));
    if (cudaStatus != cudaSuccess) {
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_fermone_map, mapSize * sizeof(double));
    if (cudaStatus != cudaSuccess) {
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_distances, citiesCount * sizeof(double));
    if (cudaStatus != cudaSuccess) {
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_city_sequences, mapSize * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&devStates, citiesCount * sizeof(curandState));
    if (cudaStatus != cudaSuccess) {
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_city_map, cityMap1D, mapSize * sizeof(double), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_distance_map, distanceMap1D, mapSize * sizeof(double), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        goto Error;
    }
    

    cudaStatus = cudaMemcpy(dev_fermone_map, fermoneMap1D, mapSize * sizeof(double), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        goto Error;
    }

    setupKernel << <1, citiesCount >> > (devStates);

    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        goto Error;
    }

    moveAnt << <1, citiesCount >> > (dev_city_map, dev_distance_map, dev_fermone_map, mapSize, citiesCount, fermoneImportance, distanceImportance, dev_distances, dev_city_sequences, devStates);

    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        goto Error;
    }

    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        goto Error;
    }

    cudaStatus = cudaMemcpy(distances, dev_distances, citiesCount * sizeof(double), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        goto Error;
    }

    cudaStatus = cudaMemcpy(citySequences, dev_city_sequences, mapSize * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        goto Error;
    }

Error:
    cudaFree(dev_city_map);
    cudaFree(dev_distance_map);
    cudaFree(dev_fermone_map);
    cudaFree(dev_distances);
    cudaFree(dev_city_sequences);
}

void leaveFermones(double* fermoneMap1D, int* citySequences, double* distances, double fermoneIncrease, unsigned int mapSize, unsigned int citiesCount) {

    double* dev_fermone_map = 0;
    int* dev_city_sequences = 0;
    double* dev_distancecs = 0;

    cudaError_t cudaStatus;

    cudaStatus = cudaMalloc((void**)&dev_fermone_map, mapSize * sizeof(double));
    if (cudaStatus != cudaSuccess) {
        goto Error;
    }


    cudaStatus = cudaMalloc((void**)&dev_city_sequences, mapSize * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_distancecs, citiesCount * sizeof(double));
    if (cudaStatus != cudaSuccess) {
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_city_sequences, citySequences, mapSize * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_fermone_map, fermoneMap1D, mapSize * sizeof(double), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_distancecs, distances, citiesCount * sizeof(double), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        goto Error;
    }


    leaveFermone << <citiesCount, citiesCount >> > (dev_fermone_map, dev_city_sequences, dev_distancecs, fermoneIncrease, citiesCount);


    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        goto Error;
    }

    cudaStatus = cudaMemcpy(fermoneMap1D, dev_fermone_map, mapSize * sizeof(double), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        goto Error;
    }


Error:
    cudaFree(dev_fermone_map);
    cudaFree(dev_city_sequences);
    cudaFree(dev_distancecs);
}

extern "C" {

    void scale_city_matrix_wrp(double* flatScaledCityMap, const double* flatCityMap, unsigned int size, double distanceScaler) {
        scaleMap(flatScaledCityMap, flatCityMap, distanceScaler, size);
    }

    void evaporate_fermone_wrp(double* fermoneMap1D, unsigned int size, double fermoneEvaporation) {
        evaporateFermone(fermoneMap1D, size, fermoneEvaporation);
    }


    void move_ants_wrp(double* cityMap1D, double * distanceMap1D, double* fermoneMap1D, unsigned int mapSize, unsigned int citiesCount, double fermoneImportance, double distanceImportance, double* distances, int* citySequences) {
        moveAnts(cityMap1D, distanceMap1D, fermoneMap1D, mapSize, citiesCount, fermoneImportance, distanceImportance, distances, citySequences);
    }

    void leave_fermone_wrp(double* fermoneMap1D, int* citySequences, double * distances, double fermoneIncrease, unsigned int mapSize, unsigned int cityCount) {
        leaveFermones(fermoneMap1D, citySequences, distances, fermoneIncrease, mapSize, cityCount);
    }
}
