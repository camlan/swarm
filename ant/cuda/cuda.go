package cuda

import (
	"math"
	"sort"

	"github.com/camlan/swarm/ant/internal"
)

/*
#cgo LDFLAGS:  -L ./lib -lant -lcuda -lcudart -lm
#include <stdlib.h>
void scale_city_matrix_wrp(double* flatScaledCityMap, const double* flatCityMap, unsigned int size, double distanceScaler);
void evaporate_fermone_wrp(double* fermoneMatrix1D, unsigned int size, double fermoneEvaporation);
void move_ants_wrp(double* cityMap1D, double* distanceMap1D, double* fermoneMap1D, unsigned int mapSize, unsigned int citiesCount, double fermoneImportance, double distanceImportance, double* distances, int* citySequences);
void leave_fermone_wrp(double* fermoneMap1D, int* citySequences, double * distances, double fermoneIncrease, unsigned int mapSize, unsigned int cityCount);
*/
import "C"

func FindShortestPath(distanceMap [][]float64) (bestPathPtr *internal.Path) {
	initFermone, fermoneImportance, distanceScaler, distanceImportance, fermoneEvaporation, fermoneLeft, iterations := internal.SetUpParamters(distanceMap)

	cityDistanceMatrixScaled := generateScaledCityDistanceMatrix(distanceMap, distanceScaler)
	numberOfCities := len(distanceMap)
	distanceMap1D := flattenArray(distanceMap)

	fermoneMap := internal.GenerateFermoneMap(numberOfCities, initFermone)
	fermoneMap1D := flattenArray(fermoneMap)

	bestPath := internal.Path{Distance: math.MaxInt, CitySequence: []int{}}
	paths := []internal.Path{}

	for iteration := 0; iteration < iterations; iteration++ {
		paths = []internal.Path{}
		distances, citySequence1D := moveAnts(numberOfCities, cityDistanceMatrixScaled, distanceMap1D, fermoneMap1D, fermoneImportance, distanceImportance)
		citySequences := dim2DArrayInt(citySequence1D)
		for i, distance := range distances {
			paths = append(paths, internal.Path{Distance: distance, CitySequence: citySequences[i]})
		}

		sort.Slice(paths, func(i, j int) bool {
			return paths[i].Distance < paths[j].Distance
		})

		if bestPath.Distance > paths[0].Distance {
			bestPath = paths[0]
		}

		fermoneMap1D = evaporateFermone(fermoneMap1D, fermoneEvaporation)
		fermoneMap1D = leaveFermone(fermoneMap1D, citySequence1D, distances, fermoneLeft)
	}

	bestPathPtr = &bestPath
	return
}

func moveAnts(numberOfCities int, cityMapScaled []float64, distanceMap1D []float64, fermoneMap1D []float64, fermoneImportance float64, distanceImportance float64) (distance []float64, citySequence1D []int32) {
	distance = make([]float64, numberOfCities)
	citySequence1D = make([]int32, numberOfCities*numberOfCities)
	C.move_ants_wrp((*C.double)(&cityMapScaled[0]), (*C.double)(&distanceMap1D[0]), (*C.double)(&fermoneMap1D[0]), C.uint(len(fermoneMap1D)), C.uint(numberOfCities),
		C.double(fermoneImportance), C.double(distanceImportance), (*C.double)(&distance[0]), (*C.int)(&citySequence1D[0]))
	return
}

func leaveFermone(fermoneMap1D []float64, citySequence1D []int32, distances []float64, fermoneIncrease float64) []float64 {
	C.leave_fermone_wrp((*C.double)(&fermoneMap1D[0]), (*C.int)(&citySequence1D[0]), (*C.double)(&distances[0]), C.double(fermoneIncrease), C.uint(len(fermoneMap1D)), C.uint(len(distances)))
	return fermoneMap1D
}

func evaporateFermone(fermoneMap1D []float64, fermoneEvaporation float64) []float64 {
	C.evaporate_fermone_wrp((*C.double)(&fermoneMap1D[0]), C.uint(len(fermoneMap1D)), C.double(fermoneEvaporation))
	return fermoneMap1D
}

func generateScaledCityDistanceMatrix(cityMap [][]float64, distanceScaler float64) (cityMapScaled []float64) {
	flatSize := len(cityMap) * len(cityMap)
	cityMapScaled = make([]float64, flatSize)
	flatCityMap := flattenArray(cityMap)

	C.scale_city_matrix_wrp((*C.double)(&cityMapScaled[0]), (*C.double)(&flatCityMap[0]), C.uint(flatSize), C.double(distanceScaler))

	return
}

func flattenArray(arr [][]float64) (flatArray []float64) {
	offset := len(arr)
	flatArray = make([]float64, offset*offset)
	for i, singleArray := range arr {
		for j, value := range singleArray {
			flatArray[i*offset+j] = value
		}
	}
	return
}

func dim2DArray(flatArray []float64) (arr [][]float64) {
	size := int(math.Sqrt(float64(len(flatArray))))
	arr = make([][]float64, size)
	for i := range arr {
		arr[i] = make([]float64, size)
		for j := range size {
			arr[i][j] = flatArray[i*size+j]
		}
	}
	return
}

func dim2DArrayInt(flatArray []int32) (arr [][]int) {
	size := int(math.Sqrt(float64(len(flatArray))))
	arr = make([][]int, size)
	for i := range arr {
		arr[i] = make([]int, size)
		for j := range size {
			arr[i][j] = int(flatArray[i*size+j])
		}
	}
	return
}
