package cuda

import (
	"fmt"
	"math"

	"github.com/camlan/swarm/ant/internal"
)

/*
#cgo LDFLAGS:  -L ./lib -lant -lcuda -lcudart -lm
#include <stdlib.h>
void scale_city_matrix_wrp(double* flatScaledCityMap, const double* flatCityMap, unsigned int size, double distanceScaler);
*/
import "C"

func FindShortestPath(distanceMap [][]float64) (bestPathPtr *internal.Path) {
	// initFermone, fermoneImportance, distanceScaler, distanceImportance, fermoneEvaporation, fermoneLeft, iterations := internal.SetUpParamters(distanceMap)
	_, _, distanceScaler, _, _, _, _ := internal.SetUpParamters(distanceMap)

	cityDistanceMatrixScaled := generateScaledCityDistanceMatrix(distanceMap, distanceScaler)
	numberOfCities := len(distanceMap)

	fmt.Printf("Number of cities: %d\n", numberOfCities)
	fmt.Printf("Scaled matrix: %v\n", dim2DArray(cityDistanceMatrixScaled))

	bestPath := internal.Path{Distance: math.MaxInt, CitySequence: []int{}}

	bestPathPtr = &bestPath
	return
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
