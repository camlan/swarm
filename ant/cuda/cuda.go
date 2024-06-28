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
	initFermone, _, distanceScaler, _, _, _, iterations := internal.SetUpParamters(distanceMap)

	cityDistanceMatrixScaled := generateScaledCityDistanceMatrix(distanceMap, distanceScaler)
	numberOfCities := len(distanceMap)

	fermoneMatrix := internal.GenerateFermoneMatrix(len(cityDistanceMatrixScaled), initFermone)

	fmt.Printf("Number of cities: %d\n", numberOfCities)
	fmt.Printf("Scaled matrix: %v\n", dim2DArray(cityDistanceMatrixScaled))
	fmt.Printf("Initial fermone matrix: %v\n", fermoneMatrix)

	bestPath := internal.Path{Distance: math.MaxInt, CitySequence: []int{}}
	// paths := []internal.Path{};

	for iteration := 0; iteration < iterations; iteration++ {
		// paths = []internal.Path{};
		moveAnts(numberOfCities, cityDistanceMatrixScaled)

	}

	bestPathPtr = &bestPath
	return
}

func moveAnts(numberOfCities int, cityMapScaled []float64) {

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
