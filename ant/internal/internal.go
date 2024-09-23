package internal

import (
	"slices"
)

type Path struct {
	CitySequence []int
	Distance     float64
}

type NextCity struct {
	CityIndex   int
	Probability float64
}

func SetUpParamters(distanceMap [][]float64) (float64, float64, float64, float64, float64, float64, int) {
	initFermone := 0.2
	fermoneImportance := 1.3
	distanceImportance := 1.0
	fermoneEvaporation := 0.6
	fermoneLeft := 7.0
	iterations := 2
	// iterations := int(math.Min(math.Max(5000, float64(len(distanceMap))*2000), 60000))
	return initFermone, fermoneImportance, calculateDistanceScalerParam(distanceMap), distanceImportance, fermoneEvaporation, fermoneLeft, iterations
}

func GenerateFermoneMap(citiesCount int, initFermone float64) (fermoneMatrix [][]float64) {
	fermoneMatrix = make([][]float64, citiesCount)
	for i := range fermoneMatrix {
		fermoneMatrix[i] = make([]float64, citiesCount)
		for j := range citiesCount {
			fermoneMatrix[i][j] = initFermone
		}
	}
	return
}

func calculateDistanceScalerParam(distanceMap [][]float64) (scaler float64) {
	max := -1.0
	for _, distances := range distanceMap {
		currentMax := slices.Max(distances)
		if currentMax > max {
			max = currentMax
		}
	}
	return max / 10
}
