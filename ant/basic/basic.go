package basic

import (
	"math"
	"math/rand"
	"slices"
	"sort"

	"github.com/hashicorp/go-set/v2"
)

type Path struct {
	citySequence []int
	distance     float64
}

type NextCity struct {
	cityIndex   int
	probability float64
}

func FindShortestPath(distanceMap [][]float64) (bestPath Path) {
	bestPath = Path{citySequence: []int{}, distance: 0}
	initFermone, fermoneImportance, distanceScaler, distanceImportance, fermoneEvaporation, fermoneLeft, iterations := setUpParamters(distanceMap)
	cityDistanceMatrixScaled := generateScaledCityDistanceMatrix(distanceMap, distanceScaler)
	numberOfCities := len(distanceMap)

	fermoneMatrix := generateFermoneMatrix(len(cityDistanceMatrixScaled), initFermone)

	bestPath = Path{distance: math.MaxInt, citySequence: []int{}}
	paths := []Path{}
	for iteration := 0; iteration < iterations; iteration++ {
		paths = []Path{}
		for ant := 0; ant < numberOfCities; ant++ {
			visited := set.New[int](numberOfCities)
			citySequence := []int{}
			currentCity := ant
			for i := 0; i < numberOfCities; i++ {
				visited.Insert(currentCity)
				citySequence = append(citySequence, currentCity)
				nextCityProbabilities := calculatePathsSelectionProbabilies(cityDistanceMatrixScaled, fermoneMatrix,
					fermoneImportance, distanceImportance, currentCity, visited)
				citySelection := rand.Float64()
				currentCity = selectNextCity(nextCityProbabilities, citySelection)
			}
			path := Path{distance: calculatePathDistance(citySequence, distanceMap), citySequence: citySequence}
			paths = append(paths, path)
		}
		sort.Slice(paths, func(i, j int) bool {
			return paths[i].distance < paths[j].distance
		})
		if bestPath.distance > paths[0].distance {
			bestPath = paths[0]
		}
		fermoneMatrix = evaporateFermone(fermoneMatrix, fermoneEvaporation)
		fermoneMatrix = leaveFermone(fermoneMatrix, paths, fermoneLeft)
	}
	return
}

func setUpParamters(distanceMap [][]float64) (float64, float64, float64, float64, float64, float64, int) {
	initFermone := 0.2
	fermoneImportance := 1.3
	distanceImportance := 1.0
	fermoneEvaporation := 0.6
	fermoneLeft := 7.0
	iterations := int(math.Min(math.Max(5000, float64(len(distanceMap))*2000), 60000))
	return initFermone, fermoneImportance, calculateDistanceScalerParam(distanceMap), distanceImportance, fermoneEvaporation, fermoneLeft, iterations
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

func generateScaledCityDistanceMatrix(cityMap [][]float64, distanceScaler float64) (cityMapScaled [][]float64) {
	cityMapScaled = make([][]float64, len(cityMap))
	for i := range cityMap {
		cityMapScaled[i] = make([]float64, len(cityMap))
		for j := range len(cityMap) {
			if cityMap[i][j] > 0.0 {
				cityMapScaled[i][j] = distanceScaler / float64(cityMap[i][j])
			} else {
				cityMapScaled[i][j] = 0.0
			}
		}
	}
	return
}

func generateFermoneMatrix(citiesCount int, initFermone float64) (fermoneMatrix [][]float64) {
	fermoneMatrix = make([][]float64, citiesCount)
	for i := range fermoneMatrix {
		fermoneMatrix[i] = make([]float64, citiesCount)
		for j := range citiesCount {
			fermoneMatrix[i][j] = initFermone
		}
	}
	return
}

func calculatePathsSelectionProbabilies(cityDistanceMatrix [][]float64,
	fermoneMatrix [][]float64,
	fermoneImportance float64,
	distanceImportance float64,
	currentCity int,
	visited *set.Set[int]) (nextCityProbabilities []NextCity) {
	distances := cityDistanceMatrix[currentCity]

	citiesToCalculate := []int{}
	for i, distance := range distances {
		if distance > 0 && !visited.Contains(i) {
			citiesToCalculate = append(citiesToCalculate, i)
		}
	}

	nextCityProbabilities = []NextCity{}
	for _, cityToCalculate := range citiesToCalculate {
		nextCity := NextCity{cityIndex: cityToCalculate,
			probability: calculatePathSelectionProbalitity(cityDistanceMatrix[currentCity][cityToCalculate], distanceImportance,
				fermoneMatrix[currentCity][cityToCalculate], fermoneImportance)}
		nextCityProbabilities = append(nextCityProbabilities, nextCity)
	}

	var totalProbabilty float64
	totalProbabilty = 0
	for i := 0; i < len(nextCityProbabilities); i++ {
		totalProbabilty += nextCityProbabilities[i].probability
	}

	for i := 0; i < len(nextCityProbabilities); i++ {
		nextCityProbabilities[i].probability /= totalProbabilty
	}

	for i := 1; i < len(nextCityProbabilities); i++ {
		nextCityProbabilities[i].probability += nextCityProbabilities[i-1].probability
	}

	return
}

func calculatePathSelectionProbalitity(distance float64, distanceImporance float64, fermone float64, fermoneImportance float64) float64 {
	return math.Pow(distance, distanceImporance) * math.Pow(fermone, fermoneImportance)
}

func selectNextCity(cityProbablities []NextCity, selection float64) int {
	for i := 0; i < len(cityProbablities); i++ {
		if selection <= cityProbablities[i].probability {
			return cityProbablities[i].cityIndex
		}
	}

	if len(cityProbablities) > 0 {
		return cityProbablities[len(cityProbablities)-1].cityIndex
	}
	return -1
}

func calculatePathDistance(citySequence []int, cityMap [][]float64) float64 {
	distance := 0.0
	for i := 1; i < len(citySequence); i++ {
		distance += cityMap[citySequence[i-1]][citySequence[i]]
	}
	distance += cityMap[citySequence[len(citySequence)-1]][citySequence[0]]
	return distance
}

func evaporateFermone(fermoneMatrix [][]float64, fermoneEvaporation float64) [][]float64 {
	for i := 0; i < len(fermoneMatrix); i++ {
		for j := 0; j < len(fermoneMatrix); j++ {
			fermoneMatrix[i][j] *= fermoneEvaporation
		}
	}
	return fermoneMatrix
}

func leaveFermone(fermoneMatrix [][]float64, paths []Path, fermoneIncrease float64) [][]float64 {
	for _, path := range paths {
		leftFermone := float64(fermoneIncrease) / float64(path.distance)
		for i := 1; i < len(path.citySequence); i++ {
			fermoneMatrix[path.citySequence[i-1]][path.citySequence[i]] += leftFermone
			fermoneMatrix[path.citySequence[i]][path.citySequence[i-1]] += leftFermone
		}
		fermoneMatrix[path.citySequence[len(path.citySequence)-1]][path.citySequence[0]] += leftFermone
		fermoneMatrix[path.citySequence[0]][path.citySequence[len(path.citySequence)-1]] += leftFermone
	}
	return fermoneMatrix
}
