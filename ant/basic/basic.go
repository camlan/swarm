package basic

import (
	"math"
	"math/rand"
	"slices"
	"sort"
	"sync"

	"github.com/camlan/swarm/ant/internal"
	"github.com/hashicorp/go-set/v2"
)

func FindShortestPath(distanceMap [][]float64) (bestPathPtr *internal.Path) {
	initFermone, fermoneImportance, distanceScaler, distanceImportance, fermoneEvaporation, fermoneLeft, iterations := internal.SetUpParamters(distanceMap)
	cityDistanceMatrixScaled := generateScaledCityDistanceMatrix(distanceMap, distanceScaler)
	numberOfCities := len(distanceMap)

	fermoneMap := internal.GenerateFermoneMap(numberOfCities, initFermone)

	bestPath := internal.Path{Distance: math.MaxInt, CitySequence: []int{}}
	paths := []internal.Path{}
	for iteration := 0; iteration < iterations; iteration++ {
		paths = []internal.Path{}

		for ant := 0; ant < numberOfCities; ant++ {
			visited := set.New[int](numberOfCities)
			citySequence := []int{}
			currentCity := ant
			for i := 0; i < numberOfCities; i++ {
				visited.Insert(currentCity)
				citySequence = append(citySequence, currentCity)
				nextCityProbabilities := calculatePathsSelectionProbabilies(cityDistanceMatrixScaled, fermoneMap,
					fermoneImportance, distanceImportance, currentCity, visited)
				citySelection := rand.Float64()
				currentCity = selectNextCity(nextCityProbabilities, citySelection)
			}
			path := internal.Path{Distance: calculatePathDistance(citySequence, distanceMap), CitySequence: citySequence}
			paths = append(paths, path)
		}

		sort.Slice(paths, func(i, j int) bool {
			return paths[i].Distance < paths[j].Distance
		})
		if bestPath.Distance > paths[0].Distance {
			bestPath = paths[0]
		}
		fermoneMap = evaporateFermone(fermoneMap, fermoneEvaporation)
		fermoneMap = leaveFermone(fermoneMap, paths, fermoneLeft)
	}
	bestPathPtr = &bestPath
	return
}

func FindShortestPathRouties(distanceMap [][]float64) (bestPathPtr *internal.Path) {
	initFermone, fermoneImportance, distanceScaler, distanceImportance, fermoneEvaporation, fermoneLeft, iterations := setUpParamters(distanceMap)
	cityDistanceMatrixScaled := generateScaledCityDistanceMatrix(distanceMap, distanceScaler)
	numberOfCities := len(distanceMap)

	fermoneMap := internal.GenerateFermoneMap(numberOfCities, initFermone)

	bestPath := internal.Path{Distance: math.MaxInt, CitySequence: []int{}}
	paths := []internal.Path{}
	var wg sync.WaitGroup
	var mutex sync.Mutex
	for iteration := 0; iteration < iterations; iteration++ {
		paths = []internal.Path{}
		wg.Add(numberOfCities)
		for ant := 0; ant < numberOfCities; ant++ {
			go func(ant int) {
				visited := set.New[int](numberOfCities)
				citySequence := []int{}
				currentCity := ant
				for i := 0; i < numberOfCities; i++ {
					visited.Insert(currentCity)
					citySequence = append(citySequence, currentCity)
					nextCityProbabilities := calculatePathsSelectionProbabilies(cityDistanceMatrixScaled, fermoneMap,
						fermoneImportance, distanceImportance, currentCity, visited)
					citySelection := rand.Float64()
					currentCity = selectNextCity(nextCityProbabilities, citySelection)
				}
				path := internal.Path{Distance: calculatePathDistance(citySequence, distanceMap), CitySequence: citySequence}
				mutex.Lock()
				{
					paths = append(paths, path)
				}
				mutex.Unlock()
				wg.Done()
			}(ant)
		}
		wg.Wait()
		sort.Slice(paths, func(i, j int) bool {
			return paths[i].Distance < paths[j].Distance
		})
		if bestPath.Distance > paths[0].Distance {
			bestPath = paths[0]
		}
		fermoneMap = evaporateFermone(fermoneMap, fermoneEvaporation)
		fermoneMap = leaveFermone(fermoneMap, paths, fermoneLeft)
	}

	bestPathPtr = &bestPath
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
				cityMapScaled[i][j] = distanceScaler / cityMap[i][j]
			} else {
				cityMapScaled[i][j] = 0.0
			}
		}
	}
	return
}

func calculatePathsSelectionProbabilies(cityDistanceMatrix [][]float64,
	fermoneMatrix [][]float64,
	fermoneImportance float64,
	distanceImportance float64,
	currentCity int,
	visited *set.Set[int]) (nextCityProbabilities []internal.NextCity) {

	distances := cityDistanceMatrix[currentCity]
	nextCityProbabilities = []internal.NextCity{}
	totalProbabilty := 0.0

	for i, distance := range distances {
		if distance > 0 && !visited.Contains(i) {
			cityToCalculate := i
			nextCity := internal.NextCity{CityIndex: cityToCalculate,
				Probability: calculatePathSelectionProbalitity(cityDistanceMatrix[currentCity][cityToCalculate], distanceImportance,
					fermoneMatrix[currentCity][cityToCalculate], fermoneImportance)}
			nextCityProbabilities = append(nextCityProbabilities, nextCity)
			totalProbabilty += nextCity.Probability
		}
	}

	if len(nextCityProbabilities) > 0 {
		nextCityProbabilities[0].Probability /= totalProbabilty
		for i := 1; i < len(nextCityProbabilities); i++ {
			nextCityProbabilities[i].Probability /= totalProbabilty
			nextCityProbabilities[i].Probability += nextCityProbabilities[i-1].Probability
		}
	}
	return
}

func calculatePathSelectionProbalitity(distance float64, distanceImporance float64, fermone float64, fermoneImportance float64) float64 {
	return math.Pow(distance, distanceImporance) * math.Pow(fermone, fermoneImportance)
}

func selectNextCity(cityProbablities []internal.NextCity, selection float64) int {
	for i := 0; i < len(cityProbablities); i++ {
		if selection <= cityProbablities[i].Probability {
			return cityProbablities[i].CityIndex
		}
	}

	if len(cityProbablities) > 0 {
		return cityProbablities[len(cityProbablities)-1].CityIndex
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

func leaveFermone(fermoneMatrix [][]float64, paths []internal.Path, fermoneIncrease float64) [][]float64 {
	for _, path := range paths {
		leftFermone := float64(fermoneIncrease) / float64(path.Distance)
		for i := 1; i < len(path.CitySequence); i++ {
			fermoneMatrix[path.CitySequence[i-1]][path.CitySequence[i]] += leftFermone
			fermoneMatrix[path.CitySequence[i]][path.CitySequence[i-1]] += leftFermone
		}
		fermoneMatrix[path.CitySequence[len(path.CitySequence)-1]][path.CitySequence[0]] += leftFermone
		fermoneMatrix[path.CitySequence[0]][path.CitySequence[len(path.CitySequence)-1]] += leftFermone
	}
	return fermoneMatrix
}
