package main

import (
	"flag"
	"fmt"
	"math"
	"math/rand"
	"os"
	"sort"

	"github.com/camlan/swarm/ant/basic"
	"github.com/camlan/swarm/ant/file"
	"github.com/hashicorp/go-set/v2"
)

type NextCity struct {
	cityIndex   int
	probability float64
}

type Path struct {
	citySequence []int
	distance     int
}

func main() {
	args := os.Args[1:]
	var cityMatrix [][]float64
	if len(args) == 0 {
		cityMatrix = file.GenerateCityMatrix("data\\grid04_dist.txt")
	} else {
		cityMatrix = file.GenerateCityMatrix(args[0])
	}
	fmt.Printf("%v\n", cityMatrix)
	path := basic.FindShortestPath(cityMatrix)

	fmt.Printf("%+v", path)
}

func oldmain() {

	// init
	// numberOfCitiesParam := flag.Int("c", 10, "number of cities")
	// numberOfCities := (*numberOfCitiesParam)
	// cityMap := generateCityMap(numberOfCities)
	numberOfCities := 5
	cityMap := generateCityMapFixed3()
	for _, distances := range cityMap {
		fmt.Printf("%v\n", distances)
	}

	// params
	initFermoneParam := flag.Float64("fc", 0.2, "initial fermone coeficient")
	initFermone := (*initFermoneParam)
	fermoneImportanceParam := flag.Float64("fi", 1.2, "fermone importance")
	fermoneImportance := *fermoneImportanceParam

	distanceScalerParam := flag.Int64("ds", 7, "distance scaler")
	distanceScaler := (*distanceScalerParam)
	distanceImportanceParam := flag.Float64("di", 1, "distance importance")
	distanceImportance := (*distanceImportanceParam)

	fermoneEvaporationParam := flag.Float64("fe", 0.64, "fermone evaporation")
	fermoneEvaporation := (*fermoneEvaporationParam)

	fermoneLeftParam := flag.Int("fl", 20, "fermone left")
	fermoneLeft := (*fermoneLeftParam)

	iterationsParam := flag.Int("i", 5, "iterations")
	iterations := (*iterationsParam)

	cityDistanceMatrixScaled := generateScaledCityDistanceMatrix(cityMap, float64(distanceScaler))
	for _, distancesScaled := range cityDistanceMatrixScaled {
		fmt.Printf("%v\n", distancesScaled)
	}

	fermoneMatrix := generateFermoneMatrix(len(cityDistanceMatrixScaled), initFermone)

	shortestPath := Path{distance: math.MaxInt, citySequence: []int{}}
	for iteration := 0; iteration < iterations; iteration++ {
		paths := []Path{}
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
			path := Path{distance: calculatePathDistance(citySequence, cityMap), citySequence: citySequence}
			fmt.Printf("Ant: %d traveled: %v with distance: %d\n", ant, path.citySequence, path.distance)
			paths = append(paths, path)
		}
		sort.Slice(paths, func(i, j int) bool {
			return paths[i].distance < paths[j].distance
		})
		fmt.Printf("Paths sorted by distance: %v in iteration: %d\n", paths, iteration)
		if shortestPath.distance > paths[0].distance {
			shortestPath = paths[0]
		}
		fermoneMatrix = evaporateFermone(fermoneMatrix, fermoneEvaporation)
		fermoneMatrix = leaveFermone(fermoneMatrix, paths, fermoneLeft)
	}

	fmt.Printf("Shortest path found: %v with distance: %d with %d iterations\n", shortestPath.citySequence, shortestPath.distance, iterations)
}

func randomDistance() (distance int) {
	distance = rand.Intn(20) + 10
	return
}

func generateCityMap(numberOfCities int) (cityMap [][]int) {
	cityMap = make([][]int, numberOfCities)
	for i := range cityMap {
		cityMap[i] = make([]int, numberOfCities)
	}
	for i := 0; i < numberOfCities; i++ {
		for j := i + 1; j < numberOfCities; j++ {
			distance := randomDistance()
			cityMap[i][j] = distance
			cityMap[j][i] = distance
		}
		cityMap[i][i] = 0.0
	}
	return
}

func generateCityMapFixed() (cityMap [][]int) {
	cityMap = make([][]int, 4)
	for i := range cityMap {
		cityMap[i] = make([]int, 4)
	}

	cityMap[0][0] = -1
	cityMap[0][1] = 15
	cityMap[0][2] = 7
	cityMap[0][3] = 12

	cityMap[1][0] = 15
	cityMap[1][1] = -1
	cityMap[1][2] = 5
	cityMap[1][3] = 8

	cityMap[2][0] = 7
	cityMap[2][1] = 5
	cityMap[2][2] = -1
	cityMap[2][3] = 21

	cityMap[3][0] = 12
	cityMap[3][1] = 8
	cityMap[3][2] = 21
	cityMap[3][3] = -1

	return
}

func generateCityMapFixed2() (cityMap [][]int) {
	cityMap = [][]int{
		{-1, 2, 4, 6, 8},
		{2, -1, 3, 5, 7},
		{4, 6, -1, 4, 6},
		{6, 5, 4, -1, 7},
		{8, 5, 6, 7, -1},
	}
	return
}

func generateCityMapFixed3() (cityMap [][]int) {
	cityMap = [][]int{
		{-1, 20, 40, 60, 80},
		{30, -1, 30, 50, 70},
		{40, 70, -1, 40, 60},
		{50, 50, 30, -1, 70},
		{60, 30, 40, 50, -1},
	}
	return
}

func generateCityMapFixed4() (cityMap [][]int) {
	cityMap = [][]int{
		{0, 2, 4, 6, 8},
		{0, 0, 3, 5, 7},
		{0, 6, 0, 4, 6},
		{0, 5, 4, 0, 7},
		{0, 5, 6, 7, 0},
	}
	return
}

func generateScaledCityDistanceMatrix(cityMap [][]int, distanceScaler float64) (cityMapScaled [][]float64) {
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
	return -1
}

func calculatePathDistance(citySequence []int, cityMap [][]int) int {
	distance := 0
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

func leaveFermone(fermoneMatrix [][]float64, paths []Path, fermoneIncrease int) [][]float64 {
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
