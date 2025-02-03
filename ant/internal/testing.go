package internal

import "github.com/camlan/swarm/ant/file"

func DistancesToVerifyFeed() (distances map[float64][][]float64) {
	distances = make(map[float64][][]float64)

	distances[14.2853825] = file.GenerateCityMatrix("..\\data\\grid04_dist.txt")
	distances[820] = file.GenerateCityMatrix("..\\data\\wg22_dist.txt")
	distances[1068] = file.GenerateCityMatrix("..\\data\\wg59_dist.txt")

	return
}

func IsInRange(expected float64, result float64) bool {
	return expected <= result+result*0.15 && expected >= result-result*0.15
}
