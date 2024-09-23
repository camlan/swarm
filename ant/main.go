package main

import (
	"fmt"
	"os"

	"github.com/camlan/swarm/ant/basic"
	"github.com/camlan/swarm/ant/cuda"
	"github.com/camlan/swarm/ant/file"
)

func main() {
	args := os.Args[1:]
	var cityMatrix [][]float64
	if len(args) == 0 {
		cityMatrix = file.GenerateCityMatrix("data\\grid04_dist.txt")
	} else {
		cityMatrix = file.GenerateCityMatrix(args[0])
	}
	fmt.Printf("City Matrix: %v\n", cityMatrix)
	path := basic.FindShortestPathRouties(cityMatrix)
	fmt.Printf("Basic: %+v", *path)
	path2 := cuda.FindShortestPath(cityMatrix)
	fmt.Printf("Cuda: %+v", *path2)
}
