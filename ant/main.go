package main

import (
	"fmt"
	"os"

	// "github.com/camlan/swarm/ant/basic"
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
	path := cuda.FindShortestPath(cityMatrix)
	fmt.Printf("%+v", *path)
}
