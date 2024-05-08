package main

import (
	"testing"

	"github.com/camlan/swarm/ant/basic"
	"github.com/camlan/swarm/ant/file"
)

func BenchmarkFindShortestPathBasic(b *testing.B) {
	cityMatrix := file.GenerateCityMatrix("data\\uk12_dist.txt")
	basic.FindShortestPath(cityMatrix)
}
