package main

import (
	"fmt"
	"testing"

	"github.com/camlan/swarm/ant/basic"
	"github.com/camlan/swarm/ant/file"
)

var files = []string{
	"data\\uk12_dist.txt",
	"data\\ha30_dist.txt",
	"data\\grid04_dist.txt",
	"data\\kn57_dist.txt",
	"data\\lau15_dist.txt",
	//	"data\\sgb128_dist.txt",
	"data\\sp11_dist.txt",
	"data\\uk12_dist.txt",
	"data\\wg22_dist.txt",
	"data\\wg59_dist.txt",
}

func BenchmarkFindShortestPathBasic(b *testing.B) {
	for _, f := range files {
		b.Run(fmt.Sprintf("file:%s", f), func(b *testing.B) {
			cityMatrix := file.GenerateCityMatrix(f)
			basic.FindShortestPath(cityMatrix)
		})
	}
}
