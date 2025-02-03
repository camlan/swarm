package cuda

import (
	"testing"

	"github.com/camlan/swarm/ant/file"
	"github.com/camlan/swarm/ant/internal"
)

func TestFindShortestPathSanity(t *testing.T) {
	// given
	distanceMap := [][]float64{
		{0.0, 3.0, 5.0, 4.0},
		{3.0, 0.0, 3.162277, 5.0},
		{5.0, 3.162277, 0.0, 4.1231055},
		{4.0, 5.0, 4.1231055, 0.0},
	}
	expectedDistance := 14.2853825

	// when
	foundPath := FindShortestPath(distanceMap)

	// then
	if foundPath.Distance != expectedDistance {
		t.Errorf("Expected distance %v, got:%v", expectedDistance, foundPath.Distance)
	}
}

func TestFindShortestPathToFix(t *testing.T) {
	// given
	//distancesToVerify := internal.DistancesToVerifyFeed()
	distancesToVerify := make(map[float64][][]float64)
	distancesToVerify[1068] = file.GenerateCityMatrix("..\\data\\wg59_dist.txt")

	for expected, distanceMap := range distancesToVerify {
		//when
		foundPath := FindShortestPath(distanceMap)
		//then
		if !internal.IsInRange(expected, foundPath.Distance) {
			t.Errorf("Expected distance %v, got:%v, for sequence: ", expected, foundPath.Distance)
		}
	}
}
