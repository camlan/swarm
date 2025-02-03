package basic

import (
	"testing"

	"github.com/camlan/swarm/ant/internal"
)

func TestFindShortestPath(t *testing.T) {
	// given
	distancesToVerify := internal.DistancesToVerifyFeed()

	for expected, distanceMap := range distancesToVerify {
		//when
		foundPath := FindShortestPath(distanceMap)
		//then
		if !internal.IsInRange(expected, foundPath.Distance) {
			t.Errorf("Expected distance %v, got:%v", expected, foundPath.Distance)
		}

	}
}

func TestFindShortestPathRoutines(t *testing.T) {
	// given
	distancesToVerify := internal.DistancesToVerifyFeed()

	for expected, distanceMap := range distancesToVerify {
		//when
		foundPath := FindShortestPathRouties(distanceMap)
		//then
		if !internal.IsInRange(expected, foundPath.Distance) {
			t.Errorf("Expected distance %v, got:%v", expected, foundPath.Distance)
		}
	}
}
