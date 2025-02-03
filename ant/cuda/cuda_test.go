package cuda

import (
	"testing"
)

func TestFindShortestPath(t *testing.T) {

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
