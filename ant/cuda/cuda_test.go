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

// func TestMovingAnts(t *testing.T) {
// 	// given
// 	numberOfCities := 4
// 	cityMapScaled := []float64{0, 0.16666666666666666, 0.1, 0.125, 0.16666666666666666, 0, 0.15811391601684482, 0.1, 0.1, 0.15811391601684482, 0, 0.12126781621280366, 0.125, 0.1, 0.12126781621280366, 0}
// 	distanceMap1D := []float64{0, 3, 5, 4, 3, 0, 3.162277, 5, 5, 3.162277, 0, 4.1231055, 4, 5, 4.1231055, 0}
// 	fermoneMap1D := []float64{0.12, 0.6100113805143124, 0.9366756357308652, 0.6100113805143124, 0.6100113805143124, 0.12, 0.5278712865431551, 0.9376086983754203, 0.5278712865431551, 0.6100113805143124, 0.12, 0.5288043491877101, 1.0178826670574674, 0.9376086983754203, 1.0188157297020226, 0.12}
// 	//fermoneMap1D_X := []float64{0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2}
// 	fermoneImportance := 1.3
// 	distanceImportance := 1.0

// 	//expectedDistance := 14.2853825

// 	// when
// 	//moveAnts(numberOfCities, cityMapScaled, distanceMap1D, fermoneMap1D, fermoneImportance, distanceImportance)
// 	//moveAnts(numberOfCities, cityMapScaled, distanceMap1D, fermoneMap1D, fermoneImportance, distanceImportance)
// 	// moveAnts(numberOfCities, cityMapScaled, distanceMap1D, fermoneMap1D, fermoneImportance, distanceImportance)
// 	var distance []float64
// 	var citySequence1D []int32
// 	distance, citySequence1D = moveAnts(numberOfCities, cityMapScaled, distanceMap1D, fermoneMap1D, fermoneImportance, distanceImportance)
// 	fmt.Printf("1 D: %v, C: %v\n", distance, citySequence1D)

// 	distance, citySequence1D = moveAnts(numberOfCities, cityMapScaled, distanceMap1D, fermoneMap1D, fermoneImportance, distanceImportance)
// 	fmt.Printf("2 D: %v, C: %v\n", distance, citySequence1D)

// 	distance, citySequence1D = moveAnts(numberOfCities, cityMapScaled, distanceMap1D, fermoneMap1D, fermoneImportance, distanceImportance)
// 	fmt.Printf("3 D: %v, C: %v\n", distance, citySequence1D)

// 	// for i := 0; i < 2; i++ {
// 	// 	// distance, citySequence1D = moveAnts(numberOfCities, cityMapScaled, distanceMap1D, fermoneMap1D_X, fermoneImportance, distanceImportance)
// 	// 	//distance, citySequence1D = moveAnts(numberOfCities, cityMapScaled, distanceMap1D, fermoneMap1D_X, fermoneImportance, distanceImportance)
// 	// 	//distance, citySequence1D = moveAnts(numberOfCities, cityMapScaled, distanceMap1D, fermoneMap1D_X, fermoneImportance, distanceImportance)
// 	// 	distance, citySequence1D = moveAnts(numberOfCities, cityMapScaled, distanceMap1D, fermoneMap1D_X, fermoneImportance, distanceImportance)
// 	// 	//fmt.Printf("%d, D: %v, C: %v",i, distance, citySequence1D)
// 	// }

// 	// then
// 	if distance[0] < 5000 {
// 		t.Errorf("Got invalid distances: %v and cities: %v", distance, citySequence1D)
// 	} else {
// 		fmt.Printf("D: %v, C: %v", distance, citySequence1D)
// 	}
// }
