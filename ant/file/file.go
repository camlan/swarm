package file

import (
	"bufio"
	"log"
	"os"
	"strconv"
	"strings"
)

func GenerateCityMatrix(cityMatrixFile string) (cityMatrix [][]float64) {

	file, err := os.Open(cityMatrixFile)
	if err != nil {
		log.Fatalf("Error opening file: %v", err)
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)

	cityMatrix = [][]float64{}
	city := 0
	for scanner.Scan() {
		line := scanner.Text()
		distances := strings.Fields(line)
		cityMatrix = append(cityMatrix, []float64{})
		for _, distanceStr := range distances {
			distance, _ := strconv.ParseFloat(distanceStr, 64)
			cityMatrix[city] = append(cityMatrix[city], distance)
		}
		city++
	}
	return
}
