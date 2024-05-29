package main

/*
#cgo LDFLAGS:  -L . -lexamplestatic -lcuda -lcudart -lm
#include <stdlib.h>
int add_wrapper(double *a, double *b, size_t len);
*/
import "C"
import "fmt"

//  -Xcompiler="/GS-"
func test(a, b []float64) error {
	if res := C.add_wrapper((*C.double)(&a[0]), (*C.double)(&b[0]), C.size_t(len(a))); res != 0 {
		return fmt.Errorf("got bad error code from C.add %d", int(res))
	}
	return nil
}

func main() {
	a := []float64{-1.0, 2.5, 4, 0, 5, 3, 6, 2, 1}
	b := []float64{3.2, 0.5, 2, 3, 4, 5, 4, 7, 2}
	test(a, b)
	fmt.Println(a)
}
