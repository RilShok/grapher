package floatLine

import (
	"math"
	"math/rand"
)

type LineFloatArray struct {
	x, y          []int32
	length, angle []float32
	size          int
	baseLength    float32
	w, h          int
}

func initLineFloatArray(w, h, size int, length float32) *LineFloatArray {
	a := new(LineFloatArray)
	a.size = size
	a.baseLength = length
	a.w, a.h = w, h
	a.x = make([]int32, size)
	a.y = make([]int32, size)
	a.length = make([]float32, size)
	a.angle = make([]float32, size)
	return a
}

func GenerateRandomLineFloatArray(w, h, size int, length float32) *LineFloatArray {
	a := initLineFloatArray(w, h, size, length)
	for i := 0; i < size; i++ {
		a.RandomizeLine(i)
	}
	return a
}

func ReproduceLineFloatArray(parent1, parent2 *LineFloatArray) *LineFloatArray {
	if parent1 == nil || parent2 == nil {
		panic("parent is nil")
	}
	a := initLineFloatArray(parent1.w, parent1.h, parent1.size, parent1.baseLength)
	if a.size != parent2.size || a.h != parent2.h || a.w != parent2.w {
		panic("bad reproduce")
	}
	for i := 0; i < a.size; i++ {
		if rand.Intn(2) == 0 {
			parent1, parent2 = parent2, parent1
		}
		a.x[i] = parent1.x[i]
		a.y[i] = parent1.y[i]
		a.length[i] = parent1.length[i]
		a.angle[i] = parent1.angle[i]
	}
	return a
}
func (a *LineFloatArray) checkIndex(idx int) {
	if idx >= a.size || idx < 0 {
		panic("bad index")
	}
}
func (a *LineFloatArray) RandomizeLine(idx int) *LineFloatArray {
	a.checkIndex(idx)
	a.length[idx] = a.baseLength + rand.Float32()*(a.baseLength*0.5) - a.baseLength*0.25 // 0.2
	a.x[idx] = int32(rand.Intn(a.w-int(2.*a.length[idx]+1.)) + int(a.length[idx]))
	a.y[idx] = int32(rand.Intn(a.h-int(2.*a.length[idx]+1.)) + int(a.length[idx]))
	a.angle[idx] = rand.Float32() * 2. * math.Pi
	return a
}

func (a *LineFloatArray) MutateLine(idx int) *LineFloatArray {
	a.checkIndex(idx)
	mutation := float32(0.3) //0.1
	a.x[idx] += int32(rand.Intn(int(float32(a.w)*mutation)) * (rand.Intn(2)*2 - 1))
	a.y[idx] += int32(rand.Intn(int(float32(a.h)*mutation)) * (rand.Intn(2)*2 - 1))
	a.angle[idx] += 2. * math.Pi * mutation * (rand.Float32()*2. - 1.)
	a.length[idx] += a.length[idx] * mutation * (rand.Float32()*2. - 1.)
	return a
}
func (a *LineFloatArray) Mutate(probability float32) *LineFloatArray {
	number_mutable_lines := int(float32(a.size) * probability)
	for i := 0; i < number_mutable_lines; i++ {
		if rand.Intn(2) == 0 {
			a.MutateLine(rand.Intn(a.size))
		} else {
			a.RandomizeLine(rand.Intn(a.size))

		}
	}
	return a
}


func (a *LineFloatArray) Size() int {
	return a.size
}

func (a *LineFloatArray) X() *int32 {
	return &a.x[0]
}

func (a *LineFloatArray) Y() *int32 {
	return &a.y[0]
}

func (a *LineFloatArray) Length() *float32 {
	return &a.length[0]
}

func (a *LineFloatArray) Angle() *float32 {
	return &a.angle[0]
}
