package floatLine

import (
	//	"fmt"
	. "grapher/imgFile"
	"math"
	"math/rand"
)

type LineFloat struct {
	x, y          uint
	length, angle float64
	value         float32
}

func (l *LineFloat) GET() (x, y uint, length, angle float64, value float32) {
	return l.x, l.y, l.length, l.angle, l.value
}
func (l *LineFloat) Mutate(w, h uint, mutation float32) {
	switch rand.Intn(3) {
	case 0:
		stepX := rand.Intn(int(float32(w) * mutation))
		stepY := rand.Intn(int(float32(h) * mutation))
		if rand.Intn(2)%2 == 0 {
			stepX *= -1
		}
		if rand.Intn(2)%2 == 0 {
			stepY *= -1
		}
		l.x = uint(int(l.x) + stepX)
		l.y = uint(int(l.y) + stepY)
	case 1:
		rotate := rand.Float64() * 2 * math.Pi * float64(mutation)
		if rand.Intn(2)%2 == 0 {
			rotate *= -1.
		}
		lengthStep := rand.Float64() * l.length * float64(mutation)
		if rand.Intn(2)%2 == 0 {
			lengthStep *= -1.
		}
		l.length += lengthStep
		l.angle += rotate
	case 2:
		newValue := rand.Float32() * mutation
		if rand.Intn(2)%2 == 0 {
			newValue *= -1.
		}
		newValue += l.value
		if 0. < newValue && newValue < 1. {
			l.value = newValue
		}
	}
}
func GenerateLine(w, h uint, minLength, maxLength float64) LineFloat {
	var l LineFloat
	l.length = minLength + rand.Float64()*(maxLength-minLength)
	l.x = uint(rand.Intn(int(w)-int(2.*l.length+1.)) + int(l.length))
	l.y = uint(rand.Intn(int(h)-int(2.*l.length+1.)) + int(l.length))
	l.angle = rand.Float64() * 2. * math.Pi
	l.value = rand.Float32()
	return l
}
func (l *LineFloat) SetValue(v float32) {
	l.value = v
}
func (l *LineFloat) Value() float32 {
	return l.value
}
func (l *LineFloat) Get(k float64) (x, y uint) {
	x = l.x + uint(math.Cos(l.angle)*k*l.length)
	y = l.y - uint(math.Sin(l.angle)*k*l.length)
	return
}
func (l *LineFloat) Length() float64 {
	return l.length
}

func DrawLine(img *ImgFloat32, line LineFloat, coef float32) {
	lastX, lastY := line.Get(0.)
	for k := 0.; k <= 1; k += 1. / (1.3 * line.length) {
		x, y := line.Get(k)
		if x != lastX && y != lastY {
			img.Set(x, y, img.Get(x, y)*coef)
		}
	}
}

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
	mutation := float32(0.2) //0.1
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

// func (p *population) Mutation(mutation float32) {
// 	w, h := p.img.Width(), p.img.Height()
// 	for individ, _ := range p.individs {
// 		count_lines := len(individ.lines)
// 		random_count_lines := int(mutation * float32(count_lines))
// 		for i := 0; i < random_count_lines; i++ {
// 			//if rand.Float32() < mutation {
// 			idx := rand.Intn(count_lines)
// 			if rand.Intn(2) > 0 {
// 				individ.lines[idx].Mutate(w, h, 0.1)
// 			} else {
// 				individ.lines[idx] = GenerateLine(w, h, 5, 15)
// 			}

// 		}
// 	}
// }

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
