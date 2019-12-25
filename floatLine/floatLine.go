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
	//x->w
	stepX := rand.Intn(int(float32(w) * mutation))
	stepY := rand.Intn(int(float32(h) * mutation))
	if rand.Intn(2)%2 == 0 {
		stepX *= -1
	}
	if rand.Intn(2)%2 == 0 {
		stepY *= -1
	}
	rotate := rand.Float64() * 2 * math.Pi * float64(mutation)
	if rand.Intn(2)%2 == 0 {
		rotate *= -1.
	}
	lengthStep := rand.Float64() * float64(mutation)
	if rand.Intn(2)%2 == 0 {
		lengthStep *= -1.
	}
	l.length += lengthStep
	l.angle += rotate
	l.x = uint(int(l.x) + stepX)
	l.y = uint(int(l.y) + stepY)

	newValue := rand.Float32() * mutation
	if rand.Intn(2)%2 == 0 {
		newValue *= -1.
	}
	newValue += l.value
	if 0. < newValue && newValue < 1. {
		l.value = newValue
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

func AvrErr(img, delta *ImgFloat32, line LineFloat, coef float32) float64 {
	ans := 0.
	count := 0
	for k := 0.; k <= 1; k += 1. / (1.5 * line.length) {
		x, y := line.Get(k)
		delta := float64(delta.Get(x, y)*coef - img.Get(x, y))

		ans += math.Exp(math.Pow(1.-delta, 2)) / (k + 0.01)
		count++
	}
	return ans / float64(count)
}

func AvrErr2(img *ImgFloat32, line LineFloat, coef float32) float64 {
	ans := 0.
	for k := 0.; k <= 1; k += 1. / (1.5 * line.length) {
		x, y := line.Get(k)
		delta := float64(coef - img.Get(x, y))
		ans += math.Exp(1. - delta)
	}
	return ans
}
