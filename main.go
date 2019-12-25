package main

import (
	"fmt"
	"grapher/cuda"
	. "grapher/floatLine"
	. "grapher/imgFile"
	"math"
	"math/rand"
)

const (
	LINE_COUNT      = 5000
	POPULATION_SIZE = 50
	EPOCH_COUNT     = 300000
	SELECTION_LEVEL = 0.85
)

var (
	MUTATION_LEVEL = float32(0.005)
)

func main() {
	fmt.Println("hello")

	floatImg, _ := NewImgFloat32(LoadImg("in.jpg"))
	cuda.InitImageOnCuda(floatImg, LINE_COUNT)
	defer cuda.DestroyImageOnCuda()
	W, H := floatImg.Width(), floatImg.Height()
	//SaveImg(floatImg.Image(), "gray.jpeg")

	m_population := MakePopulation(floatImg)

	for i := 0; i < POPULATION_SIZE; i++ {
		m_population.AddIndivid(NewIndividGenerate(LINE_COUNT, W, H))
	}
	m_population.UpdateEstimate()

	for epoch := 0; epoch < EPOCH_COUNT; epoch++ {

		m_population.KillFewWorseIndivids(SELECTION_LEVEL)
		m_population.ReproduceTo(POPULATION_SIZE, MUTATION_LEVEL)
		m_population.UpdateEstimate()
		bestIndivid, bestEstimate := m_population.BestIndivide()
		fmt.Println("Epoch: ", epoch, "Err:", bestEstimate, "mutation:", MUTATION_LEVEL, "population_Size:", m_population.CountOfIndivids())
		if epoch%10 == 0 {
			result := drawIndivide(W, H, bestIndivid)
			//name := fmt.Sprintf("out/%d.jpeg", epoch)
			name := "out/out.jpeg"
			SaveImg(result.Image(), name)
			//MUTATION_LEVEL = MUTATION_LEVEL * 0.99
		}

	}

	resultImg, _ := NewImgFloat32Blank(W, H)

	SaveImg(resultImg.Image(), "out.jpeg")
}

func drawIndivide(w, h uint, ind *individ) *ImgFloat32 {
	return cuda.DrawImageOnCuda(ind.lines)

	//!!! result, _ := NewImgFloat32Blank(w, h)
	//!!! for _, line := range ind.lines {
	//!!! 	DrawLine(result, line, line.Value())
	//!!! }
	//!!! return result
}

type individ struct {
	lines []LineFloat
}

func NewIndivid() *individ {
	i := new(individ)
	i.lines = make([]LineFloat, 0)
	return i
}

func NewIndividGenerate(count, w, h uint) *individ {
	i := NewIndivid()
	for k := uint(0); k < count; k++ {
		line := GenerateLine(w, h, 5, 15)
		line.SetValue(rand.Float32())
		i.lines = append(i.lines, line)
	}
	return i
}
func (i *individ) addLine(line LineFloat) {
	i.lines = append(i.lines, line)
}
func (i *individ) Estimate(img *ImgFloat32) float64 {
	return float64(cuda.EstimateLinesOnCuda(i.lines))
	//!!!	return float64(img.CompareError(drawIndivide(img.Width(), img.Height(), i)))
}

type population struct {
	img      *ImgFloat32
	individs map[*individ]float64
}

func MakePopulation(img *ImgFloat32) population {
	var p population
	p.img = img
	p.individs = make(map[*individ]float64)
	return p
}

func (p *population) AddIndivid(i *individ) {
	if i == nil {
		panic("individ is nil")
	}
	p.individs[i] = math.Inf(1)
}

func (p *population) UpdateEstimate() {
	for individ, _ := range p.individs {
		p.individs[individ] = individ.Estimate(p.img)
	}
}

func (p *population) BestIndivide() (*individ, float64) {
	var result *individ
	resultEstimate := math.Inf(1)
	for individ, estimate := range p.individs {
		if estimate < resultEstimate {
			result = individ
			resultEstimate = estimate
		}
	}
	return result, resultEstimate
}

func (p *population) WorseIndivide() *individ {
	var result *individ
	resultEstimate := 0.
	for individ, estimate := range p.individs {
		if estimate > resultEstimate {
			result = individ
			resultEstimate = estimate
		}
	}
	return result
}

func (p *population) killIndivid(i *individ) {
	_, ok := p.individs[i]
	if ok {
		delete(p.individs, i)
	}
}

func (p *population) CountOfIndivids() int {
	return len(p.individs)
}

func (p *population) KillFewWorseIndivids(proportion float32) {
	count := int(float32(p.CountOfIndivids()) * proportion)
	for i := 0; i < count && p.CountOfIndivids() > 2; i++ {
		p.killIndivid(p.WorseIndivide())
	}
}

func (p *population) RandomIndivid() *individ {
	number := rand.Intn(p.CountOfIndivids())
	i := 0
	for individ, _ := range p.individs {
		if i == number {
			return individ
		}
		i++
	}
	panic("not found individ")
}

func (p *population) ReproduceTo(number int, mutation float32) {
	//w, h := p.img.Width(), p.img.Height()
	parent_individs := p.individs
	rand_individ := func() *individ {
		number := rand.Intn(len(parent_individs))
		i := 0
		for individ, _ := range parent_individs {
			if i == number {
				return individ
			}
			i++
		}
		panic("not found individ")
	}
	newPopulation := MakePopulation(p.img)
	for newPopulation.CountOfIndivids() < number-p.CountOfIndivids() {
		parent_individ_1 := rand_individ()
		parent_individ_2 := rand_individ()
		child_individ := NewIndivid()
		//скрещивание. кроссинговер равномерный
		for i := 0; i < len(parent_individ_1.lines) && i < len(parent_individ_2.lines); i++ {
			if rand.Intn(2) == 0 {
				child_individ.addLine(parent_individ_1.lines[i])
			} else {
				child_individ.addLine(parent_individ_2.lines[i])
			}

		}
		newPopulation.AddIndivid(child_individ)
	}
	newPopulation.Mutation(mutation)
	for individ, _ := range newPopulation.individs {
		p.AddIndivid(individ)
	}
}
func (p *population) Mutation(mutation float32) {
	w, h := p.img.Width(), p.img.Height()
	for individ, _ := range p.individs {
		count_lines := len(individ.lines)
		random_count_lines := int(mutation * float32(count_lines))
		for i := 0; i < random_count_lines; i++ {
			//if rand.Float32() < mutation {
			idx := rand.Intn(count_lines)
			individ.lines[idx].Mutate(w, h, 0.1)
			//}

		}
	}
}
