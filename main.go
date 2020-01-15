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
	POPULATION_SIZE = 10
	EPOCH_COUNT     = 500000
	SELECTION_LEVEL = 0.6
	LINE_VALUE      = 0.35 //0.35 //интенсивность линии

)

var (
	LINE_COUNT     = 960 * 6
	MUTATION_LEVEL = float32(0.0005)
	BASE_LENGTH    = float32(18.)
)

func main() {
	//открыть целевое изображение
	targetImg, _ := NewImgFloat32(LoadImg("marlin.jpg"))
	W, H := targetImg.Width(), targetImg.Height()
	LINE_COUNT = int(float32(W*H)/BASE_LENGTH/2./LINE_VALUE) / 1024 * 1024

	//инициализация целевого изображения на GPU
	cuda.InitImageOnCuda(targetImg, LINE_COUNT, LINE_VALUE)
	defer cuda.DestroyImageOnCuda()

	//Создать популяцию
	m_population := MakePopulation()

	//Заполнить популяцию случайными линиями
	for i := 0; i < POPULATION_SIZE; i++ {
		m_population.AddIndivid(NewIndividGenerate(uint(LINE_COUNT), W, H, BASE_LENGTH))
	}

	//оценить популяцию
	m_population.UpdateEstimate()

	for epoch := 0; epoch < EPOCH_COUNT; epoch++ {
		//селекционный отбор в популяции
		m_population.KillFewWorseIndivids(SELECTION_LEVEL)
		//воспроизведение популяции. скрещевание
		m_population.ReproduceTo(POPULATION_SIZE, MUTATION_LEVEL)
		//оценить популяцию
		m_population.UpdateEstimate()

		//обработка промежуточных результатов
		bestIndivid, bestEstimate := m_population.BestIndivide()
		fmt.Println(
			"Epoch: ", epoch,
			"Err:", bestEstimate,
			"mutation:", MUTATION_LEVEL,
			"population_Size:", m_population.CountOfIndivids(),
			"line count", LINE_COUNT,
		)
		if epoch%100 == 0 {
			result := drawIndivide(W, H, bestIndivid)
			name := fmt.Sprintf("out/%d.jpg", epoch)
			//name := "out/out.jpg"
			SaveImg(result.Image(), name)
			SaveImg(result.Image(), "out.jpg")
		}

	}

	resultImg, _ := NewImgFloat32Blank(W, H)

	SaveImg(resultImg.Image(), "out.jpeg")
}

func drawIndivide(w, h uint, ind *individ) *ImgFloat32 {
	return cuda.DrawImageOnCuda(ind.lines)
}

type individ struct {
	lines *LineFloatArray
}

func NewIndividGenerate(count, w, h uint, base_length float32) *individ {
	i := new(individ)
	i.lines = GenerateRandomLineFloatArray(int(w), int(h), int(count), base_length)
	return i
}

func ReproducIndivid(parent1, parent2 *individ) *individ {
	i := new(individ)
	i.lines = ReproduceLineFloatArray(parent1.lines, parent2.lines)
	return i
}

func (i *individ) Estimate() float32 {
	return cuda.EstimateLinesOnCuda(i.lines)
}

type population struct {
	individs map[*individ]float32
}

func MakePopulation() population {
	var p population
	p.individs = make(map[*individ]float32)
	return p
}

func (p *population) AddIndivid(i *individ) {
	if i == nil {
		panic("individ is nil")
	}
	p.individs[i] = float32(math.Inf(1))
}

func (p *population) UpdateEstimate() {
	for individ, _ := range p.individs {
		p.individs[individ] = individ.Estimate()
	}
}

func (p *population) BestIndivide() (*individ, float32) {
	var result *individ
	resultEstimate := float32(math.Inf(1))
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
	resultEstimate := float32(0.)
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
	parent_popuplation := MakePopulation()
	parent_popuplation.individs = p.individs

	newPopulation := MakePopulation()

	for newPopulation.CountOfIndivids() < number-parent_popuplation.CountOfIndivids() {
		parent_individ_1 := parent_popuplation.RandomIndivid()
		parent_popuplation.killIndivid(parent_individ_1)
		parent_individ_2 := parent_popuplation.RandomIndivid()
		parent_popuplation.AddIndivid(parent_individ_1)

		child_individ := ReproducIndivid(parent_individ_1, parent_individ_2)

		newPopulation.AddIndivid(child_individ)
	}
	newPopulation.Mutation(mutation)

	for individ, _ := range newPopulation.individs {
		p.AddIndivid(individ)
	}
}
func (p *population) Mutation(mutation float32) {
	for individ, _ := range p.individs {
		individ.lines.Mutate(mutation)
	}
}
