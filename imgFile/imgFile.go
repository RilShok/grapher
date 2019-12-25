package imgFile

import (
	_ "errors"
	"image"
	"image/color"
	"image/jpeg"

	//"image/png"
	"log"
	"math"
	"os"
)

func LoadImg(path string) image.Image {
	f, err := os.Open(path)
	if err != nil {
		log.Fatalln(err)
	}
	defer f.Close()

	img, _, err := image.Decode(f)
	if err != nil {
		log.Fatalln(err)
	}
	return img
}

func SaveImg(img image.Image, path string) {
	f, err := os.Create(path)
	if err != nil {
		log.Fatalln(err)
	}
	defer f.Close()
	jpeg.Encode(f, img, nil)
	if err != nil {
		log.Fatalln(err)
	}
}

type ImgFloat32 struct {
	data []float32
	w, h uint
}

func (i *ImgFloat32) Data() *float32 {
	return &i.data[0]
}

func (i *ImgFloat32) CompareError(obj *ImgFloat32) float32 {
	ans := float32(0.)
	for k, v := range i.data {
		ans += math.Float32frombits(math.Float32bits(v-obj.data[k]) &^ (1 << 31))
	}
	return ans
}

func NewImgFloat32FromData(width, hight uint, data []float32) *ImgFloat32 {
	ans := new(ImgFloat32)
	ans.w = width
	ans.h = hight
	ans.data = make([]float32, ans.w*ans.h)
	copy(ans.data, data)
	return ans
}
func NewImgFloat32Blank(width, hight uint) (*ImgFloat32, error) {
	ans := new(ImgFloat32)
	ans.w = width
	ans.h = hight
	ans.data = make([]float32, ans.w*ans.h)
	for x := uint(0); x < ans.w; x++ {
		for y := uint(0); y < ans.h; y++ {
			ans.data[y*ans.w+x] = 1.
		}
	}
	return ans, nil
}

func NewImgFloat32(img image.Image) (*ImgFloat32, error) {

	ans := new(ImgFloat32)
	bounds := img.Bounds()
	ans.w = uint(bounds.Dx())
	ans.h = uint(bounds.Dy())
	ans.data = make([]float32, ans.w*ans.h)
	for x := uint(0); x < ans.w; x++ {
		for y := uint(0); y < ans.h; y++ {
			r, g, b, _ := img.At(int(x), int(y)).RGBA()

			ans.data[y*ans.w+x] = (.2126*float32(r) + .7152*float32(g) + .0722*float32(b)) / float32(math.MaxUint16)
			// if (ans.data[y*ans.w+x]>0.8){
			// 	log.Fatalln("!!!")
			// }
		}
	}
	return ans, nil
}

func (i *ImgFloat32) Image() image.Image {
	img := image.NewRGBA(image.Rect(0, 0, int(i.w), int(i.h)))

	for x := uint(0); x < i.w; x++ {
		for y := uint(0); y < i.h; y++ {
			img.Set(int(x), int(y), i.At(x, y))
		}
	}
	return img
}

func (i *ImgFloat32) At(x, y uint) color.Color {
	v := i.data[y%i.h*i.w+x%i.w] * 255.
	c := uint8(v)
	if v > 255. {
		c = 255
	}
	//log.Println(i.data[y*i.w+x], v)
	return color.RGBA{R: c, G: c, B: c, A: 255}
}
func (i *ImgFloat32) Get(x, y uint) float32 {
	return i.data[y%i.h*i.w+x%i.w]
}
func (i *ImgFloat32) Height() uint {
	return i.h
}
func (i *ImgFloat32) Width() uint {
	return i.w
}

func (i *ImgFloat32) Set(x, y uint, value float32) {
	i.data[y%i.h*i.w+x%i.w] = value
}

func (i *ImgFloat32) UpdateData(data []float32) {
	i.data = data
}
