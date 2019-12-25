package cuda

/*
void initImage(float *img, int W, int H, int line_count, float line_value);
void destroyImage();
void estimate(float *result, int *x, int *y, float *length, float *angle);
void draw(float *result, int *x, int *y, float *length, float *angle);
#cgo LDFLAGS: -L. -L../cuda -ldrawLine
*/
import "C"
import (
	. "grapher/floatLine"
	. "grapher/imgFile"
)

var (
	w, h  int
	image *ImgFloat32
)

func InitImageOnCuda(img *ImgFloat32, line_count int, line_value float32) {
	w = int(img.Width())
	h = int(img.Height())
	image = img
	C.initImage((*C.float)(image.Data()), C.int(w), C.int(h), C.int(line_count), C.float(line_value))
}
func DestroyImageOnCuda() {
	C.destroyImage()
}
func EstimateLinesOnCuda(lines *LineFloatArray) (result float32) {
	result = 0.
	C.estimate(
		(*C.float)(&result),
		(*C.int)(lines.X()),
		(*C.int)(lines.Y()),
		(*C.float)(lines.Length()),
		(*C.float)(lines.Angle()),
	)
	return
}
func DrawImageOnCuda(lines *LineFloatArray) *ImgFloat32 {
	//func DrawImageOnCuda(lines []LineFloat) *ImgFloat32 {

	// lineCount := len(lines)
	// x_array := make([]int32, lineCount)
	// y_array := make([]int32, lineCount)
	// lenght_array := make([]float32, lineCount)
	// angle_array := make([]float32, lineCount)
	// value_array := make([]float32, lineCount)

	// for i, line := range lines {
	// 	//x, y uint, length, angle float64, value float32
	// 	x, y, length, angle, value := line.GET()
	// 	x_array[i] = int32(x)
	// 	y_array[i] = int32(y)
	// 	lenght_array[i] = float32(length)
	// 	angle_array[i] = float32(angle)
	// 	value_array[i] = value
	// }

	// result := make([]float32, w*h)
	// C.draw(
	// 	(*C.float)(&result[0]),
	// 	(*C.int)(&x_array[0]),
	// 	(*C.int)(&y_array[0]),
	// 	(*C.float)(&lenght_array[0]),
	// 	(*C.float)(&angle_array[0]),
	// 	(*C.float)(&value_array[0]),
	// )
	// return NewImgFloat32FromData(uint(w), uint(h), result)

	result := make([]float32, w*h)
	C.draw(
		(*C.float)(&result[0]),
		(*C.int)(lines.X()),
		(*C.int)(lines.Y()),
		(*C.float)(lines.Length()),
		(*C.float)(lines.Angle()),
	)
	return NewImgFloat32FromData(uint(w), uint(h), result)
}
