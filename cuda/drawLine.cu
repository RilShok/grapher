#include <stdio.h>
#include <cuda.h>
//__global__ void mult_cuda(float *a, int N) {
//  int idx = blockIdx.x * blockDim.x + threadIdx.x;
//  if (idx<N) a[idx] = a[idx] * a[idx];
//}

float *DEVICE_IMAGE=NULL;
int _W = 0;
int _H = 0;
int _N = 0;
float _VALUE=0.;
int     _LINE_COUNT=0;
unsigned int   *DEVICE_RESULT_COUNTS=NULL;
float   *DEVICE_RESULT=NULL;
int     *DEVICE_X=NULL, *DEVICE_Y=NULL;
float   *DEVICE_LENGTH=NULL, *DEVICE_ANGLE=NULL;
const int __BLOCK_SIZE = 1024;

__global__ void
calculate_counts_cuda(int line_count, int W, int H, unsigned int *result, int *x, int *y, float *length, float *angle){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int lastX, lastY, m_x, m_y, xx, yy;
    float m_length, m_angle;
    //цикл избыточен у нас не так много линий чтобы выйти за сетку, но всеже
    for (int i = idx; i < line_count; i += gridDim.x * blockDim.x) {
        m_length = length[i];
        m_angle = angle[i];
        lastX = lastY =-1;
        m_x = x[i]; m_y = y[i];
        float k=0.;
        do{
            xx = (int)(float(m_x)+__cosf(m_angle)*k*m_length);
            yy = (int)(float(m_y)-__sinf(m_angle)*k*m_length);
            if (xx != lastX ||  yy != lastY) {
                lastX=xx;
                lastY=yy;
                atomicInc(result+lastY%H*W+lastX%W, 99999);
                //result[lastY%H*W+lastX%W] = 2;
            }
            k += 1./(1.5*m_length);
        }while(k<=1.);
    }
}


__global__ void // result должен быть размера равного количеству блоков 
calculate_cross_cuda2(unsigned int *result, int counts_size,unsigned int *counts){
    __shared__ unsigned int cache[__BLOCK_SIZE];
    unsigned int sum = 0;  
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < counts_size; i += gridDim.x * blockDim.x) {
        if (counts[i]>1){
            sum+=counts[i]-1;
        }
    }
    cache[threadIdx.x] = sum;
    __syncthreads();
    int i=blockDim.x/2;
    while(i!=0) {
        if (threadIdx.x<i)
            cache[threadIdx.x]+= cache[threadIdx.x+i];
        __syncthreads();
        i/=2;
    }
    if (threadIdx.x == 0)
        result[blockIdx.x]=cache[0];
    
}
//result хранит массив с счетчиком пересечений
__global__ void
calculate_cross_cuda(int line_count, unsigned int *result, int *x, int *y, float *length, float *angle){
    int line_idx_1 = blockIdx.x * blockDim.x + threadIdx.x;
    int line_idx_2 = blockIdx.y * blockDim.y + threadIdx.y;

    //необходимо внести обработку в цикл если произведение количества блоков на количество нитий меньше чем линий 
    if (line_idx_1 <= line_idx_2 || line_idx_1 >= line_count || line_idx_2 >= line_count) {
        return;
    }
    //1 линия
    float l1_length = length[line_idx_1];                                 // 
    float l1_angle  = angle[line_idx_1];                                  // 
    float l1_x1     = x[line_idx_1];                                      // x первой точки первой линии
    float l1_y1     = y[line_idx_1];                                      // y первой точки первой линии
    int   l1_x2     = (int)(float(l1_x1) + __cosf(l1_angle) * l1_length); // x второй точки первой линии
    int   l1_y2     = (int)(float(l1_y1) - __sinf(l1_angle) * l1_length); // y второй точки первой линии
    
if(l1_x1>l1_x2){
    int temp_x=l1_x1,temp_y=l1_y1;
    l1_x1=l1_x2;l1_y1=l1_y2;
    l1_x2=temp_x;l1_y2=temp_y;
}
if(l1_x1 == l1_x2 || l1_y1 == l1_y2){
    atomicInc(result+line_idx_1, 99999);
    return;
}
    //2 линия
    float l2_length = length[line_idx_2];                                 // 
    float l2_angle  = angle[line_idx_2];                                  // 
    float l2_x1     = x[line_idx_2];                                      // x первой точки первой линии
    float l2_y1     = y[line_idx_2];                                      // y первой точки первой линии
    int   l2_x2     = (int)(float(l2_x1) + __cosf(l2_angle) * l2_length); // x второй точки первой линии
    int   l2_y2     = (int)(float(l2_y1) - __sinf(l2_angle) * l2_length); // y второй точки первой линии

if(l2_x1>l2_x2){
    int temp_x=l2_x1,temp_y=l2_y1;
    l2_x1=l2_x2;l2_y1=l2_y2;
    l2_x2=temp_x;l2_y2=temp_y;
}

if(l2_x1 == l2_x2 || l2_y1 == l2_y2){
    atomicInc(result+line_idx_1, 99999);
    return;
}

    //координаты точки пересечения двух прямых на которых лежат отрезки
    float cross_x = ((l1_x1*l1_y2-l1_x2*l1_y1)*(l2_x2-l2_x1)-(l2_x1*l2_y2-l2_x2*l2_y1)*(l1_x2-l1_x1))/((l1_y1-l1_y2)*(l2_x2-l2_x1)-(l2_y1-l2_y2)*(l1_x2-l1_x1));
    float cross_y = ((l2_y1-l2_y2)*cross_x-(l2_x1*l2_y2-l2_x2*l2_y1))/(l2_x2-l2_x1);
    
    //проверка пересекаются ли две прямые
    if((((l1_x1<=cross_x)&&(l1_x2>=cross_x)&&(l2_x1<=cross_x)&&(l2_x2>=cross_x))||((l1_y1<=cross_y)&&(l1_y2>=cross_y)&&(l2_y1<=cross_y)&&(l2_y2>=cross_y)))&& ((l1_x2-l1_x1)/(l1_y2-l1_y1)!=(l2_x2-l2_x1)/(l2_y2-l2_y1)) ){
    // 
        //в кэш зафиксировали что есть пересечение
        atomicInc(result+line_idx_1, 99999);
    }
}

__global__ void
fillOne(float* fillArray, int N){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    while (idx < N) {
        fillArray[idx]=1.0;
        idx+=gridDim.x*blockDim.x;
    }
}


__global__ void
estimate_result_cuda(float *destinate_image, unsigned int *result_counts, int N, float value, float *estimate){
    __shared__ float cache[__BLOCK_SIZE];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    //int cacheIndex = threadIdx.x
    float sum = 0.;    
    while(idx<N){
        sum += powf(destinate_image[idx] - powf(value, (float)result_counts[idx]),2.);
        idx += gridDim.x * blockDim.x;
    }
    cache[threadIdx.x] = sum;
    __syncthreads();
    int i=blockDim.x/2;
    while(i!=0) {
        if (threadIdx.x<i)
            cache[threadIdx.x]+= cache[threadIdx.x+i];
        __syncthreads();
        i/=2;
    }
    if (threadIdx.x == 0)
        estimate[blockIdx.x]=cache[0];
}

//запускать в одном блоке на максимальном колличестве нитий
__global__ void
my_sum_array(unsigned int *result, unsigned int *array, int size){
    __shared__ unsigned int cache[__BLOCK_SIZE];
    unsigned int sum = 0;  
    for (int i=threadIdx.x;i<size;i+=blockDim.x){
        sum+=array[i];
    }
    cache[threadIdx.x] = sum;
    __syncthreads();
    int i=blockDim.x/2;
    while(i!=0) {
        if (threadIdx.x<i)
            cache[threadIdx.x]+= cache[threadIdx.x+i];
        __syncthreads();
        i/=2;
    }
    if (threadIdx.x == 0)
        *result=cache[0];
}

__global__ void
draw_result_cuda(unsigned int *result_counts, int N, float value, float *result){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    //int cacheIndex = threadIdx.x
    while(idx<N){
        result[idx] = powf(value, (float)result_counts[idx]);
        idx += gridDim.x * blockDim.x;
    }
}

extern "C" {

void destroyImage(){
    if (DEVICE_IMAGE){
        cudaFree(DEVICE_IMAGE);
        cudaFree(DEVICE_RESULT);
        cudaFree(DEVICE_RESULT_COUNTS);
        cudaFree(DEVICE_X);
        cudaFree(DEVICE_Y);
        cudaFree(DEVICE_LENGTH);
        cudaFree(DEVICE_ANGLE);
        
        DEVICE_IMAGE=NULL;
        _LINE_COUNT=0;
    }
}

void initImage(float *img, int W, int H, int line_count, float line_value){
    if (DEVICE_IMAGE != NULL || line_count != _LINE_COUNT){
        destroyImage();
    }
    _W = W; _H = H; _N = _W*_H;
    _VALUE = 1.-line_value;
    _LINE_COUNT = line_count;
    cudaMalloc((void **)&DEVICE_IMAGE, sizeof(float) * _N);
    cudaMemcpy(DEVICE_IMAGE, img, sizeof(float) * _N, cudaMemcpyHostToDevice);
    
    cudaMalloc((void**)&DEVICE_RESULT, sizeof(float)*_N);
    cudaMalloc((unsigned int**)&DEVICE_RESULT_COUNTS, sizeof(float)*_N);
    cudaMalloc((void**)&DEVICE_X, sizeof(int)*_LINE_COUNT);
    cudaMalloc((void**)&DEVICE_Y, sizeof(int)*_LINE_COUNT);
    cudaMalloc((void**)&DEVICE_LENGTH, sizeof(float)*_LINE_COUNT);
    cudaMalloc((void**)&DEVICE_ANGLE, sizeof(float)*_LINE_COUNT);
}

void copy_lines_to_GPU(int *x, int *y, float *length, float *angle){
    cudaMemcpy(DEVICE_X, x, sizeof(int)*_LINE_COUNT, cudaMemcpyHostToDevice);
    cudaMemcpy(DEVICE_Y, y, sizeof(int)*_LINE_COUNT, cudaMemcpyHostToDevice);
    cudaMemcpy(DEVICE_LENGTH, length, sizeof(float)*_LINE_COUNT, cudaMemcpyHostToDevice);
    cudaMemcpy(DEVICE_ANGLE, angle, sizeof(float)*_LINE_COUNT, cudaMemcpyHostToDevice);
}

//под ПЕРЕМЕННУЮ result память выделена в go коде
void estimate(float *result, int *x, int *y, float *length, float *angle) {
    //скопировать полученные данные в память устройства
    copy_lines_to_GPU(x, y, length, angle);  
    //обнулить масив счетчиков 
    cudaMemset(DEVICE_RESULT_COUNTS, 0, sizeof(unsigned int)*_N);  
    //посчитать количество попаданий каждой линии в пиксели
    int numBlocks = (_LINE_COUNT + __BLOCK_SIZE - 1) / __BLOCK_SIZE;
    calculate_counts_cuda <<< numBlocks, __BLOCK_SIZE >>> (
        _LINE_COUNT,
        _W, _H, 
        DEVICE_RESULT_COUNTS, 
        DEVICE_X, DEVICE_Y, 
        DEVICE_LENGTH, DEVICE_ANGLE
    );

    numBlocks = (_N + __BLOCK_SIZE - 1) / __BLOCK_SIZE; // но не больше чем ограничение по сетке! ИСПРАВИТЬ
    float *device_estimate_array;
    cudaMalloc((void**)&device_estimate_array, sizeof(float) * numBlocks);
    estimate_result_cuda <<< numBlocks, __BLOCK_SIZE >>>(
        DEVICE_IMAGE,
        DEVICE_RESULT_COUNTS,
        _N,
        _VALUE,
        device_estimate_array
    );
    float *estimate_array = (float*)malloc(sizeof(float) * numBlocks);
    cudaMemcpy(estimate_array, device_estimate_array, sizeof(float) * numBlocks, cudaMemcpyDeviceToHost);
    cudaFree(device_estimate_array);
    float res=0.;
    for(int i=0; i < numBlocks; i++){
        res+=estimate_array[i];
    }       
    free(estimate_array);

    //проверка на пересечения
    unsigned int cross_count=0;
    /* старый способ нахождения пересечений

    unsigned int *device_estimate_cross_array;
    //выделить память под массив оценок
    cudaMalloc((void**)&device_estimate_cross_array, sizeof(unsigned int) * _LINE_COUNT);
    cudaMemset(device_estimate_cross_array, 0, sizeof(unsigned int) * _LINE_COUNT);
    dim3 dimBlock(32, 32);
    numBlocks = (_LINE_COUNT + 32 - 1) / 32; 
    dim3 dimGrid(numBlocks, numBlocks);     
    calculate_cross_cuda <<< dimGrid, dimBlock >>> (
        _LINE_COUNT,
        device_estimate_cross_array,
        DEVICE_X, DEVICE_Y, 
        DEVICE_LENGTH, DEVICE_ANGLE
    );


    //считаем сумарное количество пересечений
    
    unsigned int *_device_cross_count;
    cudaMalloc((void**)&_device_cross_count, sizeof(unsigned int));
    my_sum_array<<< 1, __BLOCK_SIZE >>>(_device_cross_count,device_estimate_cross_array, _LINE_COUNT);
    cudaMemcpy(&cross_count, _device_cross_count, sizeof(unsigned int), cudaMemcpyDeviceToHost);
    cudaFree(_device_cross_count);
    //копируем результат массив с счетчиками пересечений в оперативную память хоста
    cudaFree(device_estimate_cross_array);
    
    */

    //новый способ нахождения количества пересечений
/*
    unsigned int post_calculation_cross_count[__BLOCK_SIZE];

    unsigned int *device_post_calculation_cross_count;
    cudaMalloc((void**)&device_post_calculation_cross_count, sizeof(unsigned int) * __BLOCK_SIZE);
    cudaMemset(device_post_calculation_cross_count, 0, sizeof(unsigned int) * __BLOCK_SIZE);
    calculate_cross_cuda2 <<<(_N + __BLOCK_SIZE - 1) / __BLOCK_SIZE, __BLOCK_SIZE>>> (
        device_post_calculation_cross_count,
        _N,
        DEVICE_RESULT_COUNTS
    );
    cudaMemcpy(
        post_calculation_cross_count,
        device_post_calculation_cross_count,
        sizeof(unsigned int)*__BLOCK_SIZE,
        cudaMemcpyDeviceToHost
    );
    for (int i=0;i<__BLOCK_SIZE;i++){
        cross_count+=post_calculation_cross_count[i];
    }
*/
    *result=res+((float)cross_count);

}


void draw(float *result, int *x, int *y, float *length, float *angle) {  
    //!!!!drawImage(x, y, length, angle, value);

    //скопировать полученные данные в память устройства
    copy_lines_to_GPU(x, y, length, angle);  
    //обнулить масив счетчиков 
    cudaMemset(DEVICE_RESULT_COUNTS, 0, sizeof(unsigned int) * _N);
    //посчитать количество попаданий каждой линии в пиксели
    int numBlocks = (_LINE_COUNT + __BLOCK_SIZE - 1) / __BLOCK_SIZE;
    calculate_counts_cuda <<< numBlocks, __BLOCK_SIZE >>> (
        _LINE_COUNT,
        _W, _H, 
        DEVICE_RESULT_COUNTS, 
        DEVICE_X, DEVICE_Y, 
        DEVICE_LENGTH, DEVICE_ANGLE
    );

    numBlocks = (_N + __BLOCK_SIZE - 1) / (__BLOCK_SIZE); // но не больше чем ограничение по сетке! ИСПРАВИТЬ
    
    draw_result_cuda <<< numBlocks, __BLOCK_SIZE >>>(
        DEVICE_RESULT_COUNTS,
        _N,
        _VALUE,
        DEVICE_RESULT
    );

    cudaMemcpy(result,DEVICE_RESULT,sizeof(float)*_N,cudaMemcpyDeviceToHost);
}

}

