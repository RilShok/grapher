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
const int __BLOCK_SIZE = 96;

__global__ void
calculate_counts_cuda(int line_count, int W, int H, unsigned int *result, int *x, int *y, float *length, float *angle){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int lastX, lastY, m_x, m_y, xx, yy;
    float m_length, m_angle;
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
    //__syncthreads();
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
        sum += fabsf(destinate_image[idx] - powf(value, (float)result_counts[idx]));
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

//проверка девайскаунтов
//unsigned int testCounts[7000];
//cudaMemcpy(testCounts, DEVICE_RESULT_COUNTS, sizeof(unsigned int) * 7000, cudaMemcpyDeviceToHost);
//for(int i=0;i<7000;i++){
//    printf("%d ",testCounts[i]);
//}
//printf("\n");
    //посчитать целевую функцию ошибки
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
    *result=res;
}

//void drawImage(int *x, int *y, float *length, float *angle){
//    int numBlocks = (_N + __BLOCK_SIZE - 1) / (_N); // но не больше чем ограничение по сетке! ИСПРАВИТЬ
//    fillOne <<<numBlocks, __BLOCK_SIZE>>> (DEVICE_RESULT,_N);
//    //подготовка линий
//    copy_lines_to_GPU(x,y,length,angle);
//    
//
//    numBlocks = (_LINE_COUNT + __BLOCK_SIZE - 1) / (_LINE_COUNT); // но не больше чем ограничение по сетке! ИСПРАВИТЬ
//    draw_lines_cuda <<<numBlocks, __BLOCK_SIZE>>> (
//        _LINE_COUNT, _W, _H,
//        DEVICE_RESULT, 
//        DEVICE_X, DEVICE_Y, 
//        DEVICE_LENGTH, 
//        DEVICE_ANGLE, 
//        DEVICE_VALUE
//    );
//}


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

