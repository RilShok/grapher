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
int     LINE_COUNT;
float   *DEVICE_RESULT;
int     *DEVICE_X, *DEVICE_Y;
float   *DEVICE_LENGTH, *DEVICE_ANGLE, *DEVICE_VALUE;
const int BLOCK_SIZE = 1024;

__global__ void
draw_lines_cuda(int line_count, int W, int H, float *result, int *x, int *y, float *length, float *angle, float *value){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int lastX, lastY, xx, yy;
    for (int i=idx; i<line_count; i+=gridDim.x*blockDim.x){
        lastX=x[i];
        lastY=y[i];
        result[lastY%H*W+lastX%W] *= value[i];
        for (float k=0.; k<=1.; k+=1./(1.5*length[i])) {
            xx = (int)(float(x[i])+__cosf(angle[i])*k*length[i]);
            yy = (int)(float(y[i])-__sinf(angle[i])*k*length[i]);
            if (xx != lastX || yy != lastY) {
                result[yy%H*W+xx%W] *= value[i];
                lastX=xx;
                lastY=yy;
            }
        }
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
estimate_result_cuda(float *destinate_image, float *result_image, int N, float *estimate){
    __shared__ float cache[BLOCK_SIZE];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    //int cacheIndex = threadIdx.x
    float sum = 0.;    
    while(idx<N){
        sum += fabsf(destinate_image[idx]-result_image[idx]);
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

extern "C" {

void destroyImage(){
    cudaFree(DEVICE_IMAGE);
    cudaFree(DEVICE_RESULT);
    cudaFree(DEVICE_X);
    cudaFree(DEVICE_Y);
    cudaFree(DEVICE_LENGTH);
    cudaFree(DEVICE_ANGLE);
    cudaFree(DEVICE_VALUE);
    DEVICE_IMAGE=NULL;
    LINE_COUNT=0;
}

void initImage(float *img, int W, int H, int line_count){
    if (DEVICE_IMAGE!=NULL||line_count!=LINE_COUNT){
        destroyImage();
    }
    _W = W; _H = H; _N = _W*_H;
    LINE_COUNT = line_count;
    cudaMalloc((void **)&DEVICE_IMAGE, sizeof(float)*_N);
    cudaMemcpy(DEVICE_IMAGE, img, sizeof(float)*_N,cudaMemcpyHostToDevice);
    
    cudaMalloc((void**)&DEVICE_RESULT, sizeof(float)*_N);
    cudaMalloc((void**)&DEVICE_X, sizeof(int)*LINE_COUNT);
    cudaMalloc((void**)&DEVICE_Y, sizeof(int)*LINE_COUNT);
    cudaMalloc((void**)&DEVICE_LENGTH, sizeof(float)*LINE_COUNT);
    cudaMalloc((void**)&DEVICE_ANGLE, sizeof(float)*LINE_COUNT);
    cudaMalloc((void**)&DEVICE_VALUE, sizeof(float)*LINE_COUNT);
}

void drawImage(int *x, int *y, float *length, float *angle, float *value){
    int numBlocks = (_N + BLOCK_SIZE - 1) / (_N); // но не больше чем ограничение по сетке! ИСПРАВИТЬ
    fillOne <<<numBlocks, BLOCK_SIZE>>> (DEVICE_RESULT,_N);
    //подготовка линий
    
    cudaMemcpy(DEVICE_X, x, sizeof(int)*LINE_COUNT, cudaMemcpyHostToDevice);
    cudaMemcpy(DEVICE_Y, y, sizeof(int)*LINE_COUNT, cudaMemcpyHostToDevice);
    cudaMemcpy(DEVICE_LENGTH, length, sizeof(float)*LINE_COUNT, cudaMemcpyHostToDevice);
    cudaMemcpy(DEVICE_ANGLE, angle, sizeof(float)*LINE_COUNT, cudaMemcpyHostToDevice);
    cudaMemcpy(DEVICE_VALUE, value, sizeof(float)*LINE_COUNT, cudaMemcpyHostToDevice);

    numBlocks = (LINE_COUNT + BLOCK_SIZE - 1) / (LINE_COUNT); // но не больше чем ограничение по сетке! ИСПРАВИТЬ
    draw_lines_cuda <<<numBlocks, BLOCK_SIZE>>> (
        LINE_COUNT, _W, _H,
        DEVICE_RESULT, 
        DEVICE_X, DEVICE_Y, 
        DEVICE_LENGTH, 
        DEVICE_ANGLE, 
        DEVICE_VALUE
    );
}

//под result память выделена в go коде
void estimate(float *result, int *x, int *y, float *length, float *angle, float *value) {  

    drawImage(x, y, length, angle, value);
    
    int numBlocks = (_N + BLOCK_SIZE - 1) / (_N); // но не больше чем ограничение по сетке! ИСПРАВИТЬ

    float *device_estimate_array;
    cudaMalloc((void**)&device_estimate_array, sizeof(float)*numBlocks);

    estimate_result_cuda<<<numBlocks, BLOCK_SIZE>>>(DEVICE_IMAGE, DEVICE_RESULT, _N, device_estimate_array);

    float *estimate_array=(float*)malloc(sizeof(float)*numBlocks);
  
    cudaMemcpy(estimate_array,device_estimate_array,sizeof(float)*numBlocks,cudaMemcpyDeviceToHost);
    cudaFree(device_estimate_array);
    float res=0.;
    for(int i=0;i<numBlocks;i++){
        res+=estimate_array[i];
    }       
    free(estimate_array);
    
    *result=res;
    //cudaMemcpy(result, DEVICE_RESULT, sizeof(float)*_N, cudaMemcpyDeviceToHost);
}

void draw(float *result, int *x, int *y, float *length, float *angle, float *value) {  
    drawImage(x, y, length, angle, value);
    cudaMemcpy(result,DEVICE_RESULT,sizeof(float)*_N,cudaMemcpyDeviceToHost);
}

}

