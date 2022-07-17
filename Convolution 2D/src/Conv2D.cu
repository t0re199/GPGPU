#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <string.h>
#include <time.h>
#include <cuda.h>


extern "C" float *loadPng(char *path, int *height, int *width);
extern "C" void dumpPng(char *path, float *floatImage, int height, int width);



#define BLOCK_SIZE 0x20
#define KERNEL_SIZE 0x3

#define tx threadIdx.x
#define ty threadIdx.y


#define RGB_CHANNELS 0x3
#define RGBA_CHANNELS 0x4


char SRC_PATH[] = "../Data/Sample.png",
     CPU_OUTPUT_PATH[] = "../Data/CpuResult.png",
     CUDA_STD_OUTPUT_PATH[] = "../Data/CudaResult.png",
     CUDA_TILED_OUTPUT_PATH[] = "../Data/TiledResult.png";


////////////////////////////DEVICE CODE HERE////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

__constant__ float KERNEL[KERNEL_SIZE][KERNEL_SIZE] = {
                                                        {0.0f, -1.0f, 0.0f},
                                                        {-1.0f, 5.0f, 1.0f},
                                                        {0.0f, 1.0f, 0.0f}
                                                    };


float CPU_KERNEL[KERNEL_SIZE][KERNEL_SIZE] = {
                                                {0.0f, -1.0f, 0.0f},
                                                {-1.0f, 5.0f, 1.0f},
                                                {0.0f, 1.0f, 0.0f}
                                             };


__global__ void cudaConv2d(float *dest, float *src, int height, int width)
{
    
    int row = blockIdx.y * blockDim.y + ty,
        col = blockIdx.x * blockDim.x + tx;
    
    int kerRadius = KERNEL_SIZE / 0x2,
        rowBase = row - kerRadius,
        colBase = col - kerRadius,
        i = 0x0,
        j;


    if(row < height && col < width)
    {
        float acc = 0.0f;
        for(; i < KERNEL_SIZE; i++)
        {
            for(j = 0x0; j < KERNEL_SIZE; j++) 
            {
                if( (colBase + j >= 0x0) && (colBase + j < width) && (rowBase + i >= 0x0) && (rowBase + i < height))
                {
                    acc += src[((rowBase + i) * width) + (colBase + j)] * KERNEL[i][j];
                } 
            }
        }
        dest[row * width + col] = acc;
    }
}


__global__ void cudaConv2dTiled(float *dest, float *src, int height, int width, int tileSize)
{
    __shared__ float sharedData[BLOCK_SIZE][BLOCK_SIZE];
    
    int row = blockIdx.y * tileSize + ty,
        col = blockIdx.x * tileSize + tx,

        rowBase = row - (KERNEL_SIZE / 0x2),
        colBase = col - (KERNEL_SIZE / 0x2);
       

    if( (rowBase >= 0x0) && (rowBase < height) && 
        (colBase >= 0x0) && (colBase < width) )
    {
        sharedData[ty][tx] = src[rowBase * width + colBase];
    }
    else
    {
        sharedData[ty][tx] = 0.0f;
    }

    __syncthreads();

    if((ty < tileSize) && (tx < tileSize) && row < height && col < width)
    {
        int i = 0x0,
            j;

        float acc = 0.0f;

        for(; i < KERNEL_SIZE; i++)
        {
            for(j = 0x0; j < KERNEL_SIZE; j++) 
            {
                acc += sharedData[ty + i][tx + j] * KERNEL[i][j];
            }
        }
        dest[row * width + col] = acc;
    }
}


__global__ void cudaFill(float *buffer, float value, const int len)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < len)
    {
        buffer[i] = value;
    }
}

//////////////////////////////////////////////////////////////////////////////

////////////////////////////HOST CODE HERE////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
void fill(float *buffer, const float value, const int len);
void printUpTo(float *buffer, const int len);

void conv2d(float *dest, float *src, int height, int width);
void extractChannel(float *dest, float *image, int height, int width, int channelIdx);
void setChannel(float *image, float *buffer, int height, int width, int channelIdx);



void fill(float *buffer, const float value, const int len)
{
    int i = 0x0;
    for (; i < len; i++)
    {
        buffer[i] = value;
    }
}


void printUpTo(float *buffer, const int len)
{
    int i = 0x0;
    for (; i < len; i++)
    {
        printf("%1.2f\t", buffer[i]);
    }
    printf("\n");
}


void extractChannel(float *buffer, float *image, int height, int width, int channelIdx)
{
    int i = 0x0,
        j,
        cIdx;

    for(; i < height; i++)
    {
        cIdx = channelIdx;
        for(j = 0x0; j < width; j++)
        {
            buffer[i * width + j] = image[i * width * RGBA_CHANNELS + cIdx];
            cIdx += RGBA_CHANNELS; 
        }
    }
}


void setChannel(float *image, float *buffer, int height, int width, int channelIdx)
{
    int i = 0x0,
        j,
        cIdx;

    for(; i < height; i++)
    {
        cIdx = channelIdx;
        for(j = 0x0; j < width; j++)
        {
            image[i * width * RGBA_CHANNELS + cIdx] = buffer[i * width + j]; 
            cIdx += RGBA_CHANNELS; 
        }
    }
}


void conv2d(float *dest, float *src, int height, int width)
{
    
    int row = 0x0,
        kerRadius = KERNEL_SIZE / 0x2,
        col,
        i,
        j;


    for(; row < height; row++)
    {
        for(col = 0x0; col < width; col++)
        {
            int rowBase = row - kerRadius,
                colBase = col - kerRadius;
            
            float acc = 0.0f;

            for(i = 0x0; i < KERNEL_SIZE; i++)
            {
                for(j = 0x0; j < KERNEL_SIZE; j++) 
                {
                    if( (colBase + j >= 0x0) && (colBase + j < width) && (rowBase + i >= 0x0) && (rowBase + i < height))
                    {
                        acc += src[((rowBase + i) * width) + (colBase + j)] * CPU_KERNEL[i][j];
                    } 
                }
            }
            dest[row * width + col] = acc;
        }
    }
}
//////////////////////////////////////////////////////////////////////////////


//////////////////////EXECTUTION TYPE HERE////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
void exec_host_only();
void exec_cuda_std();
void exec_cuda_tiled();


void exec_host_only()
{
    float *src,
          *dest,
          *output,
          *cpuImage,
          time,
          elapsedTime = 0xffffffff;
    
    clock_t tmp;

    int height,
        width,
        c = 0x0;
    
    cpuImage = loadPng(SRC_PATH, &height, &width);
    printf("[*] Image Shape (%d, %d)\n", height, width);

    size_t size = height * width * sizeof(float);

    output = (float*) malloc(size * RGBA_CHANNELS);
    memcpy(output, cpuImage, size * RGBA_CHANNELS);
    
    src = (float *) malloc(size);
    dest = (float *) malloc(size);

    for(; c < RGB_CHANNELS; c++)
    {
        extractChannel(src, cpuImage, height, width, c);
        
        tmp = -clock();
        conv2d(dest, src, height, width);
        tmp += clock();

        time = (float) tmp / (CLOCKS_PER_SEC / 1000);
        if(time < elapsedTime)
        {
            elapsedTime = time;
        }

        setChannel(output, dest, height, width, c);
    }

    printf("[*] Elapsed Time %1.3f ms\n", elapsedTime);
    dumpPng(CPU_OUTPUT_PATH, output, height, width);

    free(src);
    free(dest);
    free(cpuImage);
    free(output);
}



void exec_cuda_std()
{
    float *src,
          *dest,
          *cpuImage,
          *output;

    int height,
        width,
        c = 0x0;

    cpuImage = loadPng(SRC_PATH, &height, &width);
    printf("[*] Image Shape (%d, %d)\n", height, width);

    size_t size = height * width * sizeof(float);

    output = (float*) malloc(size * RGBA_CHANNELS);
    memcpy(output, cpuImage, size * RGBA_CHANNELS);
    
    cudaMallocManaged((void **) &src, size, cudaMemAttachGlobal);
    cudaMallocManaged((void **) &dest, size, cudaMemAttachGlobal);
    
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE, 0x1);
    dim3 gridDim(ceil(width / (float) BLOCK_SIZE), ceil(height / (float) BLOCK_SIZE), 0x1);

    printf("[*] gridDim(%d, %d)\n", gridDim.x, gridDim.y);
    printf("[*] blockDim(%d, %d)\n", blockDim.x, blockDim.y);

    for(; c < RGB_CHANNELS; c++)
    {
        extractChannel(src, cpuImage, height, width, c);
        cudaConv2d<<<gridDim, blockDim>>>(dest, src, height, width);
        cudaDeviceSynchronize();
        setChannel(output, dest, height, width, c);
    }

    dumpPng(CUDA_STD_OUTPUT_PATH, output, height, width);

    cudaFree(src);
    cudaFree(dest);
    free(cpuImage);
    free(output);
}


void exec_cuda_tiled()
{
    float *src,
          *dest,
          *cpuImage,
          *output;

    int height,
        width,
        c = 0x0,
        tileSize = BLOCK_SIZE - (KERNEL_SIZE - 0x1);

    cpuImage = loadPng(SRC_PATH, &height, &width);
    printf("[*] Image Shape (%d, %d)\n", height, width);

    size_t size = height * width * sizeof(float);

    output = (float*) malloc(size * RGBA_CHANNELS);
    memcpy(output, cpuImage, size * RGBA_CHANNELS);
    
    cudaMallocManaged((void **) &src, size, cudaMemAttachGlobal);
    cudaMallocManaged((void **) &dest, size, cudaMemAttachGlobal);
    
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE, 0x1);
    dim3 gridDim(ceil(width / (float) tileSize), ceil(height / (float) tileSize), 0x1);

    printf("[*] gridDim(%d, %d)\n", gridDim.x, gridDim.y);
    printf("[*] blockDim(%d, %d)\n", blockDim.x, blockDim.y);

    for(; c < RGB_CHANNELS; c++)
    {
        extractChannel(src, cpuImage, height, width, c);
        cudaConv2dTiled<<<gridDim, blockDim>>>(dest, src, height, width, tileSize);
        cudaDeviceSynchronize();
        setChannel(output, dest, height, width, c);
    }

    dumpPng(CUDA_TILED_OUTPUT_PATH, output, height, width);

    cudaFree(src);
    cudaFree(dest);
    free(cpuImage);
    free(output);
}


//////////////////////////////////////////////////////////////////////////////

int main(int argc, char const *argv[])
{
    if(argc != 0x2)
    {
        printf("Usage: %s -[cgt]\n", argv[0x0]);
        return 0x10;
    }

    if(strncmp(argv[0x1], "-c", 0x2) == 0x0)
    {
        exec_host_only();
    }
    else if(strncmp(argv[0x1], "-g", 0x2) == 0x0)
    {
        exec_cuda_std();
    }
    else if(strncmp(argv[0x1], "-t", 0x2) == 0x0)
    {
        exec_cuda_tiled();
    }
    else
    {
        printf("Usage: %s -[cgt]\n", argv[0x0]);
        return 0x11;
    }
    
    return 0x0;
}







