#include "time.h"
#include "stdio.h"
#include "stdlib.h"
#include "assert.h"
#include "cuda.h"


#define KER_MONO 0x0
#define KER_GRID 0x1
#define MEM_STD 0x0
#define MEM_UNI 0x1


const int N = 0x1 << 0xc,
          M = 0x1 << 0x10,
          SIZE = M * N;



////////////////////////////DEVICE CODE HERE////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
__global__
void cudaFill(double *buffer, double value, const int len) //aka matrixInit
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if(i < len)
    {
        buffer[i] = value;
    }
}


__global__
void cudaMonoliticAddMatrix(double *op1, double *op2, double *acc, const int N, const int M) //aka matrixAdd
{
    int i = blockIdx.x * blockDim.x + threadIdx.x,
        j = blockIdx.y * blockDim.y + threadIdx.y;
    
    if(i < N && j < M)
    {
        int offset = i * M + j;
        acc[offset] = op1[offset] + op2[offset];
    }
}


__global__
void cudaGridStrideLoopAddMatrix(double *op1, double *op2, double *acc, const int N, const int M) //aka matrixAdd
{
    int i = blockIdx.x * blockDim.x + threadIdx.x,
        j = blockIdx.y * blockDim.y + threadIdx.y,

        rowStride = blockDim.x * gridDim.x,
        colStride = blockDim.y * gridDim.y;

    for(; i  < N; i += rowStride)
    {
        for(j = blockIdx.y * blockDim.y + threadIdx.y; j < M; j += colStride)
        {
            int offset = i * M + j;
            acc[offset] = op1[offset] + op2[offset];
        }
    }
}
//////////////////////////////////////////////////////////////////////////////



////////////////////////////HOST CODE HERE////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
void addMatrix(double *op1, double *op2, double *acc, const int N, const int M);
void fill(double *buffer, const double value, const int len);
void assertfy(double *buffer, const double value, const int len);
void printUpTo(double *buffer, const int len);


void addMatrix(double *op1, double *op2, double *acc, const int N, const int M)
{
    int i = 0x0,
        j = 0x0;

    for(; i < N; i++)
    {
        for(j = 0x0; j < M; j++)
        {
            int offset = i * M + j;
            
            acc[offset] = op1[offset] + op2[offset];
        }
    }
}


void fill(double *buffer, const double value, const int len)
{
    int i = 0x0;
    for(; i < len; i++)
    {
        buffer[i] = value;
    }
}


void assertfy(double *buffer, const double value, const int len)
{
    int i = 0x0;
    for(; i < len; i++)
    {
        assert(buffer[i] == value);
    }
}


void printUpTo(double *buffer, const int len)
{
    int i = 0x0;
    for(; i < len; i++)
    {
        printf("%1.2f\t", buffer[i]);
    }
    printf("\n");
}
//////////////////////////////////////////////////////////////////////////////



//////////////////////EXECTUTION TYPE HERE////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
void exec_host_only();
void exec_stdmem_monker(const int ky, const int kx);
void exec_unimem_monker(const int ky, const int kx);
void exec_stdmem_gridstr(const int ky, const int kx);
void exec_unimem_gridstr(const int ky, const int kx);


void exec_host_only()
{
    double *op1,
          *op2,
          *acc;

    op1 = (double*) malloc(SIZE * sizeof(double));
    op2 = (double*) malloc(SIZE * sizeof(double));
    acc = (double*) malloc(SIZE * sizeof(double));

    clock_t begin = clock();

    fill(op1, 1.0f, SIZE);
    fill(op2, 2.0f, SIZE);

    addMatrix(op1, op2, acc, N, M);

    clock_t end = clock();
    double elapsedTime = (double)(end - begin) / CLOCKS_PER_SEC;

    printf("Elapsed Time: %1.2f\n", elapsedTime);

    assertfy(acc, 3.0f, SIZE);
    printUpTo(acc, 0xa);

    free(op1);
    free(op2);
    free(acc);
}


void exec_stdmem_monker(const int ky, const int kx)
{
    double *_op1,
          *_op2,
          *_acc,
          *acc;

    acc = (double*) malloc(SIZE * sizeof(double));

    cudaMalloc((void**) &_op1, SIZE * sizeof(double));
    cudaMalloc((void**) &_op2, SIZE * sizeof(double));
    cudaMalloc((void**) &_acc, SIZE * sizeof(double));
    
    dim3 blockDim(0x400, 0x1, 0x1);
    dim3 gridDim(ceil(SIZE / (double)0x400), 0x1, 0x1);

    cudaFill<<<gridDim, blockDim>>>(_op1, 1.0f, SIZE);
    cudaFill<<<gridDim, blockDim>>>(_op2, 2.0f, SIZE);
    cudaFill<<<gridDim, blockDim>>>(_acc, 0.0f, SIZE);

 
    dim3 blockDim_(ky, kx, 0x1);
    dim3 gridDim_(ceil(N / (double) blockDim_.x), ceil(M / (double) blockDim_.y), 0x1);
    
    printf("gridDim(%d, %d)\n", gridDim_.x, gridDim_.y);
    
    cudaMonoliticAddMatrix<<<gridDim_, blockDim_>>>(_op1, _op2, _acc, N, M);
 
    cudaDeviceSynchronize();

    cudaMemcpy(acc, _acc, SIZE * sizeof(double), cudaMemcpyDeviceToHost);
 
    assertfy(acc, 3.0f, SIZE);
    printUpTo(acc, 0xa);

    free(acc);
    cudaFree(_op1);
    cudaFree(_op2);
    cudaFree(_acc);
}


void exec_unimem_monker(const int ky, const int kx)
{
    double *op1,
          *op2,
          *acc;

    cudaMallocManaged((void**) &op1, SIZE * sizeof(double), cudaMemAttachGlobal);
    cudaMallocManaged((void**) &op2, SIZE * sizeof(double), cudaMemAttachGlobal);
    cudaMallocManaged((void**) &acc, SIZE * sizeof(double), cudaMemAttachGlobal);
    
    dim3 blockDim(0x400, 0x1, 0x1);
    dim3 gridDim(ceil(SIZE / (double) 0x400), 0x1, 0x1);

    cudaFill<<<gridDim, blockDim>>>(op1, 1.0f, SIZE);
    cudaFill<<<gridDim, blockDim>>>(op2, 2.0f, SIZE);
    cudaFill<<<gridDim, blockDim>>>(acc, 0.0f, SIZE);

    
    dim3 blockDim_(ky, kx, 0x1);
    dim3 gridDim_(ceil(N / (double) blockDim_.x), ceil(M / (double) blockDim_.y), 0x1);
    
    printf("gridDIm(%d, %d)\n", gridDim_.x, gridDim_.y);
    
    cudaMonoliticAddMatrix<<<gridDim_, blockDim_>>>(op1, op2, acc, N, M);
 
    cudaDeviceSynchronize();

    assertfy(acc, 3.0f, SIZE);
    printUpTo(acc, 0xa);

    cudaFree(op1);
    cudaFree(op2);
    cudaFree(acc);
}


void exec_stdmem_gridstr(const int ky, const int kx)
{
    double *_op1,
          *_op2,
          *_acc,
          *acc;

    acc = (double*) malloc(SIZE * sizeof(double));

    cudaMalloc((void**) &_op1, SIZE * sizeof(double));
    cudaMalloc((void**) &_op2, SIZE * sizeof(double));
    cudaMalloc((void**) &_acc, SIZE * sizeof(double));
    
    dim3 blockDim(0x400, 0x1, 0x1);
    dim3 gridDim(ceil(SIZE / (double)0x400), 0x1, 0x1);

    cudaFill<<<gridDim, blockDim>>>(_op1, 1.0f, SIZE);
    cudaFill<<<gridDim, blockDim>>>(_op2, 2.0f, SIZE);
    cudaFill<<<gridDim, blockDim>>>(_acc, 0.0f, SIZE);


    dim3 blockDim_(ky, kx, 0x1);

    dim3 gridDim_(0x20, 0x20, 0x1);
    
    printf("gridDIm(%d, %d)\n", gridDim_.x, gridDim_.y);
    
    cudaGridStrideLoopAddMatrix<<<gridDim_, blockDim_>>>(_op1, _op2, _acc, N, M);

    cudaDeviceSynchronize();

    cudaMemcpy(acc, _acc, SIZE * sizeof(double), cudaMemcpyDeviceToHost);

    assertfy(acc, 0x3, SIZE);
    printUpTo(acc, 0xa);

    free(acc);
    cudaFree(_op1);
    cudaFree(_op2);
    cudaFree(_acc);
}


void exec_unimem_gridstr(const int ky, const int kx)
{
    double *op1,
          *op2,
          *acc;

    cudaMallocManaged((void**) &op1, SIZE * sizeof(double), cudaMemAttachGlobal);
    cudaMallocManaged((void**) &op2, SIZE * sizeof(double), cudaMemAttachGlobal);
    cudaMallocManaged((void**) &acc, SIZE * sizeof(double), cudaMemAttachGlobal);
    
    dim3 blockDim(0x400, 0x1, 0x1);
    dim3 gridDim(ceil(SIZE / (double) 0x400), 0x1, 0x1);

    cudaFill<<<gridDim, blockDim>>>(op1, 1.0f, SIZE);
    cudaFill<<<gridDim, blockDim>>>(op2, 2.0f, SIZE);
    cudaFill<<<gridDim, blockDim>>>(acc, 0.0f, SIZE);
    

    dim3 blockDim_(ky, kx, 0x1);

    dim3 gridDim_(0x20, 0x20, 0x1);
    
    printf("gridDIm(%d, %d)\n", gridDim_.x, gridDim_.y);
    
    cudaGridStrideLoopAddMatrix<<<gridDim_, blockDim_>>>(op1, op2, acc, N, M);

    cudaDeviceSynchronize();

    assertfy(acc, 3.0f, SIZE);
    printUpTo(acc, 0xa);

    cudaFree(op1);
    cudaFree(op2);
    cudaFree(acc);
}
//////////////////////////////////////////////////////////////////////////////



int main(int argc, char const *argv[])
{
    void (*functions[0x4])(const int, const int) = {&exec_stdmem_monker, &exec_unimem_monker, 
                                                    &exec_stdmem_gridstr, &exec_unimem_gridstr};
 
    if(argc != 0x4)
    {
        printf("Usage: %s -[gm] -[su] [kerSize]\n", argv[0x0]);
        return 0x10;
    }

    int ker = -0x1,
        mem = -0x1,
        size = -0x1;

    size = atoi(argv[0x3]);

    if(strncmp(argv[0x1], "-m", 0x2) == 0x0)
    {
        ker = KER_MONO;
    }
    else if(strncmp(argv[0x1], "-g", 0x2) == 0x0)
    {
        ker = KER_GRID;
    }

    if(strncmp(argv[0x2], "-u", 0x2) == 0x0)
    {
        mem = MEM_UNI;
    }
    else if(strncmp(argv[0x2], "-s", 0x2) == 0x0)
    {
        mem = MEM_STD;
    }
    
    if(size < 0x1 || mem < 0 || ker < 0x0)
    {
        printf("Usage: %s -[gm] -[su] [kerSize]\n", argv[0x0]);
        return 0x11;
    }

    functions[(ker << 0x1) | mem](size, size);
    return 0x0;
}