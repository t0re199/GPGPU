#include "time.h"
#include "stdio.h"
#include "stdlib.h"
#include "assert.h"
#include "cuda.h"


#define tx threadIdx.x
#define ty threadIdx.y


#define KER_SIZE 0x8


const int OP1_ROWS = 2000,
          OP1_COLS = 500,
          OP2_ROWS = 500,
          OP2_COLS = 2000,
          SIZE = OP1_ROWS * OP1_COLS,
          ACC_SIZE = OP1_ROWS * OP2_COLS;



////////////////////////////DEVICE CODE HERE////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
__global__
void cudaFill(float *buffer, float value, const int len)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if(i < len)
    {
        buffer[i] = value;
    }
}


__global__
void cudaStdMatMult(float* op1, float* op2, float* acc, 
                    const int op1Rows, const int op1Cols, const int op2Rows,
                    const int op2Cols, int const accRows, const int accCols)
{
    int row = blockIdx.y * blockDim.y + ty,
        col = 0x0,
        k = 0x0,

        rowStride = blockDim.y * gridDim.y,
        colStride = blockDim.x * gridDim.x;


    for(; row < op1Rows; row += rowStride)
    {
        for(col = blockIdx.x * blockDim.x + tx; col < op2Cols; col += colStride)
        {
            float scalarProduct = 0.0f;

            for(k = 0x0; k < op1Cols; k++)
            {
                scalarProduct += op1[row * op1Cols + k] * op2[k * op2Cols + col];
            }
            acc[row * accCols + col] = scalarProduct;
        }
    }
}


__global__ 
void cudaTiledMatMul(float* op1, float* op2, float* acc, 
                    const int op1Rows, const int op1Cols, const int op2Rows,
                    const int op2Cols, int const accRows, const int accCols)
{
    
    __shared__ float s_op1[KER_SIZE][KER_SIZE];
    __shared__ float s_op2[KER_SIZE][KER_SIZE];
    
    int row = blockIdx.y * KER_SIZE + ty,
        col = blockIdx.x * KER_SIZE + tx,

        rowStride = blockDim.y * gridDim.y,
        colStride = blockDim.x * gridDim.x,
 

        rowLimit = op1Rows + (op1Rows % (blockDim.y * gridDim.y)),
        colLimit = op2Cols + (op2Cols % (blockDim.x * gridDim.x)),
 
        phases = ceil(op1Cols/(float)KER_SIZE);
 
    for(; row < rowLimit; row += rowStride)
    {
        for(col = blockIdx.x * KER_SIZE + tx; col < colLimit; col += colStride)
        {
            float scalarProduct = 0.0f;
            for (int p = 0; p < phases; p++) 
            {
                int i = p * KER_SIZE + tx,
                    j = p * KER_SIZE + ty;

                if (i < op1Cols && row < op1Rows)
                {
                    s_op1[ty][tx] = op1[row * op1Cols + i];
                }
                else
                {
                    s_op1[ty][tx] = 0.0f;
                }

                if (j < op2Rows && col < op2Cols)
                {
                    s_op2[ty][tx] = op2[j * op2Cols + col];
                }
                else
                {   
                    s_op2[ty][tx] = 0.0f;
                }

                __syncthreads();

                #pragma unroll
                for (int k = 0; k < KER_SIZE; k++)
                {
                    scalarProduct += s_op1[ty][k] * s_op2[k][tx];
                }

                __syncthreads();
            }

            if (row < accRows && col < accCols)
            {
                acc[row * accCols + col] = scalarProduct;
            }
        }
    }
}
//////////////////////////////////////////////////////////////////////////////



////////////////////////////HOST CODE HERE////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
void matMul(float* op1, float* op2, float* acc, const int op1Rows, const int op1Cols, const int op2Rows, const int op2Cols, int const accRows, const int accCols);
void fill(float *buffer, const float value, const int len);
void assertfy(float *buffer, const float value, const int len);
void printUpTo(float *buffer, const int len);


void matMul(float* op1, float* op2, float* acc, 
            const int op1Rows, const int op1Cols, const int op2Rows,
            const int op2Cols, int const accRows, const int accCols)
{
    int row = 0x0,
        col = 0x0,
        k = 0x0;

    for(; row < op1Rows; row++)
    {
        for(col = 0x0; col < op2Cols; col++)
        {
            float scalarProduct = 0.0f;

            for(k = 0x0; k < op1Cols; k++)
            {
                scalarProduct += op1[row * op1Cols + k] * op2[k * op2Cols + col];
            }
            acc[row * accCols + col] = scalarProduct;
        }
    }
}


void fill(float *buffer, const float value, const int len)
{
    int i = 0x0;
    for(; i < len; i++)
    {
        buffer[i] = value;
    }
}


void assertfy(float *buffer, const float value, const int len)
{
    int i = 0x0;
    for(; i < len; i++)
    {
        assert(buffer[i] == value);
    }
}


void printUpTo(float *buffer, const int len)
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
void exec_cuda_std(const int ky, const int kx);
void exec_cuda_tiled(const int ky, const int kx);


void exec_host_only()
{
        float *op1,
          *op2,
          *acc;

    op1 = (float*) malloc(SIZE * sizeof(float));
    op2 = (float*) malloc(SIZE * sizeof(float));
    acc = (float*) malloc(ACC_SIZE * sizeof(float));

    clock_t begin = clock();

    fill(op1, 1.0f, SIZE);
    fill(op2, 2.0f, SIZE);

    matMul(op1, op2, acc, OP1_ROWS, OP1_COLS, OP2_ROWS, OP2_COLS, OP1_ROWS, OP2_COLS);

    clock_t end = clock();
    float elapsedTime = (float)(end - begin) / CLOCKS_PER_SEC;

    printf("Elapsed Time: %1.2f\n", elapsedTime);

    assertfy(acc, 1000.0f, ACC_SIZE);
    printUpTo(acc, 0xa);

    free(op1);
    free(op2);
    free(acc);
}


void exec_cuda_std(const int ky, const int kx)
{
    float *op1,
          *op2,
          *acc;

    cudaMallocManaged((void**) &op1, SIZE * sizeof(float), cudaMemAttachGlobal);
    cudaMallocManaged((void**) &op2, SIZE * sizeof(float), cudaMemAttachGlobal);
    cudaMallocManaged((void**) &acc, ACC_SIZE * sizeof(float), cudaMemAttachGlobal);

    dim3 blockDim(0x400, 0x1, 0x1);
    dim3 gridDim(ceil(SIZE / (float) 0x400), 0x1, 0x1);

    cudaFill<<<gridDim, blockDim>>>(op1, 1.0f, SIZE);
    cudaFill<<<gridDim, blockDim>>>(op2, 2.0f, SIZE);

    dim3 blockDim_(ky, kx, 0x1);
    dim3 gridDim_(0x20, 0x20, 0x1);

    printf("[*] gridDim(%d, %d)\n", gridDim_.x, gridDim_.y);
    printf("[*] blockDim(%d, %d)\n", blockDim_.x, blockDim_.y);

    cudaStdMatMult<<<gridDim_, blockDim_>>>(op1, op2, acc, OP1_ROWS, OP1_COLS, OP2_ROWS, OP2_COLS, OP1_ROWS, OP2_COLS);

    cudaDeviceSynchronize();

    assertfy(acc, 1000.0f, ACC_SIZE);
    printUpTo(acc, 0xa);

    cudaFree(op1);
    cudaFree(op2);
    cudaFree(acc);
}


void exec_cuda_tiled(const int ky, const int kx)
{
    float *op1,
          *op2,
          *acc;

    cudaMallocManaged((void**) &op1, SIZE * sizeof(float), cudaMemAttachGlobal);
    cudaMallocManaged((void**) &op2, SIZE * sizeof(float), cudaMemAttachGlobal);
    cudaMallocManaged((void**) &acc, ACC_SIZE * sizeof(float), cudaMemAttachGlobal);

    dim3 blockDim(0x400, 0x1, 0x1);
    dim3 gridDim(ceil(SIZE / (float) 0x400), 0x1, 0x1);

    cudaFill<<<gridDim, blockDim>>>(op1, 1.0f, SIZE);
    cudaFill<<<gridDim, blockDim>>>(op2, 2.0f, SIZE);

    dim3 blockDim_(ky, kx, 0x1);
    dim3 gridDim_(0x20, 0x20, 0x1);

    printf("[*] gridDim(%d, %d)\n", gridDim_.x, gridDim_.y);
    printf("[*] blockDim(%d, %d)\n", blockDim_.x, blockDim_.y);

    cudaTiledMatMul<<<gridDim_, blockDim_>>>(op1, op2, acc, OP1_ROWS, OP1_COLS, OP2_ROWS, OP2_COLS, OP1_ROWS, OP2_COLS);

    cudaDeviceSynchronize();

    assertfy(acc, 1000.0f, ACC_SIZE);
    printUpTo(acc, 0xa);

    cudaFree(op1);
    cudaFree(op2);
    cudaFree(acc);
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
        exec_cuda_std(KER_SIZE, KER_SIZE);
    }
    else if(strncmp(argv[0x1], "-t", 0x2) == 0x0)
    {
        exec_cuda_tiled(KER_SIZE, KER_SIZE);
    }
    else
    {
        printf("Usage: %s -[cgt]\n", argv[0x0]);
        return 0x11;
    }
    
    return 0x0;
}