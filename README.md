# PCA-EXP-6-MATRIX-TRANSPOSITION-USING-SHARED-MEMORY-AY-23-24
<h3>AIM:</h3>
<h3>ENTER YOUR NAME</h3>
G Sushanth

<h3>ENTER YOUR REGISTER NO</h3>  212225230088

<h3>EX. NO</h3>
<h3>DATE</h3>
<h1> <align=center> MATRIX TRANSPOSITION USING SHARED MEMORY </h3>
  Implement Matrix transposition using GPU Shared memory.</h3>

## AIM:
To perform Matrix Multiplication using Transposition using shared memory.

## EQUIPMENTS REQUIRED:
Hardware – PCs with NVIDIA GPU & CUDA NVCC
Google Colab with NVCC Compiler

## PROCEDURE:
 CUDA_SharedMemory_AccessPatterns:

1. Begin Device Setup
    1.1 Select the device to be used for computation
    1.2 Retrieve the properties of the selected device
2. End Device Setup

3. Begin Array Size Setup
    3.1 Set the size of the array to be used in the computation
    3.2 The array size is determined by the block dimensions (BDIMX and BDIMY)
4. End Array Size Setup

5. Begin Execution Configuration
    5.1 Set up the execution configuration with a grid and block dimensions
    5.2 In this case, a single block grid is used
6. End Execution Configuration

7. Begin Memory Allocation
    7.1 Allocate device memory for the output array d_C
    7.2 Allocate a corresponding array gpuRef in the host memory
8. End Memory Allocation

9. Begin Kernel Execution
    9.1 Launch several kernel functions with different shared memory access patterns (Use any two patterns)
        9.1.1 setRowReadRow: Each thread writes to and reads from its row in shared memory
        9.1.2 setColReadCol: Each thread writes to and reads from its column in shared memory
        9.1.3 setColReadCol2: Similar to setColReadCol, but with transposed coordinates
        9.1.4 setRowReadCol: Each thread writes to its row and reads from its column in shared memory
        9.1.5 setRowReadColDyn: Similar to setRowReadCol, but with dynamic shared memory allocation
        9.1.6 setRowReadColPad: Similar to setRowReadCol, but with padding to avoid bank conflicts
        9.1.7 setRowReadColDynPad: Similar to setRowReadColPad, but with dynamic shared memory allocation
10. End Kernel Execution

11. Begin Memory Copy
    11.1 After each kernel execution, copy the output array from device memory to host memory
12. End Memory Copy

13. Begin Memory Free
    13.1 Free the device memory and host memory
14. End Memory Free

15. Reset the device

16. End of Algorithm

## PROGRAM:
```
#include <stdio.h>
#include <cuda_runtime.h>

#define BDIMX 4
#define BDIMY 4

__global__ void setRowReadRow(int *C)
{
    __shared__ int tile[BDIMY][BDIMX];

    int idx = threadIdx.y * blockDim.x + threadIdx.x;

    tile[threadIdx.y][threadIdx.x] = idx;

    __syncthreads();

    C[idx] = tile[threadIdx.y][threadIdx.x];
}

__global__ void setRowReadCol(int *C)
{
    __shared__ int tile[BDIMY][BDIMX];

    int idx = threadIdx.y * blockDim.x + threadIdx.x;

    tile[threadIdx.y][threadIdx.x] = idx;

    __syncthreads();

    C[idx] = tile[threadIdx.x][threadIdx.y];
}

int main()
{
    int size = BDIMX * BDIMY;

    int *d_C;
    int *gpuRef;

    gpuRef = (int *)malloc(size * sizeof(int));

    cudaMalloc((void **)&d_C, size * sizeof(int));

    dim3 block(BDIMX, BDIMY);
    dim3 grid(1, 1);

    cudaEvent_t start, stop;
    float elapsedTime;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaMemset(d_C, 0, size * sizeof(int));

    cudaEventRecord(start);

    setRowReadRow<<<grid, block>>>(d_C);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&elapsedTime, start, stop);

    cudaMemcpy(gpuRef, d_C, size * sizeof(int),
               cudaMemcpyDeviceToHost);

    printf("setRowReadRow Output:\n");

    for (int i = 0; i < BDIMY; i++)
    {
        for (int j = 0; j < BDIMX; j++)
        {
            printf("%d ", gpuRef[i * BDIMX + j]);
        }
        printf("\n");
    }

    printf("setRowReadRow Elapsed Time: %f ms\n", elapsedTime);

    cudaMemset(d_C, 0, size * sizeof(int));

    cudaEventRecord(start);

    setRowReadCol<<<grid, block>>>(d_C);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&elapsedTime, start, stop);

    cudaMemcpy(gpuRef, d_C, size * sizeof(int),
               cudaMemcpyDeviceToHost);

    printf("\nsetRowReadCol Output:\n");

    for (int i = 0; i < BDIMY; i++)
    {
        for (int j = 0; j < BDIMX; j++)
        {
            printf("%d ", gpuRef[i * BDIMX + j]);
        }
        printf("\n");
    }

    printf("setRowReadCol Elapsed Time: %f ms\n", elapsedTime);

    cudaFree(d_C);
    free(gpuRef);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaDeviceReset();

    return 0;
}
```


## OUTPUT:
<img width="440" height="301" alt="643985178-3f9f7b66-9374-4686-9205-6680f0c48bd9" src="https://github.com/user-attachments/assets/ceeb314b-c447-4d65-be66-451143e31e17" />

## RESULT:
 Thus, the program has been successfully executed using CUDA to transpose a matrix using shared memory. Two shared memory access patterns, setRowReadRow and setRowReadCol, were implemented. The elapsed times were recorded as 96.898941 ms and 0.044192 ms, respectively. It is observed that different shared memory access patterns result in variations in execution time.
