#include <cuda_runtime.h>

__global__ void updateKernel(double* grid, double* newGrid, int n) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;  // row
    int j = blockIdx.x * blockDim.x + threadIdx.x;  // column

    if (i > 0 && i < n-1 && j > 0 && j < n-1) {
        int idx = i * n + j;
        double tempSum = grid[(i-1)*n + j] + grid[(i+1)*n + j] +
                         grid[i*n + (j-1)] + grid[i*n + (j+1)];
        newGrid[idx] = tempSum / 4.0;
    }
}

void updateGPU(double* h_grid, double* h_newGrid, int n, int numIterations) {
    double *d_grid, *d_newGrid;
    size_t size = n * n * sizeof(double);

    cudaMalloc(&d_grid, size);
    cudaMalloc(&d_newGrid, size);

    cudaMemcpy(d_grid, h_grid, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_newGrid, h_newGrid, size, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16,16);
    dim3 numBlocks((n + 15)/16, (n + 15)/16);

    for(int iter = 0; iter < numIterations; ++iter) {
        updateKernel<<<numBlocks, threadsPerBlock>>>(d_grid, d_newGrid, n);
        cudaDeviceSynchronize();
        std::swap(d_grid, d_newGrid);
    }

    cudaMemcpy(h_grid, d_grid, size, cudaMemcpyDeviceToHost);

    cudaFree(d_grid);
    cudaFree(d_newGrid);
}
