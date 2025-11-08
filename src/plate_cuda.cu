/*
Author: Jack Newcomb
Class: ECE6122
Last Date Modified: 11/8/2025
Description:

The cuda GPU implementation. Provides an update kernel to update a specific index, and the GPU update function.

*/

#include <cuda_runtime.h>
#include <cstdio>
#include <iostream>

/**
 * @brief Update kernel. Provides an update method for a given point index
 * @param A pointer to the current grid
 * @param A pointer to the new grid
 * @param The point index to update
 */
__global__ void updateKernel(double* grid, double* newGrid, int n) {
    // Get the row and column
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    // If we are NOT on an edge point...
    if (i > 0 && i < n-1 && j > 0 && j < n-1) {
        // Get the index within the grid
        int idx = i * n + j;

        // Get the sum of temperatures
        double tempSum = 0.0;
        tempSum += grid[(i - 1) * n + j];  // Neighbor up
        tempSum += grid[(i + 1) * n + j];  // Neighbor down
        tempSum += grid[i * n + (j - 1)];  // Neighbor left
        tempSum += grid[i * n + (j + 1)];  // Neighbor right

        // Write to the new grid with the average
        newGrid[idx] = tempSum / 4.0;
    }
}

/**
 * @brief GPU update function. Offloads multi-threaded updates to grid points to the GPU via cuda.
 * @param A pointer to the host memory containing the grid
 * @param A pointer to the host memory for the new grid
 * @param The number of points on any given side of the grid
 * @param The number of times to perform the update
 */
void updateGPU(double* hostGrid, double* hostNewGrid, int pointsPerSide, int iterations) {
    // Initialize pointers for the grid and new grid
    double *gpuGrid, *gpuNewGrid;

    // Determine the total memory required for the grids (the square of the length of a side, times the size of a double
    size_t size = pointsPerSide * pointsPerSide * sizeof(double);

    // Allocate memory on the GPU for the grid and new grid 
    cudaMalloc(&gpuGrid, size);
    cudaMalloc(&gpuNewGrid, size);

    // Copy the info from the host grids to the GPU-memory grids we just made
    cudaMemcpy(gpuGrid, hostGrid, size, cudaMemcpyHostToDevice);
    cudaMemcpy(gpuNewGrid, hostNewGrid, size, cudaMemcpyHostToDevice);

    // Defines the number of threads per block (TODO: maybe try different threads per block values? 16x16 seems reasonable considering max GPU thread count is 1024)
    dim3 threadsPerBlock(16,16);
    dim3 numBlocks((pointsPerSide + 15)/16, (pointsPerSide + 15)/16);

    // Initialize start and stop events 
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Start the timer
    cudaEventRecord(start);

    // For each iteration requested by the user...
    for(int iter = 0; iter < iterations; ++iter) {
        // Call the update kernel
        updateKernel<<<numBlocks, threadsPerBlock>>>(gpuGrid, gpuNewGrid, pointsPerSide);

        // Wait for all kernels to finish
        cudaDeviceSynchronize();

        // Swaps pointers to GPU grids: the grid is now the newGrid, becoming the source for the next iteration
        std::swap(gpuGrid, gpuNewGrid);
    }

    // Stop the timer
    cudaEventRecord(stop);
    cudaEventSynchronize(stop); // wait for the event to complete

    //Compute time elapsed, and print it to the console
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Solution took " << milliseconds << " ms\n";



    // Result of a bug in which, depending on whether we were on an odd or even iteration, either grid or new grid might have the latest temps
    // Copy memory from the GPU grid back to the host grid
    if (iterations % 2 == 0)
        cudaMemcpy(hostGrid, gpuGrid, size, cudaMemcpyDeviceToHost);
    else
        cudaMemcpy(hostGrid, gpuNewGrid, size, cudaMemcpyDeviceToHost);

    // Free GPU memory for the grids, prevents memory leaks
    cudaFree(gpuGrid);
    cudaFree(gpuNewGrid);
}

