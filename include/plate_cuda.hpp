/*
Author: Jack Newcomb
Class: ECE6122
Last Date Modified: 11/8/2025
Description:

The plate cuda header. Provides a declaration for the GPU update function.

*/

#pragma once

/**
 * @brief GPU update function. Offloads multi-threaded updates to grid points to the GPU via cuda.
 * @param A pointer to the host memory containing the grid
 * @param A pointer to the host memory for the new grid
 * @param The number of points on any given side of the grid
 * @param The number of times to perform the update
 */
void updateGPU(double *hostGrid, double *hostNewGrid, int pointsPerSide, int iterations);
