/*
Author: Jack Newcomb
Class: ECE6122
Last Date Modified: 11/8/2025
Description:

The plate header. Provides a class that defines a plate, with attributes for size, initial temperatures, and lengths.
Also declares the GPU update function.

*/

#pragma once

#include <cmath>
#include <iostream>
#include <vector>

class Plate
{
  public:
    /**
     * @brief Plate constructor. Takes in an int representing the total number of points in the grid, and constructs a
     * Plate
     * @param The total number of points in the grid
     */
    Plate(const int points);

    /**
     * @brief Getter for the grid
     *
     * @return A reference to the grid (a std::vector<double>)
     */
    std::vector<double> &getGrid()
    {
        return grid_;
    }

    /**
     * @brief Getter for the new grid, which gets swapped with grid each iteration by the update
     *
     * @return A reference to the new grid (a std::vector<double>)
     */
    std::vector<double> &getNewGrid()
    {
        return newGrid_;
    }

    /**
     * @brief Getter for the number of points per side
     *
     * @return An int representing the number of points making up the length of the grid
     */
    int getPointsPerSide()
    {
        return pointsPerSide;
    }

  private:
    std::vector<double> grid_;    // A flattened grid of doubles representing temperature
    std::vector<double> newGrid_; // Identical structure as grid_, allows for multi-threaded updates to
                                  // the grid without making threads overwrite the same grid concurrently

    int pointsPerSide{0};       // The length of any given side of the grid
    double sideLength{10.0};    // ft
    double filamentLength{4.0}; // ft

    double edgeTemp{20.0};      // C
    double filamentTemp{100.0}; // C

    double pointLength{0.0}; // ft
};
