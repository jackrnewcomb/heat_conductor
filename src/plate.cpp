/*
Author: Jack Newcomb
Class: ECE6122
Last Date Modified: 11/8/2025
Description:

The plate implementation. Provides an implementation of the constructor that was declared in the header: Initializes
members, fills the grid with initial temperatures, etc.

*/

#include "plate.hpp"

/**
 * @brief Plate constructor. Takes in an int representing the total number of points in the grid, and constructs a
 * Plate
 * @param The total number of points in the grid
 */
Plate::Plate(int points)
{
    // Ensure that the user inputted a perfect square as the number of points
    if ((floor(sqrt(points)) != ceil(sqrt(points))))
    {
        std::cout << "Input size to Plate is not a perfect square. Aborting.";
        return;
    }

    // Initialize pointsPerSide and pointsLength
    pointsPerSide = int(sqrt(points));
    pointLength = sideLength / pointsPerSide;

    // initialize grid_ with 0s
    std::vector<double> emptyRow;
    for (int i = 0; i < pointsPerSide * pointsPerSide; ++i)
    {
        grid_.push_back(0.0);
    }

    // Lets fill the edges with the correct starting temperatures

    // Determine the number of points that are 100 degree filaments
    auto numberOfFilamentPoints = int((filamentLength / sideLength) * pointsPerSide);

    // Determine the locations of the filament points
    auto midPoint = int(pointsPerSide / 2);
    std::vector<int> filamentIndexSpread{midPoint - (numberOfFilamentPoints / 2),
                                         midPoint + (numberOfFilamentPoints / 2)};

    // Work through the grid, and if a point is an edge point, append it with the proper starting temperature
    for (int i = 0; i < pointsPerSide; i++)
    {
        for (int j = 0; j < pointsPerSide; j++)
        {
            // If its a filament edge point...
            if ((i >= filamentIndexSpread[0] && i <= filamentIndexSpread[1]) && j == pointsPerSide - 1)
            {
                // Set it to 100 degrees
                grid_[i * pointsPerSide + j] = filamentTemp;
            }
            // else if its a usual edge point...
            else if (i == 0 || i == pointsPerSide - 1 || j == 0 || j == pointsPerSide - 1)
            {
                // Set it to 20 degrees
                grid_[i * pointsPerSide + j] = edgeTemp;
            }
        }
    }

    // Copy to newGrid_
    newGrid_ = grid_;
}
