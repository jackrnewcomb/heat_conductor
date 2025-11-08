/*
Author: Jack Newcomb
Class: ECE6122
Last Date Modified: 11/8/2025
Description:

The main execution function, which takes in user input, constructs the plate, executes the GPU update, and writes to the
console/csv file. Also provides a function to sanitize user arguments.

*/

#include "plate.hpp"
#include "plate_cuda.hpp"
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <vector>

/**
 * @brief A function to sanitize user inputs by ensuring that they follow the input requirements (integer args,
 * following -x format)
 * @param A reference to the map of args to values
 */
bool sanitizeArgMap(const std::map<std::string, std::string> &map)
{
    // Check for each pair in the map...
    for (const auto &pair : map)
    {
        // Try casting it to an int
        try
        {
            auto test = std::stoi(pair.second);
        }
        // If it fails, print an error message and abort
        catch (...)
        {
            std::cout << "Invalid input for " << pair.first << ": Value must cast to an int. Aborting!\n";
            return false;
        }
    }
    return true;
}

/**
 * @brief main function, reads in user input and executes the gpu update
 */
int main(int argc, char *argv[])
{
    // maps arg types to their arg
    std::map<std::string, std::string> argMap = {
        {"-N", "256"},  // default grid size in points
        {"-I", "10000"} // default number of iterations
    };

    // For each arg passed to the command line...
    for (int i = 1; i < argc; i++)
    {
        // Cast to a string
        std::string arg = argv[i];

        // If -q is in the args at all, terminate the program
        if (arg == "-q")
        {
            std::cout << "Quit directive received: Terminating program\n";
            return 0;
        }

        // If the arg contains a "-", the next arg should specify a value. Add it to the map and increment the argc
        // counter
        if (arg.find("-") != std::string::npos)
        {
            argMap[arg] = argv[i + 1];
            i++;
        }
        // If the arg doesn't contain "-", the user did not format their arguments correctly. Pass a message to the
        // console and move on
        else
        {
            std::cerr << "Unexpected input " << arg << ", ignoring...\n";
        }
    }

    // Check for clean inputs
    bool clean = sanitizeArgMap(argMap);
    if (!clean)
    {
        // There was an error sanitizing the map. Abort
        return -1;
    }

    int userInputSize = std::stoi(argMap["-N"]);
    Plate plate(userInputSize);

    // run GPU simulation
    int userInputIter = std::stoi(argMap["-I"]);
    updateGPU(plate.getGrid().data(), plate.getNewGrid().data(), plate.getPointsPerSide(), userInputIter);

    // write to CSV as before

    auto final_plate = plate.getGrid();

    // Open the finalTemperatures csv
    std::ofstream file("finalTemperatures.csv");

    int n = plate.getPointsPerSide();

    // For each element in the final plate...
    for (int i = 0; i < final_plate.size(); i++)
    {
        // Write the value
        file << final_plate[i];

        // If we are at the end of a row, append a new line
        if ((i + 1) % n == 0)
        {
            file << "\n";
        }
        // Otherwise add a comma to add it to the row
        else
        {
            file << ",";
        }
    }

    return 0;
}
