#include <map>
#include <string>
#include <iostream>
#include <vector>
#include <algorithm>
#include <fstream>
#include "plate.hpp"
#include "plate_cuda.hpp"


//Gonna have to clean all this up later, but i'm on a train with just my laptop so this is what we're getting
//No clang cuz laptop 


bool sanitizeArgMap(std::map<std::string, std::string>& map)
{

    // Check for each pair in the map...
    for (auto& pair : map)
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

int main(int argc, char* argv[])
{
    // maps arg types to their arg
    std::map<std::string, std::string> argMap = {
        {"-N", "256"},   // default number of threads
        {"-I", "10000"}   // default cell size
    };

    // For each arg passed to the command line...
    for (int i = 1; i < argc; i++)
    {
        // Cast to a string
        std::string arg = argv[i];

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
        return -1;
    }

    Plate plate(256);

    // run GPU simulation
    updateGPU(plate.getGrid().data(), plate.getNewGrid().data(),
        plate.getPointsPerSide(), 10000);

    // write to CSV as before


    auto final_plate = plate.getGrid();


    // Open a file in write mode
    std::ofstream file("finalTemperatures.csv");

    int n = plate.getPointsPerSide();

    for (int i = 0; i < final_plate.size(); i++) {
        file << final_plate[i];
        if ((i + 1) % n == 0)  // end of row
            file << "\n";
        else
            file << ",";       // between elements
    }
    
    return 0;
    
}