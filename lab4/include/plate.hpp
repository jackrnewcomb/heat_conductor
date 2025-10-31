
#include <vector>
#include <iostream>

class Plate {
public:
	Plate(int points);
	void updatePoint(int i, int j);
	std::vector<double> getGrid() { return grid_; }
	std::vector<double> getNewGrid() { return newGrid_; }
	int getPointsPerSide() { return pointsPerSide; }
	void updateGPU(double* h_grid, double* h_newGrid, int n, int numIterations);
private:

	std::vector<double> grid_;     
	std::vector<double> newGrid_;   


	int pointsPerSide{ 0 };
	double sideLength{ 10.0 }; //ft
	double filamentLength{ 4.0 }; //ft


	double edgeTemp{ 20.0 }; //C
	double filamentTemp{ 100.0 }; //C

	double pointLength{ 0.0 };
};