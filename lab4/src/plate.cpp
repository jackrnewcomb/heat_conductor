#include "plate.hpp"

Plate::Plate(int points) {
	// add check here that points is a square
	pointsPerSide = int(sqrt(points));

	pointLength = sideLength / pointsPerSide;

	// initialize grid_
	std::vector<double> emptyRow;

	for (int i = 0; i < pointsPerSide*pointsPerSide; ++i) {
		grid_.push_back(0.0);
	}


	//Fill the edges with the correct temperatures 

	auto number_of_filament_points = int(4 * (10.0/pointsPerSide));//number of filament points isn't an int...rounding for now

	auto mid_point = int(pointsPerSide / 2);
	std::vector<int> filament_index_spread{ mid_point - (number_of_filament_points / 2), mid_point + (number_of_filament_points / 2) };


	for (int i = 0; i < pointsPerSide; i++) {
		for (int j = 0; j < pointsPerSide; j++) {
			if ((i >= filament_index_spread[0] && i <= filament_index_spread[1]) && j == pointsPerSide - 1) {
				grid_[i*pointsPerSide + j] = 100;
			} else if (i == 0 || i == pointsPerSide-1 || j == 0 || j == pointsPerSide-1) {
				grid_[i*pointsPerSide + j] = 20;
			}
		}
	}
	newGrid_ = grid_;


}

void Plate::updatePoint(int i, int j)
{
	double tempSum{0.0};
	//assumes not updating an edge point...
	tempSum += grid_[(i - 1) * pointsPerSide + j];
	tempSum += grid_[(i + 1) * pointsPerSide + j];
	tempSum += grid_[i * pointsPerSide + j - 1];
	tempSum += grid_[i * pointsPerSide + j + 1];
		
	double newTemp = tempSum / 4.0;
	//std::cout << "Based on temps " << grid_[i - 1][j] <<", " << grid_[i + 1][j] << ", " << grid_[i ][j-1] << ", and " << grid_[i][j+1] << "...new temp = " << newTemp << "\n";
	newGrid_[i * pointsPerSide + j] = newTemp;



}

void Plate::update() {
	//std::cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << "\n";
	for (int i = 1; i < pointsPerSide - 1; i++) {
		for (int j = 1; j < pointsPerSide-1; j++)
			updatePoint(i, j);
	}
	
	// update grid to newgrid
	grid_ = newGrid_;


}