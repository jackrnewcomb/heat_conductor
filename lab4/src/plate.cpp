#include "plate.hpp"

Plate::Plate(int points) {
	// add check here that points is a square
	pointsPerSide = int(sqrt(points));
	std::cout << "Points per side: " << pointsPerSide << "\n";
	pointLength = sideLength / pointsPerSide;

	// initialize grid_
	std::vector<double> emptyRow;

	for (int i = 0; i < pointsPerSide; ++i) {
		emptyRow.push_back(0.0);
	}

	for (int i = 0; i < pointsPerSide; ++i) {
		grid_.push_back(emptyRow);
	}

	std::cout << "grid columns: " << grid_.size() << "\n";

	//Fill the edges with the correct temperatures 

	auto number_of_filament_points = int(4 * (10.0/pointsPerSide));//number of filament points isn't an int...rounding for now

	auto mid_point = int(pointsPerSide / 2);
	std::vector<int> filament_index_spread{ mid_point - (number_of_filament_points / 2), mid_point + (number_of_filament_points / 2) };


	std::cout << "Filmanent from point " << filament_index_spread[0] << " to " << filament_index_spread[1] << "\n";

	for (int i = 0; i < grid_.size(); i++) {
		for (int j = 0; j < grid_[0].size(); j++) {
			if ((i >= filament_index_spread[0] && i <= filament_index_spread[1]) && j == grid_.size() - 1) {
				grid_[i][j] = 100;
			} else if (i == 0 || i == grid_.size()-1 || j == 0 || j == grid_.size()-1) {
				grid_[i][j] = 20;
			}
		}
	}

	for (auto& row : grid_) {
		for (auto& val : row) {
			std::cout << val << " ";
		}
		std::cout << "\n";
	}
}