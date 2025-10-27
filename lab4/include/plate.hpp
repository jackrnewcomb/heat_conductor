
#include <vector>
#include <iostream>

class Plate {
public:
	Plate(int points);
	
private:

	std::vector<std::vector<double>> grid_;

	int pointsPerSide{ 0 };
	double sideLength{ 10.0 }; //ft
	double filamentLength{ 4.0 }; //ft


	double edgeTemp{ 20.0 }; //C
	double filamentTemp{ 100.0 }; //C

	double pointLength{ 0.0 };
};