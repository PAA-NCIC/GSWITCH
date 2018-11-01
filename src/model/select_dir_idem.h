
/*
This inline function was automatically generated using DecisionTreeToCpp Converter

It takes feature vector as single argument:
feature_vector[0] - v
feature_vector[1] - e
feature_vector[2] - rv
feature_vector[3] - re
feature_vector[4] - a/avg
feature_vector[5] - u/avg
feature_vector[6] - a/max
feature_vector[7] - u/max


It returns index of predicted class:
0 - Push
1 - Pull


Simply include this file to your project and use it
*/

#include <vector>

inline int select_dir_idem(const std::vector<double> & feature_vector) 
{
	if (feature_vector.at(1) <= 4.87) {
		if (feature_vector.at(1) <= 2.26) {
			if (feature_vector.at(1) <= 1.12) {
				if (feature_vector.at(0) <= 39188.04) {
					if (feature_vector.at(0) <= 1559.63) {
						return 1;
					}
					else {
						return 1;
					}
				}
				else {
					return 0;
				}
			}
			else {
				if (feature_vector.at(2) <= 0.83) {
					if (feature_vector.at(0) <= 30.19) {
						return 1;
					}
					else {
						return 1;
					}
				}
				else {
					if (feature_vector.at(0) <= 2.35) {
						return 1;
					}
					else {
						return 0;
					}
				}
			}
		}
		else {
			if (feature_vector.at(6) <= 5.7) {
				if (feature_vector.at(5) <= 1879.53) {
					if (feature_vector.at(4) <= 82.27) {
						return 1;
					}
					else {
						return 0;
					}
				}
				else {
					if (feature_vector.at(4) <= 3749.41) {
						return 1;
					}
					else {
						return 0;
					}
				}
			}
			else {
				if (feature_vector.at(7) <= 1.0) {
					if (feature_vector.at(2) <= 0.89) {
						return 1;
					}
					else {
						return 0;
					}
				}
				else {
					if (feature_vector.at(1) <= 4.29) {
						return 1;
					}
					else {
						return 0;
					}
				}
			}
		}
	}
	else {
		if (feature_vector.at(7) <= 1.0) {
			if (feature_vector.at(2) <= 0.61) {
				if (feature_vector.at(0) <= 13.13) {
					return 1;
				}
				else {
					return 0;
				}
			}
			else {
				if (feature_vector.at(0) <= 126.08) {
					if (feature_vector.at(4) <= 36.84) {
						return 1;
					}
					else {
						return 1;
					}
				}
				else {
					return 0;
				}
			}
		}
		else {
			if (feature_vector.at(1) <= 19.57) {
				if (feature_vector.at(3) <= 25.21) {
					if (feature_vector.at(5) <= 3477.37) {
						return 0;
					}
					else {
						return 1;
					}
				}
				else {
					if (feature_vector.at(5) <= 1651.31) {
						return 1;
					}
					else {
						return 0;
					}
				}
			}
			else {
				if (feature_vector.at(5) <= 986.02) {
					return 1;
				}
				else {
					if (feature_vector.at(3) <= 50.7) {
						return 0;
					}
					else {
						return 0;
					}
				}
			}
		}
	}
}