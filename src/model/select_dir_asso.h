
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

inline int select_dir_asso(const std::vector<double> & feature_vector) 
{
	if (feature_vector.at(1) <= 1.93) {
		if (feature_vector.at(1) <= 0.97) {
			if (feature_vector.at(1) <= 0.74) {
				if (feature_vector.at(1) <= 0.57) {
					if (feature_vector.at(6) <= 1.0) {
						return 1;
					}
					else {
						return 1;
					}
				}
				else {
					if (feature_vector.at(0) <= 0.3) {
						return 1;
					}
					else {
						return 1;
					}
				}
			}
			else {
				if (feature_vector.at(2) <= 1.06) {
					if (feature_vector.at(1) <= 0.75) {
						return 0;
					}
					else {
						return 1;
					}
				}
				else {
					if (feature_vector.at(5) <= 156.78) {
						return 1;
					}
					else {
						return 0;
					}
				}
			}
		}
		else {
			if (feature_vector.at(2) <= 1.05) {
				if (feature_vector.at(4) <= 21449.44) {
					if (feature_vector.at(1) <= 1.67) {
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
				if (feature_vector.at(7) <= 1.0) {
					if (feature_vector.at(6) <= 4.97) {
						return 0;
					}
					else {
						return 1;
					}
				}
				else {
					if (feature_vector.at(0) <= 0.06) {
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
			if (feature_vector.at(1) <= 95.01) {
				if (feature_vector.at(7) <= 1.0) {
					if (feature_vector.at(0) <= 2.27) {
						return 0;
					}
					else {
						return 1;
					}
				}
				else {
					if (feature_vector.at(4) <= 667.21) {
						return 0;
					}
					else {
						return 1;
					}
				}
			}
			else {
				return 0;
			}
		}
		else {
			if (feature_vector.at(1) <= 2.84) {
				if (feature_vector.at(0) <= 3.0) {
					if (feature_vector.at(2) <= 0.9) {
						return 0;
					}
					else {
						return 0;
					}
				}
				else {
					if (feature_vector.at(6) <= 9.86) {
						return 1;
					}
					else {
						return 0;
					}
				}
			}
			else {
				if (feature_vector.at(1) <= 5.09) {
					if (feature_vector.at(1) <= 5.09) {
						return 0;
					}
					else {
						return 1;
					}
				}
				else {
					if (feature_vector.at(7) <= 1.0) {
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