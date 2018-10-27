
/*
This inline function was automatically generated using DecisionTreeToCpp Converter

It takes feature vector as single argument:
feature_vector[0] - nvertex
feature_vector[1] - nedges
feature_vector[2] - avg(d)
feature_vector[3] - std(d)
feature_vector[4] - r(d)
feature_vector[5] - GI
feature_vector[6] - Her


It returns index of predicted class:
0 - WM
1 - CM
2 - STRICT


Simply include this file to your project and use it
*/

#include <vector>

inline int select_lb_asso_forward_backup(const std::vector<double> & feature_vector) 
{
	if (feature_vector.at(3) <= 14.74) {
		if (feature_vector.at(0) <= 17620.5) {
			if (feature_vector.at(3) <= 3.17) {
				if (feature_vector.at(2) <= 99.72) {
					return 0;
				}
				else {
					return 2;
				}
			}
			else {
				return 2;
			}
		}
		else {
			if (feature_vector.at(4) <= 122.0) {
				if (feature_vector.at(1) <= 120416.0) {
					if (feature_vector.at(0) <= 35166.0) {
						return 0;
					}
					else {
						return 2;
					}
				}
				else {
					if (feature_vector.at(3) <= 8.31) {
						return 0;
					}
					else {
						return 0;
					}
				}
			}
			else {
				if (feature_vector.at(1) <= 560493.5) {
					if (feature_vector.at(3) <= 7.06) {
						return 2;
					}
					else {
						return 2;
					}
				}
				else {
					if (feature_vector.at(0) <= 704131.5) {
						return 0;
					}
					else {
						return 2;
					}
				}
			}
		}
	}
	else {
		if (feature_vector.at(3) <= 20.65) {
			if (feature_vector.at(1) <= 1338941.0) {
				if (feature_vector.at(6) <= 0.94) {
					if (feature_vector.at(1) <= 705305.0) {
						return 2;
					}
					else {
						return 0;
					}
				}
				else {
					return 2;
				}
			}
			else {
				if (feature_vector.at(3) <= 15.23) {
					if (feature_vector.at(6) <= 0.96) {
						return 0;
					}
					else {
						return 2;
					}
				}
				else {
					if (feature_vector.at(4) <= 3902.0) {
						return 0;
					}
					else {
						return 2;
					}
				}
			}
		}
		else {
			if (feature_vector.at(4) <= 98.0) {
				if (feature_vector.at(0) <= 28923.5) {
					return 2;
				}
				else {
					if (feature_vector.at(4) <= 92.5) {
						return 2;
					}
					else {
						return 0;
					}
				}
			}
			else {
				if (feature_vector.at(1) <= 7260351.0) {
					if (feature_vector.at(1) <= 2222551.5) {
						return 2;
					}
					else {
						return 2;
					}
				}
				else {
					if (feature_vector.at(0) <= 168341.0) {
						return 0;
					}
					else {
						return 2;
					}
				}
			}
		}
	}
}
