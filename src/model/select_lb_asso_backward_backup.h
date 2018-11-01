
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

inline int select_lb_asso_backward_backup(const std::vector<double> & feature_vector) 
{
	if (feature_vector.at(0) <= 61044.5) {
		if (feature_vector.at(3) <= 3.85) {
			if (feature_vector.at(2) <= 14.52) {
				if (feature_vector.at(4) <= 25.0) {
					if (feature_vector.at(6) <= 1.0) {
						return 0;
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
				return 2;
			}
		}
		else {
			if (feature_vector.at(1) <= 1851420.5) {
				if (feature_vector.at(0) <= 54444.5) {
					if (feature_vector.at(1) <= 1596441.0) {
						return 2;
					}
					else {
						return 2;
					}
				}
				else {
					if (feature_vector.at(5) <= 0.16) {
						return 0;
					}
					else {
						return 2;
					}
				}
			}
			else {
				if (feature_vector.at(0) <= 18012.5) {
					return 2;
				}
				else {
					if (feature_vector.at(2) <= 95.33) {
						return 2;
					}
					else {
						return 0;
					}
				}
			}
		}
	}
	else {
		if (feature_vector.at(4) <= 1619.0) {
			if (feature_vector.at(3) <= 2.74) {
				if (feature_vector.at(2) <= 7.99) {
					if (feature_vector.at(6) <= 1.0) {
						return 0;
					}
					else {
						return 0;
					}
				}
				else {
					if (feature_vector.at(3) <= 1.24) {
						return 2;
					}
					else {
						return 0;
					}
				}
			}
			else {
				if (feature_vector.at(1) <= 1965561.5) {
					if (feature_vector.at(2) <= 7.16) {
						return 0;
					}
					else {
						return 2;
					}
				}
				else {
					if (feature_vector.at(4) <= 38.5) {
						return 2;
					}
					else {
						return 0;
					}
				}
			}
		}
		else {
			if (feature_vector.at(1) <= 12099634.0) {
				if (feature_vector.at(3) <= 17.6) {
					if (feature_vector.at(5) <= 0.55) {
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
				return 0;
			}
		}
	}
}
