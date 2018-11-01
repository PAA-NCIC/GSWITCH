
/*
This inline function was automatically generated using DecisionTreeToCpp Converter

It takes feature vector as single argument:
feature_vector[0] - avg(d)
feature_vector[1] - std(d)
feature_vector[2] - r(d)
feature_vector[3] - GI
feature_vector[4] - Her


It returns index of predicted class:
0 - NonFused
1 - Fused


Simply include this file to your project and use it
*/

#include <vector>

inline int select_fusion(const std::vector<double> & feature_vector) 
{
	if (feature_vector.at(1) <= 17.08) {
		if (feature_vector.at(0) <= 55.47) {
			if (feature_vector.at(0) <= 43.9) {
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
		if (feature_vector.at(2) <= 83.5) {
			return 1;
		}
		else {
			if (feature_vector.at(1) <= 24.85) {
				return 0;
			}
			else {
				return 0;
			}
		}
	}
}