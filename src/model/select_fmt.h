
/*
This inline function was automatically generated using DecisionTreeToCpp Converter

It takes feature vector as single argument:
feature_vector[0] - r_act(v)
feature_vector[1] - e1
feature_vector[2] - e2


It returns index of predicted class:
0 - Queue
1 - Bitmap


Simply include this file to your project and use it
*/

#include <vector>

inline int select_fmt(const std::vector<double> & feature_vector) 
{
	if (feature_vector.at(0) <= 0.06) {
		return 0;
	}
	else {
		return 1;
	}
}
