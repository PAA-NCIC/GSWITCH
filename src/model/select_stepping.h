
/*
This inline function was automatically generated using DecisionTreeToCpp Converter

It takes feature vector as single argument:
feature_vector[0] - act(v)
feature_vector[1] - act(e)
feature_vector[2] - e2


It returns index of predicted class:
0 - Increase
1 - Decrease
2 - Remain


Simply include this file to your project and use it
*/

#include <vector>

inline int select_stepping(const std::vector<double> & feature_vector) 
{
	if (feature_vector.at(1) <= 3328.06) {
	  if (feature_vector.at(0) <= 12.42) {
		  return 0;
	  }
    else {
      return 0;
    }
	}
	else {
	  if (feature_vector.at(1) <= 33679.37) {
		  return 2;
	  }
    else {
      return 1;
    }
	}
}
