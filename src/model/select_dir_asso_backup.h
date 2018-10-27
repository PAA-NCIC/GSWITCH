
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
0 - Push
1 - Pull


Simply include this file to your project and use it
*/

#include <vector>

inline int select_dir_asso_backup(const std::vector<double> & feature_vector) 
{
	if (feature_vector.at(0) <= 884465.0) {
		if (feature_vector.at(1) <= 504168.0) {
			if (feature_vector.at(3) <= 48.3) {
				if (feature_vector.at(5) <= 0.08) {
					return 0;
				}
				else {
					if (feature_vector.at(4) <= 2919.0) {
						if (feature_vector.at(4) <= 9.5) {
							return 1;
						}
						else {
							if (feature_vector.at(3) <= 6.67) {
								if (feature_vector.at(6) <= 1.0) {
									return 0;
								}
								else {
									return 1;
								}
							}
							else {
								if (feature_vector.at(3) <= 10.23) {
									return 1;
								}
								else {
									return 1;
								}
							}
						}
					}
					else {
						return 0;
					}
				}
			}
			else {
				return 0;
			}
		}
		else {
			if (feature_vector.at(0) <= 368761.0) {
				return 0;
			}
			else {
				if (feature_vector.at(0) <= 392328.5) {
					if (feature_vector.at(6) <= 0.96) {
						return 0;
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
	}
	else {
		if (feature_vector.at(6) <= 0.75) {
			return 0;
		}
		else {
			if (feature_vector.at(4) <= 329345.5) {
				if (feature_vector.at(0) <= 1517755.0) {
					if (feature_vector.at(4) <= 41.5) {
						return 0;
					}
					else {
						if (feature_vector.at(5) <= 0.55) {
							return 0;
						}
						else {
							if (feature_vector.at(6) <= 0.81) {
								return 0;
							}
							else {
								if (feature_vector.at(5) <= 0.91) {
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
					if (feature_vector.at(1) <= 119453000.0) {
						if (feature_vector.at(2) <= 3.51) {
							if (feature_vector.at(5) <= 0.92) {
								return 1;
							}
							else {
								if (feature_vector.at(4) <= 83.5) {
									return 0;
								}
								else {
									return 1;
								}
							}
						}
						else {
							if (feature_vector.at(3) <= 7.62) {
								return 0;
							}
							else {
								if (feature_vector.at(2) <= 4.36) {
									return 0;
								}
								else {
									return 1;
								}
							}
						}
					}
					else {
						return 0;
					}
				}
			}
			else {
				return 0;
			}
		}
	}
}