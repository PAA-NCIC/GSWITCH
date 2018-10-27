
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
feature_vector[7] - ina(v)
feature_vector[8] - ina(e)
feature_vector[9] - r_ina(v)
feature_vector[10] - r_ina(e)
feature_vector[11] - cur_ina_avg(d)
feature_vector[12] - cur_ina_r(d)


It returns index of predicted class:
0 - WM
1 - CM
2 - STRICT


Simply include this file to your project and use it
*/

#include <vector>

inline int select_lb_asso_backward(const std::vector<double> & feature_vector) 
{
	if (feature_vector.at(4) <= 25.5) {
		if (feature_vector.at(2) <= 8.3) {
			if (feature_vector.at(7) <= 28770.0) {
				if (feature_vector.at(8) <= 69937.5) {
					if (feature_vector.at(4) <= 18.0) {
						return 0;
					}
					else {
						return 1;
					}
				}
				else {
					return 1;
				}
			}
			else {
				if (feature_vector.at(1) <= 267646.0) {
					if (feature_vector.at(8) <= 245397.0) {
						if (feature_vector.at(12) <= 154966.0) {
							if (feature_vector.at(10) <= 0.95) {
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
					else {
						if (feature_vector.at(9) <= 0.98) {
							return 1;
						}
						else {
							if (feature_vector.at(7) <= 81829.0) {
								if (feature_vector.at(5) <= 0.14) {
									return 0;
								}
								else {
									if (feature_vector.at(9) <= 1.0) {
										return 1;
									}
									else {
										return 0;
									}
								}
							}
							else {
								return 1;
							}
						}
					}
				}
				else {
					if (feature_vector.at(1) <= 296493.0) {
						if (feature_vector.at(12) <= 515928.0) {
							return 0;
						}
						else {
							return 1;
						}
					}
					else {
						if (feature_vector.at(3) <= 0.09) {
							if (feature_vector.at(5) <= 0.0) {
								return 0;
							}
							else {
								if (feature_vector.at(9) <= 1.0) {
									return 0;
								}
								else {
									if (feature_vector.at(10) <= 1.0) {
										return 1;
									}
									else {
										return 0;
									}
								}
							}
						}
						else {
							return 0;
						}
					}
				}
			}
		}
		else {
			if (feature_vector.at(7) <= 6569.5) {
				if (feature_vector.at(3) <= 1.79) {
					if (feature_vector.at(7) <= 776.5) {
						if (feature_vector.at(1) <= 423539.0) {
							if (feature_vector.at(7) <= 518.5) {
								return 2;
							}
							else {
								if (feature_vector.at(3) <= 1.0) {
									return 2;
								}
								else {
									return 1;
								}
							}
						}
						else {
							if (feature_vector.at(12) <= 3171.0) {
								return 2;
							}
							else {
								return 1;
							}
						}
					}
					else {
						return 1;
					}
				}
				else {
					if (feature_vector.at(0) <= 127810.0) {
						if (feature_vector.at(12) <= 104665.0) {
							if (feature_vector.at(2) <= 23.79) {
								if (feature_vector.at(7) <= 376.0) {
									if (feature_vector.at(8) <= 412.5) {
										return 0;
									}
									else {
										return 2;
									}
								}
								else {
									return 0;
								}
							}
							else {
								return 1;
							}
						}
						else {
							return 1;
						}
					}
					else {
						if (feature_vector.at(0) <= 227657.0) {
							return 1;
						}
						else {
							if (feature_vector.at(12) <= 12810.0) {
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
				if (feature_vector.at(2) <= 14.52) {
					if (feature_vector.at(2) <= 8.43) {
						if (feature_vector.at(10) <= 0.81) {
							if (feature_vector.at(11) <= 8.36) {
								return 1;
							}
							else {
								if (feature_vector.at(10) <= 0.59) {
									if (feature_vector.at(8) <= 102827.0) {
										return 1;
									}
									else {
										return 1;
									}
								}
								else {
									if (feature_vector.at(11) <= 8.36) {
										return 0;
									}
									else {
										return 1;
									}
								}
							}
						}
						else {
							if (feature_vector.at(11) <= 8.32) {
								return 1;
							}
							else {
								if (feature_vector.at(11) <= 8.33) {
									if (feature_vector.at(7) <= 23003.0) {
										return 0;
									}
									else {
										return 0;
									}
								}
								else {
									if (feature_vector.at(8) <= 166463.5) {
										return 0;
									}
									else {
										return 1;
									}
								}
							}
						}
					}
					else {
						if (feature_vector.at(6) <= 0.99) {
							if (feature_vector.at(12) <= 190125.0) {
								if (feature_vector.at(7) <= 17010.0) {
									return 1;
								}
								else {
									return 0;
								}
							}
							else {
								return 1;
							}
						}
						else {
							if (feature_vector.at(7) <= 7098.5) {
								return 1;
							}
							else {
								if (feature_vector.at(5) <= 0.14) {
									return 0;
								}
								else {
									if (feature_vector.at(11) <= 13.42) {
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
				else {
					if (feature_vector.at(3) <= 4.51) {
						if (feature_vector.at(0) <= 287056.0) {
							if (feature_vector.at(9) <= 1.0) {
								if (feature_vector.at(1) <= 436778.0) {
									return 1;
								}
								else {
									if (feature_vector.at(5) <= 0.11) {
										return 1;
									}
									else {
										return 0;
									}
								}
							}
							else {
								if (feature_vector.at(10) <= 1.0) {
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
						if (feature_vector.at(8) <= 548142.5) {
							if (feature_vector.at(8) <= 423334.0) {
								if (feature_vector.at(2) <= 18.74) {
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
						else {
							return 1;
						}
					}
				}
			}
		}
	}
	else {
		if (feature_vector.at(3) <= 28.51) {
			if (feature_vector.at(7) <= 3273.0) {
				if (feature_vector.at(12) <= 23133.0) {
					if (feature_vector.at(12) <= 3782.5) {
						if (feature_vector.at(10) <= 0.0) {
							if (feature_vector.at(5) <= 0.25) {
								if (feature_vector.at(7) <= 2.5) {
									if (feature_vector.at(1) <= 992441.0) {
										return 2;
									}
									else {
										return 0;
									}
								}
								else {
									return 1;
								}
							}
							else {
								if (feature_vector.at(2) <= 5.8) {
									if (feature_vector.at(2) <= 4.45) {
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
						else {
							if (feature_vector.at(4) <= 45.5) {
								if (feature_vector.at(6) <= 0.99) {
									return 2;
								}
								else {
									return 1;
								}
							}
							else {
								if (feature_vector.at(7) <= 986.0) {
									if (feature_vector.at(1) <= 823797.5) {
										return 2;
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
						if (feature_vector.at(8) <= 15767.0) {
							if (feature_vector.at(11) <= 2.76) {
								if (feature_vector.at(0) <= 64149.0) {
									if (feature_vector.at(5) <= 0.34) {
										return 1;
									}
									else {
										return 0;
									}
								}
								else {
									return 1;
								}
							}
							else {
								if (feature_vector.at(10) <= 0.0) {
									if (feature_vector.at(1) <= 4538810.0) {
										return 0;
									}
									else {
										return 1;
									}
								}
								else {
									if (feature_vector.at(2) <= 67.89) {
										return 1;
									}
									else {
										return 1;
									}
								}
							}
						}
						else {
							if (feature_vector.at(11) <= 39.21) {
								if (feature_vector.at(4) <= 77.0) {
									if (feature_vector.at(6) <= 0.99) {
										return 0;
									}
									else {
										return 1;
									}
								}
								else {
									return 1;
								}
							}
							else {
								if (feature_vector.at(1) <= 343073.5) {
									return 1;
								}
								else {
									if (feature_vector.at(7) <= 365.0) {
										return 2;
									}
									else {
										return 2;
									}
								}
							}
						}
					}
				}
				else {
					if (feature_vector.at(11) <= 40.78) {
						if (feature_vector.at(9) <= 0.02) {
							if (feature_vector.at(3) <= 11.75) {
								return 1;
							}
							else {
								if (feature_vector.at(5) <= 0.22) {
									return 1;
								}
								else {
									if (feature_vector.at(3) <= 19.05) {
										return 2;
									}
									else {
										return 2;
									}
								}
							}
						}
						else {
							if (feature_vector.at(5) <= 0.04) {
								if (feature_vector.at(9) <= 0.03) {
									return 0;
								}
								else {
									return 0;
								}
							}
							else {
								return 1;
							}
						}
					}
					else {
						if (feature_vector.at(11) <= 43.26) {
							if (feature_vector.at(4) <= 182.0) {
								return 1;
							}
							else {
								return 2;
							}
						}
						else {
							if (feature_vector.at(1) <= 97212.5) {
								return 1;
							}
							else {
								if (feature_vector.at(11) <= 54.18) {
									if (feature_vector.at(0) <= 126266.5) {
										return 2;
									}
									else {
										return 2;
									}
								}
								else {
									if (feature_vector.at(1) <= 3472860.0) {
										return 2;
									}
									else {
										return 2;
									}
								}
							}
						}
					}
				}
			}
			else {
				if (feature_vector.at(11) <= 103.16) {
					if (feature_vector.at(2) <= 3.84) {
						if (feature_vector.at(6) <= 0.88) {
							return 1;
						}
						else {
							if (feature_vector.at(9) <= 0.08) {
								return 1;
							}
							else {
								if (feature_vector.at(11) <= 2.32) {
									return 1;
								}
								else {
									if (feature_vector.at(9) <= 0.46) {
										return 0;
									}
									else {
										return 0;
									}
								}
							}
						}
					}
					else {
						if (feature_vector.at(2) <= 14.91) {
							if (feature_vector.at(0) <= 426543.5) {
								if (feature_vector.at(4) <= 60.0) {
									if (feature_vector.at(12) <= 203496.0) {
										return 1;
									}
									else {
										return 0;
									}
								}
								else {
									if (feature_vector.at(3) <= 5.16) {
										return 1;
									}
									else {
										return 1;
									}
								}
							}
							else {
								if (feature_vector.at(5) <= 0.58) {
									if (feature_vector.at(9) <= 0.12) {
										return 2;
									}
									else {
										return 0;
									}
								}
								else {
									if (feature_vector.at(11) <= 36.1) {
										return 1;
									}
									else {
										return 2;
									}
								}
							}
						}
						else {
							if (feature_vector.at(0) <= 2526080.0) {
								if (feature_vector.at(0) <= 9523.0) {
									if (feature_vector.at(10) <= 1.0) {
										return 1;
									}
									else {
										return 0;
									}
								}
								else {
									if (feature_vector.at(7) <= 3439.0) {
										return 2;
									}
									else {
										return 1;
									}
								}
							}
							else {
								if (feature_vector.at(9) <= 0.53) {
									return 0;
								}
								else {
									return 1;
								}
							}
						}
					}
				}
				else {
					return 2;
				}
			}
		}
		else {
			if (feature_vector.at(3) <= 168.78) {
				if (feature_vector.at(7) <= 13536.5) {
					if (feature_vector.at(11) <= 31.72) {
						if (feature_vector.at(11) <= 4.88) {
							if (feature_vector.at(7) <= 490.0) {
								if (feature_vector.at(5) <= 0.45) {
									if (feature_vector.at(4) <= 99.5) {
										return 1;
									}
									else {
										return 0;
									}
								}
								else {
									if (feature_vector.at(1) <= 514292.0) {
										return 0;
									}
									else {
										return 2;
									}
								}
							}
							else {
								if (feature_vector.at(3) <= 51.4) {
									if (feature_vector.at(4) <= 3851.0) {
										return 0;
									}
									else {
										return 1;
									}
								}
								else {
									if (feature_vector.at(0) <= 1582589.5) {
										return 0;
									}
									else {
										return 1;
									}
								}
							}
						}
						else {
							if (feature_vector.at(4) <= 4886.0) {
								if (feature_vector.at(12) <= 122.5) {
									if (feature_vector.at(5) <= 0.38) {
										return 0;
									}
									else {
										return 2;
									}
								}
								else {
									if (feature_vector.at(2) <= 88.95) {
										return 1;
									}
									else {
										return 0;
									}
								}
							}
							else {
								if (feature_vector.at(3) <= 105.52) {
									if (feature_vector.at(11) <= 5.91) {
										return 1;
									}
									else {
										return 2;
									}
								}
								else {
									if (feature_vector.at(4) <= 8777.5) {
										return 0;
									}
									else {
										return 1;
									}
								}
							}
						}
					}
					else {
						if (feature_vector.at(8) <= 15481.5) {
							if (feature_vector.at(12) <= 3210.5) {
								if (feature_vector.at(0) <= 26297.0) {
									return 1;
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
								if (feature_vector.at(3) <= 61.85) {
									return 1;
								}
								else {
									if (feature_vector.at(3) <= 64.4) {
										return 2;
									}
									else {
										return 1;
									}
								}
							}
						}
						else {
							if (feature_vector.at(11) <= 72.24) {
								if (feature_vector.at(9) <= 0.25) {
									if (feature_vector.at(7) <= 394.5) {
										return 1;
									}
									else {
										return 2;
									}
								}
								else {
									if (feature_vector.at(3) <= 54.8) {
										return 1;
									}
									else {
										return 2;
									}
								}
							}
							else {
								if (feature_vector.at(10) <= 1.0) {
									if (feature_vector.at(7) <= 24.5) {
										return 1;
									}
									else {
										return 2;
									}
								}
								else {
									if (feature_vector.at(5) <= 0.36) {
										return 2;
									}
									else {
										return 1;
									}
								}
							}
						}
					}
				}
				else {
					if (feature_vector.at(6) <= 0.91) {
						if (feature_vector.at(1) <= 1357750.0) {
							if (feature_vector.at(8) <= 322531.5) {
								if (feature_vector.at(10) <= 0.04) {
									return 0;
								}
								else {
									if (feature_vector.at(6) <= 0.84) {
										return 2;
									}
									else {
										return 1;
									}
								}
							}
							else {
								if (feature_vector.at(3) <= 42.05) {
									if (feature_vector.at(6) <= 0.84) {
										return 2;
									}
									else {
										return 1;
									}
								}
								else {
									if (feature_vector.at(11) <= 3.96) {
										return 1;
									}
									else {
										return 2;
									}
								}
							}
						}
						else {
							if (feature_vector.at(5) <= 0.58) {
								if (feature_vector.at(10) <= 0.25) {
									return 1;
								}
								else {
									return 2;
								}
							}
							else {
								if (feature_vector.at(0) <= 3943285.0) {
									if (feature_vector.at(11) <= 69.42) {
										return 1;
									}
									else {
										return 2;
									}
								}
								else {
									if (feature_vector.at(11) <= 7.48) {
										return 2;
									}
									else {
										return 1;
									}
								}
							}
						}
					}
					else {
						if (feature_vector.at(7) <= 17013.0) {
							if (feature_vector.at(3) <= 70.09) {
								if (feature_vector.at(1) <= 1695060.0) {
									return 1;
								}
								else {
									if (feature_vector.at(5) <= 0.39) {
										return 1;
									}
									else {
										return 2;
									}
								}
							}
							else {
								if (feature_vector.at(4) <= 2406.0) {
									if (feature_vector.at(1) <= 700752.0) {
										return 1;
									}
									else {
										return 2;
									}
								}
								else {
									if (feature_vector.at(2) <= 15.29) {
										return 0;
									}
									else {
										return 1;
									}
								}
							}
						}
						else {
							if (feature_vector.at(5) <= 0.48) {
								if (feature_vector.at(10) <= 0.07) {
									return 0;
								}
								else {
									if (feature_vector.at(6) <= 0.92) {
										return 1;
									}
									else {
										return 1;
									}
								}
							}
							else {
								if (feature_vector.at(5) <= 0.53) {
									if (feature_vector.at(5) <= 0.51) {
										return 1;
									}
									else {
										return 2;
									}
								}
								else {
									if (feature_vector.at(7) <= 30810.0) {
										return 1;
									}
									else {
										return 1;
									}
								}
							}
						}
					}
				}
			}
			else {
				if (feature_vector.at(11) <= 3.01) {
					if (feature_vector.at(10) <= 0.7) {
						if (feature_vector.at(3) <= 181.91) {
							if (feature_vector.at(9) <= 0.02) {
								return 2;
							}
							else {
								return 0;
							}
						}
						else {
							if (feature_vector.at(0) <= 761018.5) {
								if (feature_vector.at(8) <= 112867.5) {
									if (feature_vector.at(11) <= 2.97) {
										return 0;
									}
									else {
										return 1;
									}
								}
								else {
									return 1;
								}
							}
							else {
								if (feature_vector.at(9) <= 0.28) {
									return 1;
								}
								else {
									return 0;
								}
							}
						}
					}
					else {
						if (feature_vector.at(1) <= 582411.0) {
							return 0;
						}
						else {
							return 2;
						}
					}
				}
				else {
					if (feature_vector.at(8) <= 19508850.0) {
						if (feature_vector.at(8) <= 20279.5) {
							if (feature_vector.at(8) <= 786.0) {
								return 2;
							}
							else {
								if (feature_vector.at(10) <= 0.0) {
									if (feature_vector.at(10) <= 0.0) {
										return 1;
									}
									else {
										return 2;
									}
								}
								else {
									return 1;
								}
							}
						}
						else {
							if (feature_vector.at(12) <= 6859520.0) {
								if (feature_vector.at(11) <= 4.31) {
									if (feature_vector.at(6) <= 0.87) {
										return 2;
									}
									else {
										return 1;
									}
								}
								else {
									if (feature_vector.at(10) <= 0.0) {
										return 1;
									}
									else {
										return 2;
									}
								}
							}
							else {
								if (feature_vector.at(3) <= 1031.38) {
									return 1;
								}
								else {
									return 2;
								}
							}
						}
					}
					else {
						return 1;
					}
				}
			}
		}
	}
}