
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
feature_vector[7] - act(v)
feature_vector[8] - act(e)
feature_vector[9] - r_act(v)
feature_vector[10] - r_act(e)
feature_vector[11] - cur_act_avg(d)
feature_vector[12] - cur_act_r(d)


It returns index of predicted class:
0 - WM
1 - CM
2 - STRICT


Simply include this file to your project and use it
*/

#include <vector>

inline int select_lb_asso_forward(const std::vector<double> & feature_vector) 
{
	if (feature_vector.at(11) <= 25.72) {
		if (feature_vector.at(8) <= 4102.5) {
			if (feature_vector.at(10) <= 0.0) {
				if (feature_vector.at(11) <= 12.43) {
					if (feature_vector.at(1) <= 3564390.0) {
						if (feature_vector.at(4) <= 1552.5) {
							if (feature_vector.at(4) <= 1203.0) {
								if (feature_vector.at(0) <= 47430.0) {
									if (feature_vector.at(0) <= 37395.5) {
										return 0;
									}
									else {
										return 2;
									}
								}
								else {
									if (feature_vector.at(1) <= 386280.0) {
										return 2;
									}
									else {
										return 0;
									}
								}
							}
							else {
								if (feature_vector.at(4) <= 1384.0) {
									return 1;
								}
								else {
									return 2;
								}
							}
						}
						else {
							if (feature_vector.at(12) <= 48.5) {
								if (feature_vector.at(3) <= 447.68) {
									if (feature_vector.at(10) <= 0.0) {
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
								if (feature_vector.at(6) <= 0.88) {
									if (feature_vector.at(7) <= 291.5) {
										return 0;
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
					}
					else {
						if (feature_vector.at(5) <= 0.67) {
							return 1;
						}
						else {
							if (feature_vector.at(10) <= 0.0) {
								return 0;
							}
							else {
								if (feature_vector.at(10) <= 0.0) {
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
					if (feature_vector.at(0) <= 21058.5) {
						return 0;
					}
					else {
						if (feature_vector.at(10) <= 0.0) {
							if (feature_vector.at(5) <= 0.08) {
								return 1;
							}
							else {
								if (feature_vector.at(7) <= 88.0) {
									return 2;
								}
								else {
									return 1;
								}
							}
						}
						else {
							if (feature_vector.at(2) <= 6.06) {
								if (feature_vector.at(8) <= 110.5) {
									return 0;
								}
								else {
									return 2;
								}
							}
							else {
								if (feature_vector.at(4) <= 33.5) {
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
				if (feature_vector.at(7) <= 366.0) {
					if (feature_vector.at(1) <= 108651.0) {
						if (feature_vector.at(7) <= 22.5) {
							return 2;
						}
						else {
							return 1;
						}
					}
					else {
						if (feature_vector.at(4) <= 12.5) {
							if (feature_vector.at(12) <= 26.0) {
								return 2;
							}
							else {
								if (feature_vector.at(1) <= 270769.0) {
									if (feature_vector.at(9) <= 0.0) {
										return 0;
									}
									else {
										return 2;
									}
								}
								else {
									if (feature_vector.at(6) <= 1.0) {
										return 1;
									}
									else {
										return 1;
									}
								}
							}
						}
						else {
							if (feature_vector.at(1) <= 648200.5) {
								if (feature_vector.at(12) <= 38.5) {
									return 1;
								}
								else {
									if (feature_vector.at(6) <= 0.89) {
										return 1;
									}
									else {
										return 2;
									}
								}
							}
							else {
								if (feature_vector.at(9) <= 0.0) {
									if (feature_vector.at(4) <= 1385.0) {
										return 1;
									}
									else {
										return 2;
									}
								}
								else {
									if (feature_vector.at(5) <= 0.13) {
										return 1;
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
					if (feature_vector.at(5) <= 0.19) {
						if (feature_vector.at(7) <= 421.5) {
							if (feature_vector.at(1) <= 784008.0) {
								if (feature_vector.at(12) <= 200.0) {
									return 1;
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
							return 2;
						}
					}
					else {
						if (feature_vector.at(10) <= 0.0) {
							return 1;
						}
						else {
							if (feature_vector.at(12) <= 138.5) {
								return 0;
							}
							else {
								if (feature_vector.at(1) <= 1112670.0) {
									if (feature_vector.at(10) <= 0.0) {
										return 2;
									}
									else {
										return 1;
									}
								}
								else {
									if (feature_vector.at(8) <= 1956.5) {
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
			}
		}
		else {
			if (feature_vector.at(5) <= 0.3) {
				if (feature_vector.at(7) <= 312.5) {
					if (feature_vector.at(11) <= 19.74) {
						if (feature_vector.at(1) <= 233037.0) {
							return 0;
						}
						else {
							if (feature_vector.at(6) <= 1.0) {
								return 1;
							}
							else {
								if (feature_vector.at(11) <= 16.51) {
									return 0;
								}
								else {
									if (feature_vector.at(11) <= 19.67) {
										return 1;
									}
									else {
										return 1;
									}
								}
							}
						}
					}
					else {
						return 1;
					}
				}
				else {
					if (feature_vector.at(5) <= 0.1) {
						if (feature_vector.at(0) <= 18234.0) {
							if (feature_vector.at(12) <= 706.5) {
								return 0;
							}
							else {
								return 1;
							}
						}
						else {
							if (feature_vector.at(11) <= 5.79) {
								if (feature_vector.at(6) <= 1.0) {
									if (feature_vector.at(10) <= 0.01) {
										return 0;
									}
									else {
										return 0;
									}
								}
								else {
									if (feature_vector.at(11) <= 5.74) {
										return 1;
									}
									else {
										return 0;
									}
								}
							}
							else {
								if (feature_vector.at(3) <= 29.08) {
									if (feature_vector.at(7) <= 351.5) {
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
					}
					else {
						if (feature_vector.at(6) <= 1.0) {
							if (feature_vector.at(1) <= 787209.0) {
								if (feature_vector.at(6) <= 1.0) {
									if (feature_vector.at(8) <= 4686.0) {
										return 0;
									}
									else {
										return 0;
									}
								}
								else {
									if (feature_vector.at(5) <= 0.12) {
										return 0;
									}
									else {
										return 1;
									}
								}
							}
							else {
								if (feature_vector.at(2) <= 14.38) {
									if (feature_vector.at(12) <= 9273.5) {
										return 0;
									}
									else {
										return 1;
									}
								}
								else {
									if (feature_vector.at(12) <= 6638.5) {
										return 1;
									}
									else {
										return 0;
									}
								}
							}
						}
						else {
							if (feature_vector.at(2) <= 15.25) {
								if (feature_vector.at(11) <= 14.64) {
									if (feature_vector.at(9) <= 0.01) {
										return 1;
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
								if (feature_vector.at(9) <= 0.02) {
									return 0;
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
				if (feature_vector.at(11) <= 2.76) {
					if (feature_vector.at(7) <= 2682.0) {
						if (feature_vector.at(3) <= 4.62) {
							return 0;
						}
						else {
							return 1;
						}
					}
					else {
						if (feature_vector.at(3) <= 4.84) {
							return 1;
						}
						else {
							if (feature_vector.at(4) <= 48.5) {
								return 1;
							}
							else {
								if (feature_vector.at(12) <= 2538.5) {
									if (feature_vector.at(11) <= 2.11) {
										return 0;
									}
									else {
										return 0;
									}
								}
								else {
									if (feature_vector.at(7) <= 114003.0) {
										return 1;
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
					if (feature_vector.at(12) <= 57484.5) {
						if (feature_vector.at(12) <= 2901.5) {
							if (feature_vector.at(3) <= 66.27) {
								if (feature_vector.at(3) <= 6.0) {
									if (feature_vector.at(2) <= 2.99) {
										return 1;
									}
									else {
										return 0;
									}
								}
								else {
									if (feature_vector.at(5) <= 0.56) {
										return 1;
									}
									else {
										return 0;
									}
								}
							}
							else {
								if (feature_vector.at(7) <= 455.0) {
									return 1;
								}
								else {
									if (feature_vector.at(6) <= 0.91) {
										return 0;
									}
									else {
										return 0;
									}
								}
							}
						}
						else {
							if (feature_vector.at(8) <= 1243425.0) {
								if (feature_vector.at(12) <= 28678.5) {
									if (feature_vector.at(6) <= 0.85) {
										return 1;
									}
									else {
										return 1;
									}
								}
								else {
									if (feature_vector.at(8) <= 385749.5) {
										return 2;
									}
									else {
										return 1;
									}
								}
							}
							else {
								if (feature_vector.at(9) <= 0.42) {
									return 1;
								}
								else {
									return 0;
								}
							}
						}
					}
					else {
						if (feature_vector.at(3) <= 71.63) {
							if (feature_vector.at(4) <= 18305.5) {
								return 2;
							}
							else {
								return 1;
							}
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
		if (feature_vector.at(8) <= 19486.5) {
			if (feature_vector.at(8) <= 4043.5) {
				if (feature_vector.at(11) <= 51.57) {
					if (feature_vector.at(3) <= 6.99) {
						if (feature_vector.at(2) <= 40.53) {
							if (feature_vector.at(9) <= 0.0) {
								if (feature_vector.at(2) <= 40.53) {
									if (feature_vector.at(3) <= 1.58) {
										return 2;
									}
									else {
										return 2;
									}
								}
								else {
									if (feature_vector.at(12) <= 962.0) {
										return 2;
									}
									else {
										return 2;
									}
								}
							}
							else {
								if (feature_vector.at(12) <= 3385.0) {
									return 2;
								}
								else {
									if (feature_vector.at(8) <= 3860.0) {
										return 1;
									}
									else {
										return 2;
									}
								}
							}
						}
						else {
							if (feature_vector.at(1) <= 192827.5) {
								if (feature_vector.at(10) <= 0.0) {
									return 2;
								}
								else {
									return 1;
								}
							}
							else {
								if (feature_vector.at(9) <= 0.0) {
									return 0;
								}
								else {
									if (feature_vector.at(4) <= 17.0) {
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
						if (feature_vector.at(6) <= 0.98) {
							if (feature_vector.at(8) <= 2119.5) {
								if (feature_vector.at(8) <= 188.5) {
									return 0;
								}
								else {
									if (feature_vector.at(11) <= 27.49) {
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
							if (feature_vector.at(9) <= 0.0) {
								if (feature_vector.at(4) <= 137.0) {
									return 0;
								}
								else {
									return 1;
								}
							}
							else {
								if (feature_vector.at(1) <= 104238.0) {
									if (feature_vector.at(9) <= 0.02) {
										return 1;
									}
									else {
										return 2;
									}
								}
								else {
									if (feature_vector.at(3) <= 7.1) {
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
					if (feature_vector.at(5) <= 0.11) {
						if (feature_vector.at(3) <= 5.43) {
							if (feature_vector.at(11) <= 62.0) {
								return 1;
							}
							else {
								if (feature_vector.at(12) <= 2362.5) {
									if (feature_vector.at(12) <= 2026.5) {
										return 2;
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
							if (feature_vector.at(3) <= 42.01) {
								if (feature_vector.at(3) <= 5.92) {
									if (feature_vector.at(11) <= 65.47) {
										return 2;
									}
									else {
										return 1;
									}
								}
								else {
									if (feature_vector.at(11) <= 76.7) {
										return 1;
									}
									else {
										return 1;
									}
								}
							}
							else {
								return 2;
							}
						}
					}
					else {
						if (feature_vector.at(8) <= 156.5) {
							return 0;
						}
						else {
							if (feature_vector.at(3) <= 62.87) {
								if (feature_vector.at(12) <= 1373.0) {
									if (feature_vector.at(3) <= 25.56) {
										return 1;
									}
									else {
										return 2;
									}
								}
								else {
									if (feature_vector.at(2) <= 6.3) {
										return 2;
									}
									else {
										return 2;
									}
								}
							}
							else {
								if (feature_vector.at(9) <= 0.0) {
									return 2;
								}
								else {
									if (feature_vector.at(8) <= 274.0) {
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
			}
			else {
				if (feature_vector.at(7) <= 341.0) {
					if (feature_vector.at(8) <= 13629.5) {
						if (feature_vector.at(9) <= 0.07) {
							if (feature_vector.at(12) <= 9240.5) {
								if (feature_vector.at(8) <= 4084.5) {
									if (feature_vector.at(0) <= 7034.0) {
										return 2;
									}
									else {
										return 1;
									}
								}
								else {
									if (feature_vector.at(7) <= 194.5) {
										return 1;
									}
									else {
										return 1;
									}
								}
							}
							else {
								if (feature_vector.at(9) <= 0.03) {
									if (feature_vector.at(8) <= 10029.5) {
										return 1;
									}
									else {
										return 1;
									}
								}
								else {
									if (feature_vector.at(11) <= 60.17) {
										return 2;
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
					else {
						if (feature_vector.at(4) <= 96.5) {
							if (feature_vector.at(6) <= 1.0) {
								if (feature_vector.at(9) <= 0.01) {
									if (feature_vector.at(11) <= 88.23) {
										return 1;
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
								if (feature_vector.at(12) <= 9026.0) {
									return 1;
								}
								else {
									if (feature_vector.at(12) <= 12736.0) {
										return 2;
									}
									else {
										return 1;
									}
								}
							}
						}
						else {
							if (feature_vector.at(7) <= 288.0) {
								if (feature_vector.at(4) <= 117.0) {
									if (feature_vector.at(1) <= 333550.0) {
										return 1;
									}
									else {
										return 1;
									}
								}
								else {
									if (feature_vector.at(7) <= 106.5) {
										return 1;
									}
									else {
										return 1;
									}
								}
							}
							else {
								if (feature_vector.at(9) <= 0.01) {
									return 1;
								}
								else {
									return 2;
								}
							}
						}
					}
				}
				else {
					if (feature_vector.at(8) <= 12281.0) {
						if (feature_vector.at(10) <= 0.06) {
							return 1;
						}
						else {
							return 0;
						}
					}
					else {
						if (feature_vector.at(4) <= 64.0) {
							if (feature_vector.at(7) <= 412.5) {
								if (feature_vector.at(0) <= 394127.0) {
									if (feature_vector.at(7) <= 360.5) {
										return 1;
									}
									else {
										return 0;
									}
								}
								else {
									if (feature_vector.at(11) <= 33.77) {
										return 1;
									}
									else {
										return 2;
									}
								}
							}
							else {
								return 0;
							}
						}
						else {
							if (feature_vector.at(0) <= 35062.0) {
								if (feature_vector.at(8) <= 15502.0) {
									return 0;
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
				}
			}
		}
		else {
			if (feature_vector.at(11) <= 55.33) {
				if (feature_vector.at(6) <= 1.0) {
					if (feature_vector.at(11) <= 36.14) {
						if (feature_vector.at(12) <= 26901.5) {
							if (feature_vector.at(2) <= 32.42) {
								if (feature_vector.at(7) <= 2718.5) {
									if (feature_vector.at(9) <= 0.03) {
										return 1;
									}
									else {
										return 2;
									}
								}
								else {
									if (feature_vector.at(3) <= 62.94) {
										return 1;
									}
									else {
										return 1;
									}
								}
							}
							else {
								if (feature_vector.at(11) <= 27.79) {
									return 1;
								}
								else {
									if (feature_vector.at(12) <= 22232.0) {
										return 0;
									}
									else {
										return 1;
									}
								}
							}
						}
						else {
							if (feature_vector.at(1) <= 10286780.0) {
								if (feature_vector.at(6) <= 0.86) {
									if (feature_vector.at(7) <= 32476.0) {
										return 2;
									}
									else {
										return 1;
									}
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
						if (feature_vector.at(8) <= 1567160.0) {
							if (feature_vector.at(3) <= 441.06) {
								if (feature_vector.at(6) <= 0.99) {
									if (feature_vector.at(7) <= 5053.5) {
										return 2;
									}
									else {
										return 2;
									}
								}
								else {
									if (feature_vector.at(7) <= 1480.5) {
										return 2;
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
							if (feature_vector.at(2) <= 14.27) {
								if (feature_vector.at(2) <= 8.82) {
									return 1;
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
				else {
					if (feature_vector.at(2) <= 52.1) {
						if (feature_vector.at(0) <= 55788.0) {
							return 2;
						}
						else {
							if (feature_vector.at(7) <= 562.0) {
								return 2;
							}
							else {
								if (feature_vector.at(7) <= 655.5) {
									if (feature_vector.at(9) <= 0.01) {
										return 0;
									}
									else {
										return 0;
									}
								}
								else {
									return 0;
								}
							}
						}
					}
					else {
						if (feature_vector.at(2) <= 52.67) {
							return 2;
						}
						else {
							if (feature_vector.at(7) <= 2029.5) {
								if (feature_vector.at(0) <= 213219.5) {
									if (feature_vector.at(5) <= 0.13) {
										return 0;
									}
									else {
										return 2;
									}
								}
								else {
									if (feature_vector.at(12) <= 6094.0) {
										return 2;
									}
									else {
										return 0;
									}
								}
							}
							else {
								if (feature_vector.at(9) <= 0.03) {
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
				if (feature_vector.at(8) <= 24328.0) {
					if (feature_vector.at(7) <= 158.5) {
						if (feature_vector.at(9) <= 0.0) {
							return 1;
						}
						else {
							if (feature_vector.at(11) <= 150.75) {
								return 1;
							}
							else {
								return 2;
							}
						}
					}
					else {
						if (feature_vector.at(0) <= 4462.0) {
							return 1;
						}
						else {
							if (feature_vector.at(7) <= 363.0) {
								if (feature_vector.at(4) <= 315.0) {
									return 2;
								}
								else {
									if (feature_vector.at(8) <= 21452.0) {
										return 2;
									}
									else {
										return 2;
									}
								}
							}
							else {
								if (feature_vector.at(12) <= 14016.0) {
									return 1;
								}
								else {
									return 2;
								}
							}
						}
					}
				}
				else {
					if (feature_vector.at(4) <= 377096.5) {
						if (feature_vector.at(1) <= 160444.0) {
							if (feature_vector.at(10) <= 0.38) {
								return 2;
							}
							else {
								return 1;
							}
						}
						else {
							if (feature_vector.at(8) <= 31607.5) {
								if (feature_vector.at(4) <= 2546.0) {
									if (feature_vector.at(12) <= 1554.5) {
										return 2;
									}
									else {
										return 2;
									}
								}
								else {
									if (feature_vector.at(10) <= 0.04) {
										return 1;
									}
									else {
										return 2;
									}
								}
							}
							else {
								if (feature_vector.at(11) <= 70.94) {
									if (feature_vector.at(12) <= 27481.5) {
										return 2;
									}
									else {
										return 2;
									}
								}
								else {
									return 2;
								}
							}
						}
					}
					else {
						if (feature_vector.at(10) <= 0.39) {
							return 2;
						}
						else {
							return 0;
						}
					}
				}
			}
		}
	}
}