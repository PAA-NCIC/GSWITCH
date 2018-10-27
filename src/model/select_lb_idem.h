
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
0 - TM
1 - WM
2 - CM
3 - STRICT


Simply include this file to your project and use it
*/

#include <vector>

inline int select_lb_idem(const std::vector<double> & feature_vector) 
{
	if (feature_vector.at(12) <= 28360.0) {
		if (feature_vector.at(11) <= 34.53) {
			if (feature_vector.at(0) <= 257894.5) {
				if (feature_vector.at(4) <= 25.5) {
					if (feature_vector.at(2) <= 5.36) {
						if (feature_vector.at(0) <= 79200.0) {
							if (feature_vector.at(8) <= 10.5) {
								return 1;
							}
							else {
								return 2;
							}
						}
						else {
							if (feature_vector.at(0) <= 95690.0) {
								if (feature_vector.at(10) <= 0.0) {
									if (feature_vector.at(9) <= 0.0) {
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
								return 0;
							}
						}
					}
					else {
						if (feature_vector.at(1) <= 546737.0) {
							if (feature_vector.at(0) <= 49833.0) {
								if (feature_vector.at(7) <= 277.5) {
									if (feature_vector.at(12) <= 296.0) {
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
								if (feature_vector.at(12) <= 45.5) {
									if (feature_vector.at(2) <= 5.99) {
										return 0;
									}
									else {
										return 1;
									}
								}
								else {
									if (feature_vector.at(11) <= 5.64) {
										return 0;
									}
									else {
										return 1;
									}
								}
							}
						}
						else {
							if (feature_vector.at(3) <= 0.32) {
								if (feature_vector.at(12) <= 90.0) {
									if (feature_vector.at(8) <= 793.5) {
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
								if (feature_vector.at(4) <= 20.5) {
									if (feature_vector.at(6) <= 1.0) {
										return 0;
									}
									else {
										return 0;
									}
								}
								else {
									if (feature_vector.at(12) <= 2932.5) {
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
					if (feature_vector.at(0) <= 54471.5) {
						if (feature_vector.at(10) <= 0.0) {
							if (feature_vector.at(2) <= 219.6) {
								if (feature_vector.at(11) <= 25.63) {
									if (feature_vector.at(6) <= 0.82) {
										return 2;
									}
									else {
										return 1;
									}
								}
								else {
									if (feature_vector.at(4) <= 30.5) {
										return 1;
									}
									else {
										return 2;
									}
								}
							}
							else {
								if (feature_vector.at(7) <= 1.0) {
									return 2;
								}
								else {
									if (feature_vector.at(4) <= 1996.0) {
										return 0;
									}
									else {
										return 1;
									}
								}
							}
						}
						else {
							if (feature_vector.at(12) <= 2729.5) {
								if (feature_vector.at(8) <= 19399.0) {
									if (feature_vector.at(3) <= 105.0) {
										return 2;
									}
									else {
										return 1;
									}
								}
								else {
									if (feature_vector.at(10) <= 0.03) {
										return 2;
									}
									else {
										return 1;
									}
								}
							}
							else {
								if (feature_vector.at(9) <= 0.43) {
									if (feature_vector.at(8) <= 3498.0) {
										return 1;
									}
									else {
										return 2;
									}
								}
								else {
									if (feature_vector.at(3) <= 98.42) {
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
						if (feature_vector.at(3) <= 170.84) {
							if (feature_vector.at(12) <= 4035.5) {
								if (feature_vector.at(12) <= 90.5) {
									if (feature_vector.at(9) <= 0.0) {
										return 1;
									}
									else {
										return 0;
									}
								}
								else {
									if (feature_vector.at(11) <= 33.53) {
										return 1;
									}
									else {
										return 0;
									}
								}
							}
							else {
								if (feature_vector.at(0) <= 157772.0) {
									if (feature_vector.at(7) <= 48823.0) {
										return 2;
									}
									else {
										return 1;
									}
								}
								else {
									if (feature_vector.at(10) <= 0.17) {
										return 0;
									}
									else {
										return 1;
									}
								}
							}
						}
						else {
							if (feature_vector.at(12) <= 1.5) {
								if (feature_vector.at(4) <= 18949.5) {
									if (feature_vector.at(5) <= 0.56) {
										return 2;
									}
									else {
										return 1;
									}
								}
								else {
									if (feature_vector.at(5) <= 0.75) {
										return 3;
									}
									else {
										return 2;
									}
								}
							}
							else {
								if (feature_vector.at(12) <= 1758.5) {
									return 0;
								}
								else {
									if (feature_vector.at(8) <= 154211.5) {
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
				if (feature_vector.at(9) <= 0.01) {
					if (feature_vector.at(12) <= 7388.5) {
						if (feature_vector.at(3) <= 280.33) {
							if (feature_vector.at(4) <= 91.5) {
								if (feature_vector.at(8) <= 21767.0) {
									if (feature_vector.at(9) <= 0.0) {
										return 0;
									}
									else {
										return 1;
									}
								}
								else {
									if (feature_vector.at(12) <= 995.0) {
										return 1;
									}
									else {
										return 0;
									}
								}
							}
							else {
								if (feature_vector.at(11) <= 21.18) {
									if (feature_vector.at(12) <= 69.0) {
										return 0;
									}
									else {
										return 0;
									}
								}
								else {
									if (feature_vector.at(0) <= 396563.5) {
										return 1;
									}
									else {
										return 0;
									}
								}
							}
						}
						else {
							if (feature_vector.at(6) <= 0.81) {
								if (feature_vector.at(0) <= 703505.0) {
									if (feature_vector.at(0) <= 314591.0) {
										return 0;
									}
									else {
										return 3;
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
					else {
						if (feature_vector.at(8) <= 39038.5) {
							return 3;
						}
						else {
							if (feature_vector.at(3) <= 36.44) {
								return 0;
							}
							else {
								return 2;
							}
						}
					}
				}
				else {
					if (feature_vector.at(11) <= 2.09) {
						if (feature_vector.at(0) <= 514937.0) {
							if (feature_vector.at(4) <= 15148.0) {
								if (feature_vector.at(12) <= 439.0) {
									return 0;
								}
								else {
									if (feature_vector.at(8) <= 6624.0) {
										return 0;
									}
									else {
										return 1;
									}
								}
							}
							else {
								if (feature_vector.at(11) <= 1.81) {
									return 0;
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
						if (feature_vector.at(12) <= 11570.5) {
							if (feature_vector.at(5) <= 0.87) {
								if (feature_vector.at(10) <= 0.0) {
									return 0;
								}
								else {
									if (feature_vector.at(3) <= 29.53) {
										return 1;
									}
									else {
										return 1;
									}
								}
							}
							else {
								if (feature_vector.at(12) <= 7502.5) {
									return 0;
								}
								else {
									return 2;
								}
							}
						}
						else {
							if (feature_vector.at(11) <= 5.52) {
								if (feature_vector.at(5) <= 0.62) {
									return 0;
								}
								else {
									return 1;
								}
							}
							else {
								if (feature_vector.at(2) <= 22.16) {
									return 2;
								}
								else {
									if (feature_vector.at(7) <= 71556.5) {
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
			}
		}
		else {
			if (feature_vector.at(0) <= 156333.5) {
				if (feature_vector.at(12) <= 927.0) {
					if (feature_vector.at(12) <= 247.0) {
						if (feature_vector.at(6) <= 0.96) {
							if (feature_vector.at(0) <= 133655.0) {
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
						if (feature_vector.at(0) <= 35048.0) {
							if (feature_vector.at(3) <= 88.0) {
								if (feature_vector.at(11) <= 38.02) {
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
						else {
							if (feature_vector.at(0) <= 132178.0) {
								if (feature_vector.at(2) <= 45.21) {
									if (feature_vector.at(11) <= 51.87) {
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
				}
				else {
					if (feature_vector.at(3) <= 412.84) {
						if (feature_vector.at(3) <= 5.02) {
							if (feature_vector.at(12) <= 7159.0) {
								if (feature_vector.at(10) <= 0.07) {
									if (feature_vector.at(3) <= 4.93) {
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
								if (feature_vector.at(2) <= 532.65) {
									return 0;
								}
								else {
									return 2;
								}
							}
						}
						else {
							if (feature_vector.at(12) <= 24178.5) {
								if (feature_vector.at(9) <= 0.0) {
									if (feature_vector.at(9) <= 0.0) {
										return 2;
									}
									else {
										return 3;
									}
								}
								else {
									if (feature_vector.at(3) <= 143.06) {
										return 2;
									}
									else {
										return 2;
									}
								}
							}
							else {
								if (feature_vector.at(3) <= 39.64) {
									if (feature_vector.at(4) <= 94.5) {
										return 0;
									}
									else {
										return 3;
									}
								}
								else {
									return 2;
								}
							}
						}
					}
					else {
						if (feature_vector.at(12) <= 19721.0) {
							if (feature_vector.at(9) <= 0.0) {
								return 2;
							}
							else {
								return 1;
							}
						}
						else {
							return 3;
						}
					}
				}
			}
			else {
				if (feature_vector.at(12) <= 689.0) {
					if (feature_vector.at(8) <= 138.0) {
						if (feature_vector.at(1) <= 1140112.0) {
							return 0;
						}
						else {
							return 2;
						}
					}
					else {
						if (feature_vector.at(7) <= 1.5) {
							return 2;
						}
						else {
							return 1;
						}
					}
				}
				else {
					if (feature_vector.at(4) <= 509.0) {
						if (feature_vector.at(0) <= 393443.0) {
							if (feature_vector.at(5) <= 0.09) {
								if (feature_vector.at(11) <= 57.37) {
									if (feature_vector.at(7) <= 379.0) {
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
								if (feature_vector.at(1) <= 762720.0) {
									if (feature_vector.at(4) <= 89.0) {
										return 0;
									}
									else {
										return 2;
									}
								}
								else {
									if (feature_vector.at(11) <= 38.19) {
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
					else {
						if (feature_vector.at(10) <= 0.01) {
							return 2;
						}
						else {
							if (feature_vector.at(10) <= 0.02) {
								if (feature_vector.at(4) <= 2908.0) {
									return 2;
								}
								else {
									return 3;
								}
							}
							else {
								if (feature_vector.at(9) <= 0.01) {
									if (feature_vector.at(11) <= 1291.36) {
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
		}
	}
	else {
		if (feature_vector.at(12) <= 35868.0) {
			if (feature_vector.at(8) <= 54151.5) {
				if (feature_vector.at(7) <= 54.5) {
					return 2;
				}
				else {
					return 3;
				}
			}
			else {
				if (feature_vector.at(7) <= 36251.5) {
					return 2;
				}
				else {
					return 1;
				}
			}
		}
		else {
			if (feature_vector.at(12) <= 51192.0) {
				if (feature_vector.at(9) <= 0.08) {
					if (feature_vector.at(2) <= 1.95) {
						return 2;
					}
					else {
						if (feature_vector.at(6) <= 0.96) {
							if (feature_vector.at(8) <= 438046.5) {
								return 3;
							}
							else {
								if (feature_vector.at(9) <= 0.03) {
									return 3;
								}
								else {
									return 2;
								}
							}
						}
						else {
							if (feature_vector.at(9) <= 0.07) {
								if (feature_vector.at(11) <= 102.82) {
									return 0;
								}
								else {
									return 2;
								}
							}
							else {
								return 3;
							}
						}
					}
				}
				else {
					if (feature_vector.at(1) <= 2025665.0) {
						if (feature_vector.at(8) <= 798484.5) {
							return 0;
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
			else {
				return 3;
			}
		}
	}
}