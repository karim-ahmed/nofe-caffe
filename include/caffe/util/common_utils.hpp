#ifndef CAFFE_COMMON_UTILS_HPP_
#define CAFFE_COMMON_UTILS_HPP_



#include "caffe/common.hpp"



namespace caffe {
	inline string convertFloatToString(float number) {
		stringstream ss;
		ss << std::fixed << number;
		return ss.str();
	}
	inline string convertIntToString(int number) {
		stringstream ss;
		ss << number;
		return ss.str();
	}
}

#endif  //


