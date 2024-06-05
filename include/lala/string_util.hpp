//
// Created by falque on 05/06/24.
//

#ifndef STRING_UTIL_H
#define STRING_UTIL_H
#include <string>
#include <vector>
#include <functional>
using namespace std;


template<typename T>
std::string join(const battery::vector<T>& vec, const std::string& separator, std::function<std::string(T)> toString) {
  std::string result;
  for (size_t i = 0; i < vec.size(); ++i) {
    result += toString(vec[i]);
    if (i < vec.size() - 1) {
      result += separator;
    }
  }
  return result;
}


#endif //STRING_UTIL_H
