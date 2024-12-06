#ifndef __JSON_CUH
#define __JSON_CUH

#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

template <typename T> std::string xtoa(T t) {
  std::stringstream ss;
  ss << t;
  return ss.str();
}

void output_key_value(std::ostream &out, std::string key, std::string value) {
  out << "\"" << key << "\": " << value;
}

struct json_t {
  int ttl = 0;
  std::vector<std::string> keys;
  std::vector<std::string> values;

  template <typename T> void add(std::string key, T value) {
    keys.push_back(key);
    values.push_back(xtoa(value));
  }

  void clean() {
    keys.clear();
    values.clear();
  }

  void dump(std::ostream &out) {
    ttl++;
    if (ttl > 100)
      return; // sample to train
    out << "{\n";
    for (size_t i = 0; i < keys.size(); ++i) {
      if (i > 0)
        out << ",\n";
      output_key_value(out, keys[i], values[i]);
    }
    out << "\n}\n";
  }
};

#endif
