#ifndef __FILESYSTEM_H
#define __FILESYSTEM_H
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <sys/stat.h>
#include <sys/time.h>
#include <type_traits>

inline off_t fsize(const char *filename) {
  struct stat st;
  if (stat(filename, &st) == 0)
    return st.st_size;
  return -1;
}

inline bool fexist(const char *filename) {
  struct stat st;
  return (stat(filename, &st) == 0);
}

inline const char *get_fname(const char *path) {
  std::string fpath(path);
  const size_t idx = fpath.find_last_of("/");
  return fpath.substr(idx + 1).c_str();
}

inline const char *get_dirname(const char *path) {
  std::string fpath(path);
  const size_t idx = fpath.find_last_of("/");
  return fpath.substr(0, idx).c_str();
}

inline const char *concat(const char *dir, const char *file) {
  std::string a(dir);
  std::string b(file);
  return (a + "/" + b).c_str();
}

#endif
