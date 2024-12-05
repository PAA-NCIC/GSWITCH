#ifndef __CMDLINE_H
#define __CMDLINE_H

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda.h>
#include <getopt.h>
#include <iostream>
#include <sstream>
#include <string>
#include <sys/stat.h>

#include "utils/common.cuh"

#define no_arg 0
#define re_arg 1
#define op_arg 2
#define LEFT_SPAN 26
#define RIGHT_SPAN 34

bool file_exist(std::string path) {
  struct stat buf;
  return (stat(path.c_str(), &buf) == 0);
}

struct ins_t {
  Direction d;
  ASFmt asf;
  LB lb;
  Fusion fusion;
  bool has_direction = true;
  bool has_asfmt = true;
  bool has_lb = true;
  bool has_fusion = true;
};

struct cmd_t {
  void output() {
    std::cout << " src: " << src << std::endl;
    std::cout << " verbose: " << (verbose ? "True" : "False") << std::endl;
    std::cout << " validation: " << (validation ? "True" : "False")
              << std::endl;
    std::cout << " with-header: " << (with_header ? "True" : "False")
              << std::endl;
    std::cout << " with-weight: " << (with_weight ? "True" : "False")
              << std::endl;
    std::cout << " directed: " << (directed ? "True" : "False") << std::endl;
  }

  void prase_configs(std::string input) {
    std::vector<std::string> cfgs;
    std::istringstream iss(input);
    std::string tmp;
    while (std::getline(iss, tmp, '-')) {
      cfgs.push_back(tmp);
    }
    while (cfgs.size() < 4)
      cfgs.push_back("x");

    if (cfgs[0] == "Push")
      ins.d = Push;
    else if (cfgs[0] == "Pull")
      ins.d = Pull;
    else
      ins.has_direction = false;

    if (cfgs[1] == "Queue")
      ins.asf = Queue;
    else if (cfgs[1] == "Bitmap")
      ins.asf = Bitmap;
    else
      ins.has_asfmt = false;

    if (cfgs[2] == "TM")
      ins.lb = TM;
    else if (cfgs[2] == "WM")
      ins.lb = WM;
    else if (cfgs[2] == "CM")
      ins.lb = CM;
    else if (cfgs[2] == "STRICT")
      ins.lb = STRICT;
    else if (cfgs[2] == "TWOD")
      ins.lb = TWOD;
    else
      ins.has_lb = false;

    if (cfgs[3] == "Fused")
      ins.fusion = Fused;
    else if (cfgs[3] == "Standalone")
      ins.fusion = Standalone;
    else
      ins.has_fusion = false;
  }

  int src = -1;
  std::string configs = "";
  std::string path = "";
  std::string json = "";
  bool verbose = false;
  bool validation = false;
  bool with_header = false; // input file with header
  bool with_weight = false; // input file with weight
  bool directed = false;
  int device = 0;
  ins_t ins;
};

struct cmd_t cmd_opt;

void space(int size) {
  for (int i = 0; i < size; ++i)
    std::cout << " ";
}

void printf_desc(std::string desc, size_t left_span, size_t right_span) {
  size_t t = (desc.size() - 1) / right_span + 1;
  size_t idx = 0;
  for (size_t i = 0; i < t; ++i) {
    for (size_t j = 0; j < right_span && idx < desc.size(); ++idx, ++j) {
      std::cout << desc[idx];
    }
    puts("");
    if (i < t - 1)
      space(left_span);
  }
}

void print_option(std::string op_name, std::string op_desc,
                  size_t left_span = LEFT_SPAN,
                  size_t right_span = RIGHT_SPAN) {
  if (op_name.size() > left_span) {
    std::cout << op_name << std::endl;
    space(left_span);
  } else {
    std::cout << op_name;
    space(left_span - op_name.size());
  }
  printf_desc(op_desc, left_span, right_span);
}

static void help(std::string exec) {
  std::cout << "./" << exec << " <graph_path> [options]" << std::endl;
  print_option("[-r, --src=<int>]",
               "Choose a root vertex. (Default: choose randomly).");
  print_option("[-v, --verbose]",
               "Print verbose per iteration info. (Default: quiet mode)");
  print_option(
      "[-V, --validation]",
      "Process the CPU reference validation. (Defaule: no validation)");
  print_option("[-H, --with-header]", "Input file has header (e.g. nvertexs, "
                                      "nvertexs, nedges, Default: no header).");
  print_option("[-W, --with-weight]",
               "Input file has weight.(Default: no weight value)");
  print_option("[-i, --ignore-weight]",
               "Ignore the graph weight.(Default: false)");
  print_option("[-d, --directed]", "Graph is directed.(Default: undirected)");
  print_option("[-c, --configs=Push-Queue-CM-Fused]",
               "Set strategies, use - to separater them (Default: <empty>).");
  print_option("[-D, --device=<int>]", "Choose GPU for testing (Default: 0)");
  print_option("[-j, --json=<string>]", "set the json path (Default: 0)");
}

void parse_cmd(int argc, char **argv, std::string exec = "test") {
  if (argc < 2) {
    help(exec);
    exit(0);
  }
  cmd_opt.path = argv[1];
  if (!file_exist(cmd_opt.path)) {
    std::cout << " file \"" << cmd_opt.path << "\" does not exist."
              << std::endl;
    exit(0);
  }
  const struct option longopts[] = {
      {"src", re_arg, 0, 'r'},         {"verbose", no_arg, 0, 'v'},
      {"validation", no_arg, 0, 'V'},  {"with-header", no_arg, 0, 'H'},
      {"with-weight", no_arg, 0, 'W'}, {"directed", no_arg, 0, 'd'},
      {"configs", re_arg, 0, 'c'},     {"json", re_arg, 0, 'j'},
      {"device", re_arg, 0, 'D'},      {0, 0, 0, 0},
  };
  int idx, opt;
  while ((opt = getopt_long(argc - 1, &argv[1], "r:vVHWdc:j:D:", longopts,
                            &idx)) != -1) {
    switch (opt) {
    case 'r':
      cmd_opt.src = atoi(optarg);
      break;
    case 'v':
      cmd_opt.verbose = true;
      break;
    case 'V':
      cmd_opt.validation = true;
      break;
    case 'H':
      cmd_opt.with_header = true;
      break;
    case 'W':
      cmd_opt.with_weight = true;
      break;
    case 'd':
      cmd_opt.directed = true;
      break;
    case 'c':
      cmd_opt.configs = optarg;
      break;
    case 'j':
      cmd_opt.json = optarg;
      break;
    case 'D':
      cmd_opt.device = atoi(optarg);
      break;
    case '?':
    default:
      help(exec);
      exit(0);
    }
  }

  cmd_opt.prase_configs(cmd_opt.configs);
  // if(cmd_opt.expand_mode != "Push" && cmd_opt.expand_mode != "Pull" &&
  // cmd_opt.expand_mode != "Auto"){ puts("expand-mode is only allowed in
  // <Push|Pull|Auto>"); exit(0);
  //}

  // if(cmd_opt.active_set_format != "Bitmap" && cmd_opt.active_set_format !=
  // "Queue" && cmd_opt.active_set_format != "Hybird"){ puts("active-set-format
  // is only allowed in <Bitmap|Queue|Auto>"); exit(0);
  //}

  cudaSetDevice(cmd_opt.device);
}

#endif
