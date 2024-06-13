#include <iostream>

// #include "Engine.h"
#include "graph.h"
#include "io.h"
#include "TCSolver.cuh"

using namespace project_GraphFold;

int main(int argc, char *argv[]) {
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <graph_path>"
              << std::endl;
    return 1;
  }
  std::string path = std::string(argv[1]);
  //project_GraphFold::Pattern pattern(argv[2]);
  std::cout << "Pattern: triangle counting" 
            << "  matching using undirected graphs."
            << "\n"
            << std::endl;
  Loader loader;
  Graph hg; // data graph
  modes cal_mode = e_centric; // TODO: support more cal_mode.
  uint64_t result = 0;
  int n_devices = 1; // TODO: support more devices.

  // load data graph
  loader.Load(path, "mtx");
  loader.Build(hg);
  hg.orientation();
  hg.copyToDevice(0, hg.get_enum(), n_devices, false, false);
  tc_solver(hg, result);

  std::cout << "Result: " << result << "\n" << std::endl;

  return 0;
}