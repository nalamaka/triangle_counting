#ifndef GRAPH_IO_H
#define GRAPH_IO_H

#include <algorithm>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <queue>
#include <vector>

#include "graph.h"
#include "logging.h"
#include "timer.h"

namespace project_GraphFold {

// TODO: support multi-GPU partitioner. and multi-thread.
class Loader {
public:
  void Load(std::string const &path, std::string const &format) {
    std::cout << "load graph from " << path << " in " << format << " format"
              << std::endl;
    if (format == "mtx")
      _load_mtx(path, true);
    else if (format == "txt")
      _load_txt(path);
    else if (format == "bin")
      _load_bin(path);
    else {
      std::cout << "unknown format " << format << std::endl;
    }
  }

  void Loadint(std::string const &path) {
    std::cout << "load vertex int from " << path << std::endl;
    auto isprefix = [](std::string prefix, std::string path) {
      return std::mismatch(prefix.begin(), prefix.end(), path.begin()).first ==
             prefix.end();
    };

    // if from random
    if (isprefix(std::string("random"), path)) {
      size_t range = std::stoi(path.substr(6));
      for (size_t i = 0; i < vsize_; ++i) {
        vids_.emplace_back(rand() % range);
      }
      return;
    }

    // if from file
    std::ifstream fin(path);
    if (!fin.is_open()) {
      std::cout << "cannot open " << path << std::endl;
      exit(-1);
    }
    std::vector<std::pair<int, int>> tmp;
    while (fin.good()) {
      int v;
      int vid;
      fin >> v >> vid;
      fin.ignore(std::numeric_limits<std::streamsize>::max(), fin.widen('\n'));
      if (fin.eof())
        break;
      tmp.emplace_back(v, vid);
    }
    if (tmp.size() != vsize_) {
      std::cout << "vertex size not match" << std::endl;
    }
    vids_.resize(vsize_);
    for (auto &p : tmp) {
      vids_[p.first - vmin_] = p.second;
    }
  }

  void Build(Graph &hg) {
    std::vector<int> vids;
    for (size_t i = 0; i < vsize_; ++i) {
      vids.push_back(i);
    }
    hg.Init(vids, vids_, edges_, 1);
  }

  void Build(Graph &hg, int n_dev) {
    std::vector<int> vids;
    for (size_t i = 0; i < vsize_; ++i) {
      vids.push_back(i);
    }

    int ndevices = n_dev;
    hg.Init(vids, vids_, edges_, ndevices); // i.e. even_split
  }

protected:
  void _load_mtx(std::string const &path, bool with_header = false) {
    double start = wtime();
    std::ifstream fin(path);
    if (!fin.is_open()) {
      std::cout << "cannot open file " << path << std::endl;
      exit(-1);
    }

    // skip comments
    while (1) {
      char c = fin.peek();
      if (c >= '0' && c <= '9')
        break;
      fin.ignore(std::numeric_limits<std::streamsize>::max(), fin.widen('\n'));
    }

    // if with_header == true
    if (with_header) {
      fin >> vsize_ >> vsize_ >> esize_;
    } else {
      vsize_ = esize_ = 0;
    }

    edges_.clear();
    vids_.clear();
    vmin_ = std::numeric_limits<int>::max();
    int vmax = std::numeric_limits<int>::min();

    // loop lines
    while (fin.good()) {
      int v0, v1;
      fin >> v0 >> v1;
      fin.ignore(std::numeric_limits<std::streamsize>::max(), fin.widen('\n'));
      if (fin.eof())
        break;
      if (v0 == v1)
        continue;

      vmin_ = std::min(vmin_, std::min(v0, v1));
      vmax = std::max(vmax, std::max(v0, v1));

      edges_.emplace_back(v0, v1);
      if (fin.eof())
        break;
    }

    vsize_ = vmax - vmin_ + 1;
    esize_ = edges_.size();
    // std::cout << esize_;
    for (auto &item : edges_) {
      item.first -= vmin_;
      item.second -= vmin_;
    }

    fin.close();
    double end = wtime();
    std::cout << "load mtx graph in " << (end - start) << " seconds"
              << std::endl;
  }

  // TODO: support other formats
  void _load_txt(std::string const &path) {
    std::cout << "not implemented" << std::endl;
  }
  void _load_bin(std::string const &path) {
    std::cout << "not implemented" << std::endl;
  }

private:
  size_t vsize_;
  size_t esize_;
  int vmin_;
  std::vector<std::pair<int, int>> edges_;
  std::vector<int> vids_;
  bool hasint_ = false;
  // TODO: maybe support evids_
};

} // namespace project_GraphFold

#endif // IO_H
