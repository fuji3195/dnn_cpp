#ifndef MAIN_HPP__
#define MAIN_HPP__

#include <vector>
#include <string>
#include <iostream>
using std::vector;
using std::pair;
using std::string;
using std::cout;
using std::endl;
using VD = vector<double>;
using VVD = vector<VD>;
#define rep(i,n)    for(int i=0; i<(int)(n); ++i)
#define fore(i,n)   for(auto i : (n))

pair<VVD, VVD> get_mnist_data(const string& path);

#endif