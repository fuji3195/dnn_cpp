#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include "main.hpp"
using std::ifstream;

pair<VVD,VVD> get_mnist_data(const string& file_path) {
    ifstream ifs(file_path);
    if (!ifs.is_open()) {cout << "file " << file_path << "is not exist" << endl; exit(1);}
    string context = string(std::istreambuf_iterator<char>(ifs), std::istreambuf_iterator<char>());
    VVD x;
    VVD y;
    long long sz = context.size(); long long i = 0; int dtsz = 28*28;
    while (i<sz) {
        int tmp_y = context[i]; ++i;
        VD y_1hot(10,0); y_1hot[tmp_y] = 1.0;
        y.push_back(y_1hot);
        VD v(dtsz);
        for (int bit=0; bit<dtsz; ++bit, ++i) v[bit] = (double)(context[i]&0xFF) / 255;
        x.push_back(v);
    }
    return std::make_pair(x,y);
}
