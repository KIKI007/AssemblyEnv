//
// Created by Ziqi Wang on 11.04.2024.
//

#include "rigid_block/Part.h"
#include "rigid_block/Assembly.h"
#include <rigid_block/MCTSNode.h>
#include <numeric>
#include <iostream>
#include <ctime>
#include <random>
#include <Eigen/Dense>
using namespace rigid_block;


template<typename T>
void print(std::vector<T> result) {
    int pos = 0;
    for(auto digit: result) {
        std::cout << digit << " ";
        pos ++;
        if(pos == 32) {
            pos = 0;
            std::cout << std::endl;
        }

    }
    std::cout << std::endl;
}

int main()
{
    std::vector<int> state;
    std::vector<double> prior;
    int N = 60;
    for(int id = 0; id < N; id++) {
        state.push_back((id * id) % 7 == id % 3);
        prior.push_back((double) (rand() % 100) / 100);
    }
    print(state);

    auto encode_num = encode(state);
    print(encode_num);

    auto decode_num = decode(encode_num, state.size());
    print(decode_num);
}