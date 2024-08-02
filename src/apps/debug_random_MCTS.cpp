//
// Created by Ziqi Wang on 11.04.2024.
//

#include "../libs/rigid_block/include/rigid_block/Part.h"
#include "../libs/rigid_block/include/rigid_block/Assembly.h"
#include <../libs/rigid_block/include/rigid_block/MCTS.h>
#include <numeric>
#include <iostream>
#include <ctime>
#include <random>
using namespace rigid_block;

int id = 0;
//std::random_device rd;
std::mt19937 gen(10); // these can be global and/or static, depending on how you use random elsewhere
std::uniform_real_distribution<> dis_double(0, 1);
std::uniform_int_distribution<> dis_int(0, 1);

std::vector<double> random(int n) {
    std::vector<double> values(n);
    std::generate(values.begin(), values.end(), [&](){ return dis_double(gen); });
    return values;
}

std::vector<int> random_int(int n) {
    std::vector<int> values(n);
    std::generate(values.begin(), values.end(), [&](){ return dis_int(gen); });
    return values;
}

std::shared_ptr<MCTSNode> random_node(const MCTS &tree) {
    int n_action = tree.n_action_;
    std::vector<double> prior; prior.resize(n_action, 1.0);
    std::vector<double> noise;
    noise.resize(n_action, 0);
    std::vector<int> temp = random_int(n_action);
    std::vector<bool> valid;
    for(auto v : temp) {
        if(v > 0) valid.push_back(true);
        else valid.push_back(false);
    }

    std::vector<int> state = {id++};
    double reward = 0;
    bool terminated = random(1)[0] > 0.7;
    if(terminated){
        double val = (random(1)[0] > 0.5);
        reward = (val - 0.5) * 2;
    }
    return tree.create_node(terminated, reward, state, prior, noise, valid);
}


int main()
{
    int n_action = 5;
    std::shared_ptr<MCTS> tree = std::make_shared<MCTS>(n_action, 1.0);
    auto root = random_node(*tree);
    root->terminated_ = false;
    tree->set_root(root);
    for(int sim = 0; sim < 1000; sim ++)
    {
        if(tree->find_leaf(tree->root_)) {
            auto new_node = random_node(*tree);
            tree->expand(new_node);
        }
        std::shared_ptr<MCTSNode> leaf_node = tree->path_endNode();
        if(leaf_node->terminated_)
        {
            tree->backup(tree->path_endNode()->reward_);
        }
        else {
            double v = (random(1)[0] - 0.5) * 2;
            tree->backup(0);
        }
    }
    tree->find_leaf(tree->root_);
    tree->save_tree("hello.dot");
}