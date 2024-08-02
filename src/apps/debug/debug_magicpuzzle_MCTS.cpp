//
// Created by Ziqi Wang on 11.04.2024.
//

#include "rigid_block/Part.h"
#include "rigid_block/Assembly.h"
#include <rigid_block/MCTS.h>
#include <numeric>
#include <iostream>
#include <ctime>
#include <random>
#include <Eigen/Dense>
using namespace rigid_block;

int node_ind = 0;

std::vector<int> init_board(int dim){
    std::vector<int> board;
    board.resize(dim * dim, -1);
    board[0] = 7;
    // board[1] = 0;
    // board[2] = 5;
    // board[3] = 2;
    return board;
}

std::vector<bool> valid_move(const std::vector<int> &board) {
    int dim = std::sqrt(board.size());
    std::vector<bool> valid_actions(board.size() * board.size());
    valid_actions.resize(board.size() * board.size(), true);

    std::vector<bool> number_exist;
    number_exist.resize(board.size(), false);
    for(int pos = 0; pos < board.size(); pos++)
    {
        if(board[pos] != -1) {
            number_exist[board[pos]] = true;
        }
    }

    //remove pos that has been taken
    for(int pos = 0; pos < board.size(); pos++) {
        bool pos_valid = (board[pos] == -1);
        for(int number = 0; number < board.size(); number++) {
            int ind = pos * board.size() + number;
            valid_actions[ind] = pos_valid * (1 - number_exist[number]);
        }
    }
    return valid_actions;
}

Eigen::MatrixXi matrix_board(const std::vector<int> &board) {
    int dim = std::sqrt(board.size());
    Eigen::MatrixXi matrix(dim, dim);
    for(int x = 0; x < dim; x++) {
        for(int y = 0; y < dim; y++) {
            matrix(x, y) = board[dim * x + y];
        }
    }
    return matrix;
}

std::tuple<bool, double> end_game(const std::vector<int> &board) {
    int dim = std::sqrt(board.size());
    Eigen::MatrixXi mat = matrix_board(board);
    std::vector<int> vals;
    std::cout << mat << std::endl;

    //row sum
    for(int x = 0; x < dim; x++)
    {
        int val = 0;
        for(int y = 0; y < dim; y++) {
            if(mat(x, y) != -1) {
                val += mat(x, y);
            }
            else {
                val = -1;
                break;
            }
        }
        if(val != -1) {
            vals.push_back(val);
        }
    }

    //col sum
    for(int y = 0; y < dim; y++)
    {
        int val = 0;
        for(int x = 0; x < dim; x++) {
            if(mat(x, y) != -1) {
                val += mat(x, y);
            }
            else {
                val = -1;
                break;
            }
        }
        if(val != -1) {
            vals.push_back(val);
        }
    }

    //diag right
    int val = 0;
    for(int x = 0; x < dim; x++) {
        int y = x;
        if(mat(x, y) != -1) {
            val += mat(x, y);
        }
        else {
            val = -1;
            break;
        }
    }
    if(val != -1) {
        vals.push_back(val);
    }

    //diag left
    val = 0;
    for(int x = 0; x < dim; x++) {
        int y = dim - x - 1;
        if(mat(x, y) != -1) {
            val += mat(x, y);
        }
        else {
            val = -1;
            break;
        }
    }
    if(val != -1) {
        vals.push_back(val);
    }

    //check valid
    if(vals.empty()) {
        return {false, 0.0};
    }
    val = vals.front();
    for(int id = 0; id < vals.size(); id++) {
        if(vals[id] != val) {
            return {true, -1.0};
        }
    }
    if(vals.size() == dim * 2 + 2) {
        return {true, 1.0};
    }
    else {
        return {false, 0.0};
    }
}


std::vector<int> step(const std::vector<int> &board, int action_id) {
    int pos = action_id / board.size();
    int num = action_id % board.size();
    std::vector<int> new_board = board;
    new_board[pos] = num;
    return new_board;
}

std::shared_ptr<MCTSNode> create_node(std::shared_ptr<MCTS> tree, const std::vector<int> &board)
{
    int n_action = board.size() * board.size();

    std::vector<double> prior; prior.resize(n_action, 1.0);
    std::vector<double> noise;
    noise.resize(n_action, 0);

    std::vector<bool> valid = valid_move(board);
    auto [terminated, reward] = end_game(board);

    return tree->create_node(node_ind ++, terminated, reward, board, prior, noise, valid);
}


int main()
{
    int dim = 3;
    int n_action = dim * dim * dim * dim;
    std::shared_ptr<MCTS> tree = std::make_shared<MCTS>(n_action, 1.0);
    std::vector<int> board = init_board(dim);
    std::shared_ptr<MCTSNode> root = create_node(tree, board);
    tree->set_root(root);
    for(int sim = 0; sim < 1000; sim ++)
    {
        if(tree->find_leaf(tree->root_)) {
            std::vector<int> board = tree->path_endNode()->state_;
            std::vector<int> new_board = step(board, tree->path_endAction());
            auto new_node = create_node(tree, new_board);
            tree->expand(new_node);
        }
        std::shared_ptr<MCTSNode> leaf_node = tree->path_endNode();
        if(leaf_node->terminated_)
        {
            tree->backup(tree->path_endNode()->reward_);
        }
        else {
            tree->backup(0);
        }
    }
    tree->find_leaf(tree->root_);
    tree->save_tree("hello.dot");
}