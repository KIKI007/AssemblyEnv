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

std::vector<int> init_board(int dim){
    std::vector<int> board;
    board.resize(dim * dim, -1);
    //board[0] = 7;
    //board[1] = 0;
    //board[2] = 5;
    //board[3] = 2;
    return board;
}

int board_next_number(const std::vector<int> &board) {
    std::vector<bool> visited;
    visited.resize(board.size(), false);
    for(int id = 0; id < board.size(); id++) {
        if(board[id] != -1) {
            visited[board[id]] = true;
        }
    }
    for(int id = 0; id < visited.size(); id++) {
        if(visited[id] == false)
            return id;
    }
    return -1;
}

std::vector<int> step(const std::vector<int> &board, int action_id) {
    int num = board_next_number(board);
    std::vector<int> new_board = board;
    new_board[action_id] = num;
    return new_board;
}

std::tuple<std::vector<bool>, std::vector<std::vector<int>>> valid_move(const std::vector<int> &board) {
    std::vector<bool> valid_actions(board.size());
    valid_actions.resize(board.size(), true);
    int number = board_next_number(board);

    //remove pos that has been taken
    std::vector<std::vector<int>> child_states;
    for(int pos = 0; pos < board.size(); pos++)
    {
        valid_actions[pos] = (board[pos] == -1);
        if(valid_actions[pos]) {
            auto child_state = step(board, pos);
            child_states.push_back(child_state);
        }
        else {
            child_states.push_back({});
        }
    }
    return {valid_actions, child_states};
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

std::string board_label(const std::vector<int> &board) {
    std::string label = "";
    int dim = std::sqrt(board.size());
    for(int state_id = 0; state_id < board.size(); state_id++) {
        int state = board[state_id];
        if(state >= 0) {
            label += std::to_string(state);
        }
        else {
            label += "*";
        }
        if((state_id + 1) % dim == 0) {
            label += '\n';
        }
        else {
            label += " ";
        }
    }
    return label;
}


std::shared_ptr<MCTSNode> create_node(std::shared_ptr<MCTS> tree, const std::vector<int> &board)
{
    int n_action = board.size();

    std::vector<double> prior; prior.resize(n_action, 1.0);
    std::vector<double> noise;
    noise.resize(n_action, 0);

    auto [valid, child_states] = valid_move(board);
    auto [terminated, reward] = end_game(board);

    auto node = tree->create_node(terminated, reward, board, child_states, prior, noise, valid);
    node->label_ = board_label(board);
    return node;
}

void sim(std::shared_ptr<MCTS> tree)
{

    if(tree->find_leaf()) {
        auto leaf = tree->leaf_node();
        std::vector<int> board = leaf->state_;
        std::vector<int> new_board = step(board, tree->path_endAction());
        auto new_node = create_node(tree, new_board);
        tree->expand(new_node);
    }

    auto leaf = tree->leaf_node();
    if(leaf->terminated_) {
        tree->backward_update(leaf->reward_);
    }
    else {
        //neural network
        tree->backward_update(0);
    }
}

std::vector<double> get_action_prob(std::shared_ptr<MCTS> tree, int num_sim) {
    for(int id = 0; id < num_sim; id++) {
        sim(tree);
    }

    auto root = tree->root_node();
    std::vector<double> prob;
    for(int action_id = 0; action_id < root->n_action_; action_id ++) {
        prob.push_back((double) root->n_visit_[action_id] / root->tot_visit_);
    }

    return prob;
}

std::mt19937 gen(0);

int main()
{
    int dim = 4;
    int n_action = dim * dim;
    std::shared_ptr<MCTS> tree = std::make_shared<MCTS>(n_action, 1.0, 1.0);
    std::vector<int> board = init_board(dim);
    std::shared_ptr<MCTSNode> root = create_node(tree, board);
    tree->set_root(root);
    root = tree->root_;

    std::vector<std::shared_ptr<MCTSNode>> path;
    while(true) {
        auto prob = get_action_prob(tree, 1E3);
        std::discrete_distribution<std::size_t> d{prob.begin(), prob.end()};
        int action_id = std::max_element(prob.begin(), prob.end()) - prob.begin();
        //int action_id = d(gen);
        path.push_back(tree->root_);
        std::cout << tree->root_node()->label_ << std::endl;
        if(tree->execute(action_id)) {
            std::cout << tree->root_node()->label_ << std::endl;
            break;
        }
    }

    MCTS_Graphviz graphviz(*tree);
    graphviz.current_path_ = path;
    graphviz.root_ = root;

    graphviz.save_tree("hello.dot");
}