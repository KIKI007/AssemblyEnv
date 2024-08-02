//
// Created by ziqwang on 10.07.24.
//

#ifndef MCTS_H
#define MCTS_H
#include <complex>
#include <memory>
#include <string>
#include <vector>
#include <fstream>
#include <iomanip>
#include <iostream>

namespace rigid_block
{
    class MCTSNode
    {

    //const parameters, same across all node
    public:
        double cpuct_ = 0;
        double eps_ = 1E-8;
        int n_action_ = 0;

    // set by users
    public:
        int node_id_ = 0;
        bool terminated_ = false;
        double reward_ = 0;
        std::vector<int> state_;
        std::vector<double> prior_;
        std::vector<double> noise_;
        std::vector<bool> valid_action_;

    // compute automatically
    public:
        std::vector<double> q_value_;
        std::vector<double> n_visit_;
        int tot_visit_ = 0;
        std::vector<std::shared_ptr<MCTSNode>> children_;

    public:

        MCTSNode(int node_id,
                 bool terminated,
                 double reward,
                 const std::vector<int> &state,
                 const std::vector<double> &prior,
                 const std::vector<double> &noise,
                 const std::vector<bool> &valid);

        MCTSNode(const MCTSNode &node);

    public:

        std::string node_label();

        int best_action();

        double ub(int action_id);

        std::shared_ptr<MCTSNode> add_child(int action_id, std::shared_ptr<MCTSNode> node);

        std::shared_ptr<MCTSNode> get_child(int action_id);

        double normalize_prior();
        void update_prior();

    };

    class MCTS
    {
    public:
        int n_action_;
        double cpuct_;

        std::shared_ptr<MCTSNode> root_;

        std::vector<std::shared_ptr<MCTSNode>> current_path_;
        std::vector<int> current_path_action_;

    public:

        MCTS(int naction, double cpuct)
        {
            n_action_ = naction;
            cpuct_ = cpuct;
        }

    public:

        std::string float_to_string(double val, int precision);

        bool check_on_path(std::shared_ptr<MCTSNode> node);

        void save_tree(std::string filename);

        void save_tree(std::shared_ptr<MCTSNode> node, std::ofstream &fout);

        std::shared_ptr<MCTSNode> create_node(int node_id,
                                              bool terminate,
                                              double reward,
                                              const std::vector<int> &state,
                                              const std::vector<double> &prior,
                                              const std::vector<double> &noise,
                                              const std::vector<bool> &valid) const;


        bool find_leaf(std::shared_ptr<MCTSNode> node);

        void expand(std::shared_ptr<MCTSNode> node);

        void backup(double reward);

        void set_root(std::shared_ptr<MCTSNode> node) {
            root_ = std::make_shared<MCTSNode>(*node);
        }

        std::shared_ptr<MCTSNode> path_endNode() {
            if(!current_path_.empty())return current_path_.back();
            else return nullptr;
        }

        int path_endAction() {
            if(!current_path_action_.empty())return current_path_action_.back();
            else return -1;
        }

        bool execute(int act) {
            if(root_->children_[act] != nullptr) {
                root_ = root_->children_[act];
                return true;
            }
            else {
                return false;
            }
        }
    };
}

#endif //MCTS_H
