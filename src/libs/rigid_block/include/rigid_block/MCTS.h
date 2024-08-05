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
#include <map>

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
        bool terminated_ = false;
        double reward_ = 0;
        std::vector<int> state_;
        std::vector<double> prior_;
        std::vector<double> noise_;
        std::vector<bool> valid_action_;
        std::vector<std::vector<int>> child_states_;

    // for visualization
    public:
        int ind_ = 0;
        std::string label_;

    // compute automatically
    public:
        std::vector<double> q_value_;
        std::vector<double> n_visit_;
        int tot_visit_ = 0;
        std::vector<std::shared_ptr<MCTSNode>> children_;

    public:

        MCTSNode(bool terminated,
                 double reward,
                 const std::vector<int> &state,
                 const std::vector<std::vector<int>> &child_states,
                 const std::vector<double> &prior,
                 const std::vector<double> &noise,
                 const std::vector<bool> &valid);

        MCTSNode(const MCTSNode &node);

    public:

        int best_action();

        double ub(int action_id);

        std::shared_ptr<MCTSNode> add_child(int action_id, std::shared_ptr<MCTSNode> node);

        std::shared_ptr<MCTSNode> get_child(int action_id);

        double normalize_prior();

        void update_prior();

    };

    class MCTS
    {
    friend class MCTS_Graphviz;
    //const parameters
    private:
        int n_action_ = 0;
        double cpuct_ = 1.0;
        double discount_ = 1.0;

    //tree search variables
    private:
        std::vector<std::shared_ptr<MCTSNode>> current_path_;
        std::vector<int> current_path_action_;
        std::map<std::vector<int>, std::shared_ptr<MCTSNode>> mapping_;

    public:
        std::shared_ptr<MCTSNode> root_;

    public:

        MCTS(int naction, double cpuct, double discount)
        {
            n_action_ = naction;
            cpuct_ = cpuct;
            discount_ = discount;
        }

    public:

        std::shared_ptr<MCTSNode> create_node(bool terminate,
                                              double reward,
                                              const std::vector<int> &state,
                                              const std::vector<std::vector<int>> &child_states,
                                              const std::vector<double> &prior,
                                              const std::vector<double> &noise,
                                              const std::vector<bool> &valid) const;

    public:

        bool find_leaf();

        std::unique_ptr<MCTSNode> leaf_node();

        int path_endAction() {
            if(!current_path_action_.empty())return current_path_action_.back();
            else return -1;
        }

    public:

        void expand(std::shared_ptr<MCTSNode> node);

        void backward_update(double reward);

        bool execute(int act);

        void set_root(std::shared_ptr<MCTSNode> node) {
            root_ = std::make_shared<MCTSNode>(*node);
            mapping_[root_->state_] = root_;
        }

        void set_root_noise(const std::vector<double> &noise) {
            if(root_) {
                root_->noise_ = noise;
            }
        }

        std::unique_ptr<MCTSNode> root_node() {
            return std::make_unique<MCTSNode>(*root_);
        }

    private:

        std::shared_ptr<MCTSNode> child_node(std::shared_ptr<MCTSNode> node, int action_id);

        std::shared_ptr<MCTSNode> path_endNode() {
            if(!current_path_.empty())return current_path_.back();
            else return nullptr;
        }
    };

    class MCTS_Graphviz {
    public:
        int n_action_;
        std::shared_ptr<MCTSNode> root_;
        std::vector<std::shared_ptr<MCTSNode>> current_path_;
        std::vector<int> current_path_action_;
        std::map<std::vector<int>, std::shared_ptr<MCTSNode>> mapping_;
        std::map<std::shared_ptr<MCTSNode>, bool> visited;
    public:

        MCTS_Graphviz(const MCTS &tree);

    public:
        std::string float_to_string(double val, int precision);

        bool check_on_path(std::shared_ptr<MCTSNode> node);

        void save_tree(std::string filename);

        void save_tree_edges(std::shared_ptr<MCTSNode> node, std::ofstream &fout);

        void save_tree_nodes(std::ofstream &fout);
    };

}

#endif //MCTS_H
