//
// Created by ziqwang on 10.07.24.
//

#ifndef MCTS_H
#define MCTS_H

#include <memory>
#include <string>
#include <vector>
#include <iomanip>
#include "encoder.h"
namespace mcts
{
    class MCTSNode
    {

    public:
        //const parameters, same across all node
        double cpuct_ = 0;
        double eps_ = 1E-8;

    public:
        //main variables
        int n_state_ = 0;
        bool terminated_ = false;
        double reward_ = 0;
        std::vector<uint32_t> state_; //encode list
        std::vector<double> prior_;
        std::vector<double> noise_;
        std::vector<int> action_to_valid, valid_to_action;

    public:
        // internal variables for MCTS
        std::vector<double> q_value_;
        std::vector<double> n_visit_;
        int tot_visit_ = 0;
        std::vector<std::weak_ptr<MCTSNode>> children_;

        // for graph visualization
    public:
        int ind_ = 0;
        std::string label_;

    public:

        /*
         * state: must be a list of binary 0/1
         * valid: must be a list of binary 0/1
         */
        MCTSNode(bool terminated,
                 double reward,
                 double cpuct,
                 const std::vector<int> &state,
                 const std::vector<double> &prior,
                 const std::vector<double> &noise,
                 const std::vector<int> &valid);

        MCTSNode(const MCTSNode &node);

        //functions for reading node's data
    public:

        int n_valid_act(){return valid_to_action.size();}

        int n_act(){return action_to_valid.size();}

        std::vector<int> v();

        std::vector<int> s();

        std::vector<int> sa(int action_id);

        std::vector<int> na();

        int N();

        std::vector<double> prior();

        std::vector<double> noise();

        //function for taking actions
    public:

        int best_action();

        double ub(int valid_act);

        bool add_child(int action_id, std::shared_ptr<MCTSNode> node);

        std::shared_ptr<MCTSNode> get_child(int action_id);

        //function for updating node's data
    public:

        void update_prior();

        double normalize_prior();

        void update_na(int action_id, int delta = 1);

        void update_qa(int action_id, double delta = 0.0);

        void update_noise(const std::vector<double> &noise);

    };
}

#endif //MCTS_H
