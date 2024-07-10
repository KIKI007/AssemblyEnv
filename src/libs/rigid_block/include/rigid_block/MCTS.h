//
// Created by ziqwang on 10.07.24.
//

#ifndef MCTS_H
#define MCTS_H
#include <complex>
#include <memory>
#include <string>
#include <vector>

namespace rigid_block {
    class MCTSNode {
    public:
        std::string label_ = "";
        bool terminated = false;
        int tot_visit = 0;
        int n_action_ = 0;
        double cpuct_ = 0;
        double eps_ = 1E-8;

    public:
        std::vector<double> prior;
        std::vector<double> noise;
        std::vector<double> q_value;
        std::vector<double> n_visit;
        std::vector<bool> valid_action;

    public:
        std::vector<std::shared_ptr<MCTSNode> > children_;

    public:
        MCTSNode(int n_action) {
            n_action_ = n_action;
            prior.resize(n_action, 0);
            noise.resize(n_action, 0);
            q_value.resize(n_action, 0);
            n_visit.resize(n_action, 0);
            valid_action.resize(n_action, false);
            children_.resize(n_action, nullptr);
        }

    public:
        int best_action();
    };

    class MCTS {
    public:
    };
}

#endif //MCTS_H
