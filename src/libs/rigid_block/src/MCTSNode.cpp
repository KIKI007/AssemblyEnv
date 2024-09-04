//
// Created by ziqwang on 10.07.24.
//

#include "mcts/MCTS.h"
#include <limits>
#include <cmath>
/*
 * MCTS NODE
 */
namespace mcts {
    MCTSNode::MCTSNode(
        bool terminated,
        double reward,
        double cpuct,
        const std::vector<int> &state,
        const std::vector<double> &prior,
        const std::vector<double> &noise,
        const std::vector<int> &valid)
    {
        terminated_ = terminated;
        reward_ = reward;
        cpuct_ = cpuct;

        // encode state
        n_state_ = state.size();
        state_ = encode(state);

        // build map between action_id and valid_action_id
        int sum = 0;
        for (int id = 0; id < valid.size(); id++)
        {
            if (valid[id])
            {
                action_to_valid.push_back(sum);
                valid_to_action.push_back(id);
                prior_.push_back(prior[id]);
                noise_.push_back(noise[id]);
                sum++;
            } else {
                action_to_valid.push_back(-1);
            }
        }

        q_value_.resize(n_valid_act(), 0);
        n_visit_.resize(n_valid_act(), 0);
        children_.resize(n_valid_act(), std::weak_ptr<MCTSNode>());

        label_ = "";
        for (auto digit: state_) {
            label_ += std::to_string(digit) + " ";
        }
    }

    MCTSNode::MCTSNode(const MCTSNode &node) {
        //const parameters
        n_state_ = node.n_state_;
        cpuct_ = node.cpuct_;
        eps_ = node.eps_;

        //for visualization
        ind_ = node.ind_;
        label_ = node.label_;

        //set by user
        terminated_ = node.terminated_;
        reward_ = node.reward_;
        state_ = node.state_;

        prior_ = node.prior_;
        noise_ = node.noise_;
        action_to_valid = node.action_to_valid;
        valid_to_action = node.valid_to_action;

        //computed automatically
        tot_visit_ = node.tot_visit_;
        q_value_ = node.q_value_;
        n_visit_ = node.n_visit_;

        //we cannot copy the children info
        children_.clear();
        children_.resize(n_valid_act(), std::weak_ptr<MCTSNode>());
    }

    bool MCTSNode::add_child(int action_id, std::shared_ptr<MCTSNode> node) {
        int valid_act = action_to_valid[action_id];
        if (valid_act != -1) {
            children_[valid_act] = node;
            return true;
        }
        return false;
    }

    std::shared_ptr<MCTSNode> MCTSNode::get_child(int action_id) {
        int valid_act = action_to_valid[action_id];
        if (valid_act != -1) {
            return children_[valid_act].lock();
        }
        return nullptr;
    }

    double MCTSNode::normalize_prior() {
        double sum_prior = 0;
        for (int valid_act = 0; valid_act < n_valid_act(); valid_act++) {
            sum_prior += prior_[valid_act];
        }

        if (sum_prior > 0) {
            for (int valid_act = 0; valid_act < n_valid_act(); valid_act++) {
                prior_[valid_act] /= sum_prior;
            }
        }
        return sum_prior;
    }

    void MCTSNode::update_prior() {
        //normalize the prior
        double sum_prior = normalize_prior();
        //if normalize fails
        if (sum_prior == 0) {
            for (int valid_act = 0; valid_act < n_valid_act(); valid_act++) {
                prior_[valid_act] += 1;
            }
            normalize_prior();
        }
    }

    int MCTSNode::best_action() {
        double best_u = std::numeric_limits<double>::lowest();
        double best_act = 0;

        for (int valid_act = 0; valid_act < n_valid_act(); valid_act++) {
            double u = ub(valid_act);
            if (u > best_u) {
                best_u = u;
                best_act = valid_to_action[valid_act];
            }
        }
        return best_act;
    }

    double MCTSNode::ub(int valid_act)
    {
        //prior = prior * 0.75 + noise * 0.25
        double prior = prior_[valid_act] * 0.75 + noise_[valid_act] * 0.25;
        auto child_node = children_[valid_act].lock();
        if (child_node != nullptr) {
            return (q_value_[valid_act]
                    + cpuct_ * prior * std::sqrt(tot_visit_) / (
                        1 + n_visit_[valid_act]));
        } else {
            return cpuct_ * prior * std::sqrt(tot_visit_ + eps_);
        }
    }

    std::vector<int> MCTSNode::v() {
        std::vector<int> valids;
        for (int ind: action_to_valid) {
            valids.push_back(ind != -1);
        }
        return valids;
    }

    std::vector<int> MCTSNode::s() {
        return decode(state_, n_state_);
    }

    std::vector<int> MCTSNode::sa(int action_id) {
        if (action_to_valid[action_id] == -1) {
            return {};
        }
        int n_part = n_act() / 2;
        std::vector<int> child_state = s();
        if (action_id >= n_part) {
            child_state[action_id - n_part] = 0;
            child_state[action_id] = 0;
        } else {
            child_state[action_id + n_part] = 1;
        }
        return child_state;
    }

    std::vector<double> MCTSNode::prior() {
        std::vector<double> vals;
        vals.resize(n_act(), 0);
        for(int valid_act = 0; valid_act < n_valid_act(); valid_act++) {
            int action_id = valid_to_action[valid_act];
            vals[action_id] = prior_[valid_act];
        }
        return vals;
    }

    std::vector<double> MCTSNode::noise() {
        std::vector<double> vals;
        vals.resize(n_act(), 0);
        for(int valid_act = 0; valid_act < n_valid_act(); valid_act++) {
            int action_id = valid_to_action[valid_act];
            vals[action_id] = noise_[valid_act];
        }
        return vals;
    }

    std::vector<int> MCTSNode::na() {
        std::vector<int> na;
        na.resize(n_act(), 0);
        for (int id = 0; id < n_valid_act(); id++) {
            na[valid_to_action[id]] = n_visit_[id];
        }
        return na;
    }

    int MCTSNode::N() {
        return tot_visit_;
    }

    void MCTSNode::update_na(int action_id, int delta) {
        int valid_act = action_to_valid[action_id];
        if (valid_act != -1) {
            n_visit_[valid_act] += delta;
        }
    }

    void MCTSNode::update_qa(int action_id, double delta) {
        int valid_act = action_to_valid[action_id];
        if (valid_act != -1) {
            q_value_[valid_act] = (n_visit_[valid_act] * q_value_[valid_act] + delta) /
                                  (n_visit_[valid_act] + 1);
        }
    }

    void MCTSNode::update_noise(const std::vector<double> &noise) {
        noise_.clear();
        for(int valid_act = 0; valid_act < n_valid_act(); valid_act++) {
            int action_id = valid_to_action[valid_act];
            noise_.push_back(noise[action_id]);
        }
    }
}
