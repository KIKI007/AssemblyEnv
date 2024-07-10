//
// Created by ziqwang on 10.07.24.
//

#include "rigid_block/MCTS.h"

int rigid_block::MCTSNode::best_action() {
    double best_u = std::numeric_limits<double>::lowest();
    double best_act = 0;
    for (int action_id = 0; action_id < n_action_; action_id++) {
        ifvalid_action[action_id] {
            double u = 0;
            if (children_[action_id] != nullptr) {
                u = (q_value[action_id]
                     + cpuct_ * prior[action_id] * std::sqrt(tot_visit) / (
                         1 + n_visit[action_id]));
            } else {
                u = cpuct_ * prior[action_id] * std::sqrt(tot_visit + eps_);
            }

            if (u > best_u) {
                best_u = u;
                best_act = action_id;
            }
        }
    }
}
