//
// Created by ziqwang on 10.07.24.
//

#include "mcts/MCTS.h"
#include <algorithm>
/*
 *MCTS
 */
namespace mcts
{
    std::shared_ptr<MCTSNode> MCTS::child_node(std::shared_ptr<MCTSNode> node, int action_id) {
        // if child node exists
        std::shared_ptr<MCTSNode> child_node = node->get_child(action_id);
        if (child_node != nullptr) {
            return child_node;
        }

        // if not, searching the map
        std::vector<int> child_state = node->sa(action_id);
        std::vector<uint32_t> encode_child_state = encode(child_state);
        if (encode_child_state.empty()) {
            return nullptr;
        }
        auto find_it = mapping_.find(encode_child_state);
        if (find_it != mapping_.end()) {
            int valid_act = node->action_to_valid[action_id];
            node->children_[valid_act] = find_it->second.lock();
            return find_it->second.lock();
        }
        return nullptr;
    }

    std::tuple<bool, std::vector<std::shared_ptr<MCTSNode>>, std::vector<int>> MCTS::find_leaf()
    {
        std::vector<std::shared_ptr<MCTSNode>> path;
        std::vector<int> path_act;
        auto node = current_root_.lock();
        while (node != nullptr) {
            path.push_back(node);
            if (node->terminated_) {
                return {true, path, path_act};
            } else {
                int action_id = node->best_action();
                path_act.push_back(action_id);
                node = child_node(node, action_id);
            }
        }
        return {false, path, path_act};
    }

    bool MCTS::execute(int act)
    {
        auto child = child_node(current_root_.lock(), act);
        if (child != nullptr) {
            current_root_ = child;
            if (child->terminated_) {
                return true;
            } else {
                return false;
            }
        }
        return true;
    }

    std::shared_ptr<MCTSNode> MCTS::new_node(const MCTSNode &node)
    {
        std::shared_ptr<MCTSNode> node_ptr = std::make_shared<MCTSNode>(node);
        node_ptr->ind_ = nodes_.size();
        nodes_.push_back(node_ptr);
        mapping_[node_ptr->state_] = node_ptr;
        return node_ptr;
    }

    void MCTS::save(const std::vector<std::shared_ptr<MCTSNode>> &nodes) {
        stored_nodes.clear();
        for(auto node : nodes_) {
            stored_nodes.push_back(std::make_shared<MCTSNode>(*node));
        }
    }

    void MCTS::reload()
    {
        for(auto node: stored_nodes)
        {
            std::shared_ptr<MCTSNode> node_in_tree = nodes_[node->ind_];
            node_in_tree->q_value_ = node->q_value_;
            node_in_tree->n_visit_ = node->n_visit_;
            node_in_tree->tot_visit_ = node->tot_visit_;
        }
    }

    void MCTS::set_root(std::shared_ptr<MCTSNode> node)
    {
        current_root_ = nodes_[node->ind_];
        root_ = nodes_[node->ind_];
    }

    void MCTS::update(double v,
                double discount,
                const std::vector<std::shared_ptr<MCTSNode>> &path,
                const std::vector<int> &path_act)
    {
        for (int id = (int) path.size() - 1; id >= 0; id--) {
            int action_id = path_act[id];
            auto node = path[id];
            auto node_in_tree = nodes_[node->ind_];
            node_in_tree->update_na(action_id, 1);
            node_in_tree->update_qa(action_id, v);
            node_in_tree->tot_visit_++;
            v = v * discount;
        }
    }
}

