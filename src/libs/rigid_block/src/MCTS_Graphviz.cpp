//
// Created by ziqwang on 16.08.24.
//

#include "mcts/MCTS_Graphviz.h"
#include <fstream>
/*
 * MCTS Graphviz
 */
namespace mcts {
    MCTS_Graphviz::MCTS_Graphviz(const MCTS &tree)
    {
        root_ = tree.root_.lock();
        mapping_ = tree.mapping_;
    }

    std::string MCTS_Graphviz::float_to_string(double val, int precision) {
        std::stringstream stream;
        stream << std::fixed << std::setprecision(precision) << val;
        return stream.str();
    }

    void MCTS_Graphviz::save_tree(std::string filename) {
        auto fout = std::ofstream(filename);
        visited.clear();
        fout << "digraph {\nrankdir=\"LR\";\n";
        save_tree_nodes(fout);
        save_tree_edges(root_, fout);
        fout << "}";
        fout.close();
    }

    void MCTS_Graphviz::save_tree_edges(std::shared_ptr<MCTSNode> node, std::ofstream &fout) {
        if (node == nullptr || visited[node]) {
            return;
        }
        visited[node] = true;

        std::string parent_node_id = std::to_string(node->ind_);
        std::string parent_label = node->label_;

        for (int action_id = 0; action_id < node->n_act(); action_id++) {
            auto child_node = node->get_child(action_id);
            if (child_node != nullptr) {
                std::string child_label = child_node->label_;
                std::string child_node_id = std::to_string(child_node->ind_);

                double ub = node->ub(action_id);
                auto ub_str = float_to_string(ub, 2);

                std::string q_str = float_to_string(node->q_value_[action_id], 2);
                std::string n_str = std::to_string((int) (node->n_visit_[action_id]));
                std::string p_str = float_to_string(node->prior_[action_id] * 0.75 + node->noise_[action_id] * 0.25, 2);
                std::string label_str = "ub(" + ub_str + "), p(" + p_str + ")\n q(" + q_str + "), n(" + n_str + ")";
                fout << parent_node_id << "-> " << child_node_id << "[label = \"" << label_str << "\"";
                // if (check_on_path(child_node) && check_on_path(node)) {
                //     fout << ", color = \"red\"";
                // }
                fout << "];" << std::endl;
                save_tree_edges(child_node, fout);
                //if (check_on_path(child_node) && check_on_path(node)) {
                //}
            }
        }
    }

    void MCTS_Graphviz::save_tree_nodes(std::ofstream &fout) {
        for (auto it = mapping_.begin(); it != mapping_.end(); ++it) {
            auto node = it->second.lock();
            if (node != nullptr) {
                std::string node_n = std::to_string(node->tot_visit_);
                std::string node_r = float_to_string(node->reward_, 0);
                std::string node_id = std::to_string(node->ind_);
                std::string node_label = node->label_;
                //if(!check_on_path(node)) continue;
                fout << node_id << " [label = \"" << node_label << "\"";
                if (node->terminated_) {
                    if (node->reward_ > 0.5) {
                        fout << ", shape = \"diamond\"";
                    } else {
                        fout << ", shape = \"box\"";
                    }
                }
                fout << "];" << std::endl;
            }
        }
    }
}