//
// Created by ziqwang on 10.07.24.
//

#include "rigid_block/MCTS.h"

rigid_block::MCTSNode::MCTSNode(
    bool terminated,
    double reward,
    const std::vector<int> &state,
    const std::vector<std::vector<int> > &child_states,
    const std::vector<double> &prior,
    const std::vector<double> &noise,
    const std::vector<bool> &valid) {
    terminated_ = terminated;
    reward_ = reward;

    state_ = state;
    child_states_ = child_states;
    prior_ = prior;
    noise_ = noise;
    valid_action_ = valid;

    n_action_ = prior.size();

    q_value_.resize(n_action_, 0);
    n_visit_.resize(n_action_, 0);
    children_.resize(n_action_, nullptr);
}

rigid_block::MCTSNode::MCTSNode(const MCTSNode &node) {
    //const parameters
    n_action_ = node.n_action_;
    cpuct_ = node.cpuct_;
    eps_ = node.eps_;

    //for visualization
    ind_ = node.ind_;
    label_ = node.label_;

    //set by user
    terminated_ = node.terminated_;
    reward_ = node.reward_;
    state_ = node.state_;
    child_states_ = node.child_states_;
    prior_ = node.prior_;
    noise_ = node.noise_;
    valid_action_ = node.valid_action_;

    //computed automatically
    tot_visit_ = node.tot_visit_;
    q_value_ = node.q_value_;
    n_visit_ = node.n_visit_;

    //we cannot copy the children info
    children_.clear();
    children_.resize(n_action_, nullptr);
}

std::shared_ptr<rigid_block::MCTSNode> rigid_block::MCTSNode::add_child(int action_id, std::shared_ptr<MCTSNode> node) {
    std::shared_ptr<MCTSNode> new_node = std::make_shared<MCTSNode>(*node);
    children_[action_id] = new_node;
    return new_node;
}

std::shared_ptr<rigid_block::MCTSNode> rigid_block::MCTSNode::get_child(int action_id) {
    if (children_[action_id] != nullptr) {
        return children_[action_id];
    }
    return nullptr;
}

double rigid_block::MCTSNode::normalize_prior() {
    double sum_prior = 0;

    for (int action_id = 0; action_id < n_action_; action_id++) {
        sum_prior += prior_[action_id];
    }

    if (sum_prior > 0) {
        for (int action_id = 0; action_id < n_action_; action_id++) {
            prior_[action_id] /= sum_prior;
        }
    }
    return sum_prior;
}

void rigid_block::MCTSNode::update_prior() {
    //remove prior with invalid moves
    for (int action_id = 0; action_id < n_action_; action_id++) {
        prior_[action_id] = prior_[action_id] * valid_action_[action_id];
    }

    //normalize the prior
    double sum_prior = normalize_prior();

    //if normalize fails
    if (sum_prior == 0) {
        for (int action_id = 0; action_id < n_action_; action_id++) {
            prior_[action_id] += valid_action_[action_id];
        }
        normalize_prior();
    }
}

int rigid_block::MCTSNode::best_action() {
    double best_u = std::numeric_limits<double>::lowest();
    double best_act = 0;
    for (int action_id = 0; action_id < n_action_; action_id++) {
        if (valid_action_[action_id]) {
            double u = ub(action_id);
            if (u > best_u) {
                best_u = u;
                best_act = action_id;
            }
        }
    }
    return best_act;
}

double rigid_block::MCTSNode::ub(int action_id) {
    //prior = prior * 0.75 + noise * 0.25
    double prior = prior_[action_id] * 0.75 + noise_[action_id] * 0.25;

    if (children_[action_id] != nullptr) {
        return (q_value_[action_id]
                + cpuct_ * prior * std::sqrt(tot_visit_) / (
                    1 + n_visit_[action_id]));
    } else {
        return cpuct_ * prior * std::sqrt(tot_visit_ + eps_);
    }
}

/*
 *MCTS
 */

std::shared_ptr<rigid_block::MCTSNode> rigid_block::MCTS::create_node(
    bool terminate,
    double reward,
    const std::vector<int> &state,
    const std::vector<std::vector<int> > &child_states,
    const std::vector<double> &prior,
    const std::vector<double> &noise,
    const std::vector<bool> &valid) const {
    std::shared_ptr<MCTSNode> node
            = std::make_shared<MCTSNode>(terminate, reward, state, child_states, prior, noise, valid);
    node->update_prior();
    node->cpuct_ = cpuct_;
    return node;
}

std::shared_ptr<rigid_block::MCTSNode> rigid_block::MCTS::child_node(std::shared_ptr<rigid_block::MCTSNode> node,
                                                                     int action_id) {
    if (node->children_[action_id] != nullptr) {
        return node->children_[action_id];
    }

    auto child_state = node->child_states_[action_id];
    if (child_state.empty()) {
        return nullptr;
    }

    auto find_it = mapping_.find(child_state);
    if (find_it != mapping_.end()) {
        node->children_[action_id] = find_it->second;
        return find_it->second;
    }
    return nullptr;
}

bool rigid_block::MCTS::find_leaf() {
    current_path_.clear();
    current_path_action_.clear();
    auto node = root_;
    while (node != nullptr) {
        current_path_.push_back(node);
        if (node->terminated_) {
            return false;
        } else {
            int act = node->best_action();
            current_path_action_.push_back(act);
            node = child_node(node, act);
        }
    }
    return true;
}

std::unique_ptr<rigid_block::MCTSNode> rigid_block::MCTS::leaf_node() {
    if (path_endNode()) {
        std::unique_ptr<MCTSNode> leaf
                = std::make_unique<MCTSNode>(*path_endNode());
        return leaf;
    }
    return nullptr;
}

void rigid_block::MCTS::expand(std::shared_ptr<MCTSNode> node) {
    auto find_it = mapping_.find(node->state_);
    if (find_it != mapping_.end()) {
        node = find_it->second;
    }

    auto new_node = path_endNode()->add_child(path_endAction(), node);
    new_node->ind_ = mapping_.size();
    current_path_.push_back(new_node);
    mapping_[node->state_] = new_node;
}

void rigid_block::MCTS::backward_update(double v) {
    // |path_node| = |path_action| + 1
    // the last node is a leaf node which does not need to be updated
    for (int id = (int) current_path_action_.size() - 1; id >= 0; id--) {
        int action_id = current_path_action_[id];
        auto node = current_path_[id];
        node->q_value_[action_id] = (node->n_visit_[action_id] * node->q_value_[action_id] + v) / (
                                        node->n_visit_[action_id] + 1);
        node->n_visit_[action_id] += 1;
        node->tot_visit_++;
        v = v * discount_;
    }
}

bool rigid_block::MCTS::execute(int act) {
    auto child = child_node(root_, act);
    if (child != nullptr) {
        root_ = child;
        if(root_->terminated_) {
            return true;
        }
        else {
            return false;
        }
    }
    return true;
}

/*
 * MCTS Graphviz
 */

rigid_block::MCTS_Graphviz::MCTS_Graphviz(const MCTS &tree) {
    root_ = tree.root_;
    current_path_ = tree.current_path_;
    current_path_action_ = tree.current_path_action_;
    mapping_ = tree.mapping_;
    n_action_ = tree.n_action_;
}

std::string rigid_block::MCTS_Graphviz::float_to_string(double val, int precision) {
    std::stringstream stream;
    stream << std::fixed << std::setprecision(precision) << val;
    return stream.str();
}

bool rigid_block::MCTS_Graphviz::check_on_path(std::shared_ptr<MCTSNode> node) {
    for (auto path_node: current_path_) {
        if (path_node == node) {
            return true;
        }
    }
    return false;
}

void rigid_block::MCTS_Graphviz::save_tree(std::string filename) {
    auto fout = std::ofstream(filename);
    visited.clear();
    fout << "digraph {\nrankdir=\"LR\";\n";
    save_tree_nodes(fout);
    save_tree_edges(root_, fout);
    fout << "}";
    fout.close();
}

void rigid_block::MCTS_Graphviz::save_tree_edges(std::shared_ptr<MCTSNode> node, std::ofstream &fout) {
    if (node == nullptr || visited[node]) {
        return;
    }
    visited[node] = true;

    std::string parent_node_id = std::to_string(node->ind_);
    std::string parent_label = node->label_;

    for (int action_id = 0; action_id < n_action_; action_id++) {
        if (node->children_[action_id] != nullptr) {
            auto child_node = node->children_[action_id];
            std::string child_label = child_node->label_;
            std::string child_node_id = std::to_string(child_node->ind_);

            double ub = node->ub(action_id);
            auto ub_str = float_to_string(ub, 2);

            std::string q_str = float_to_string(node->q_value_[action_id], 2);
            std::string n_str = std::to_string((int) (node->n_visit_[action_id]));
            std::string label_str = "ub = " + ub_str + ", q = " + q_str + ", n = " + n_str;

            fout << parent_node_id << "-> " << child_node_id << "[label = \"" << label_str << "\"";
            if (check_on_path(child_node) && check_on_path(node)) {
                fout << ", color = \"red\"";
            }
            fout << "];" << std::endl;
            save_tree_edges(child_node, fout);
            //if (check_on_path(child_node) && check_on_path(node)) {
            //}
        }
    }
}

void rigid_block::MCTS_Graphviz::save_tree_nodes(std::ofstream &fout) {
    for (auto it = mapping_.begin(); it != mapping_.end(); ++it)
    {
        auto node = it->second;
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
