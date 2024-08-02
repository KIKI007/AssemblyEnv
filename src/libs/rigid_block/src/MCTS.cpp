//
// Created by ziqwang on 10.07.24.
//

#include "rigid_block/MCTS.h"

rigid_block::MCTSNode::MCTSNode(
    int node_id,
    bool terminated,
    double reward,
    const std::vector<int> &state,
    const std::vector<double> &prior,
    const std::vector<double> &noise,
    const std::vector<bool> &valid)
{
    node_id_ = node_id;
    terminated_ = terminated;
    reward_ = reward;

    state_ = state;
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
    node_id_ = node.node_id_;
    cpuct_ = node.cpuct_;
    eps_ = node.eps_;

    //set by user
    terminated_ = node.terminated_;
    reward_ = node.reward_;
    state_ = node.state_;
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


std::string rigid_block::MCTSNode::node_label() {
    std::string label = "";
    int dim = std::sqrt(state_.size());
    for(int state_id = 0; state_id < state_.size(); state_id++) {
        int state = state_[state_id];
        label += std::to_string(state);
        if((state_id + 1) % dim == 0) {
            label += '\n';
        }
        else {
            label += " ";
        }
    }
    return label;
}

std::shared_ptr<rigid_block::MCTSNode> rigid_block::MCTSNode::add_child(int action_id, std::shared_ptr<MCTSNode> node) {
    std::shared_ptr<MCTSNode> new_node = std::make_shared<MCTSNode>(*node);
    children_[action_id] = new_node;
    return new_node;
}

std::shared_ptr<rigid_block::MCTSNode> rigid_block::MCTSNode::get_child(int action_id) {
    if(children_[action_id] != nullptr) {
        return children_[action_id];
    }
    return nullptr;
}

double rigid_block::MCTSNode::normalize_prior() {
    double sum_prior = 0;

    for(int action_id = 0; action_id < n_action_; action_id++) {
        sum_prior += prior_[action_id];
    }

    if(sum_prior > 0) {
        for(int action_id = 0; action_id < n_action_; action_id++) {
            prior_[action_id] /= sum_prior;
        }
    }
    return sum_prior;
}

void rigid_block::MCTSNode::update_prior()
{
    //remove prior with invalid moves
    for(int action_id = 0; action_id < n_action_; action_id++)
    {
        prior_[action_id] = prior_[action_id] * valid_action_[action_id];
    }

    //normalize the prior
    double sum_prior = normalize_prior();

    //if normalize fails
    if(sum_prior == 0) {
        for(int action_id = 0; action_id < n_action_; action_id++)
        {
            prior_[action_id] += valid_action_[action_id];
        }
        normalize_prior();
    }
}

int rigid_block::MCTSNode::best_action()
{
    double best_u = std::numeric_limits<double>::lowest();
    double best_act = 0;
    for (int action_id = 0; action_id < n_action_; action_id++)
    {
        if (valid_action_[action_id])
        {
            double u = ub(action_id);
            if (u > best_u)
            {
                best_u = u;
                best_act = action_id;
            }
        }
    }
    return best_act;
}

double rigid_block::MCTSNode::ub(int action_id)
{
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

std::string rigid_block::MCTS::float_to_string(double val, int precision) {
    std::stringstream stream;
    stream << std::fixed << std::setprecision(precision) << val;
    return stream.str();
}

/*
 * MCTS
 */
bool rigid_block::MCTS::check_on_path(std::shared_ptr<MCTSNode> node) {
    for(auto path_node : current_path_) {
        if(path_node == node) {
            return true;
        }
    }
    return false;
}

void rigid_block::MCTS::save_tree(std::string filename) {
    auto fout = std::ofstream(filename);
    fout << "digraph {\nrankdir=\"LR\";\n";
    save_tree(root_, fout);
    fout << "}";
    fout.close();
}

void rigid_block::MCTS::save_tree(std::shared_ptr<MCTSNode> node, std::ofstream &fout)
{
    std::string parent_node_id = std::to_string(node->node_id_);
    std::string parent_label = node->node_label();
    for(int action_id = 0; action_id < n_action_; action_id++)
        {
        if(node->children_[action_id] != nullptr)
        {
            auto child_node = node->children_[action_id];
            std::string child_label = child_node->node_label();
            std::string child_node_id = std::to_string(child_node->node_id_);

            double ub = node->ub(action_id);
            auto ub_str = float_to_string(ub, 2);

            std::string q_str = float_to_string(node->q_value_[action_id], 2);
            std::string n_str = std::to_string((int)(node->n_visit_[action_id]));
            std::string label_str = "ub = " + ub_str + ", q = " + q_str + ", n = " + n_str;

            save_tree(child_node, fout);
            fout << parent_node_id << "-> " << child_node_id << "[label = \"" << label_str << "\"";
            if(check_on_path(child_node) && check_on_path(node)) {
                fout << ", color = \"red\"";
            }
            fout << "];" <<  std::endl;
        }
    }
    std::string node_n = std::to_string(node->tot_visit_);
    std::string node_r = float_to_string(node->reward_, 0);
    std::string node_label = parent_label;
    fout << parent_node_id << " [label = \"" <<  node_label << "\"";
    if(node->terminated_) {
        if(node->reward_ > 0.5){
            fout << ", shape = \"diamond\"";
        }
        else{
            fout << ", shape = \"box\"";
        }
    }
    fout << "];" <<  std::endl;
}

std::shared_ptr<rigid_block::MCTSNode> rigid_block::MCTS::create_node(
    int node_id,
    bool terminate,
    double reward,
    const std::vector<int> &state,
    const std::vector<double> &prior,
    const std::vector<double> &noise,
    const std::vector<bool> &valid) const
{
    std::shared_ptr<MCTSNode> node
    = std::make_shared<MCTSNode>(node_id, terminate, reward, state, prior, noise, valid);
    node->update_prior();
    node->cpuct_ = cpuct_;
    return node;
}

bool rigid_block::MCTS::find_leaf(std::shared_ptr<MCTSNode> node) {
    current_path_.clear();
    current_path_action_.clear();
    while (node != nullptr)
    {
        current_path_.push_back(node);
        if (node->terminated_)
        {
            return false;
        } else {
            int act = node->best_action();
            current_path_action_.push_back(act);
            node = node->children_[act];
        }
    }
    return true;
}

void rigid_block::MCTS::expand(std::shared_ptr<MCTSNode> node) {
    path_endNode()->add_child(path_endAction(), node);
    current_path_.push_back(node);
}

void rigid_block::MCTS::backup(double v)
{
    // |path_node| = |path_action| + 1
    // the last node is a leaf node which does not need to be updated
    for(int id = 0; id < current_path_action_.size(); id++)
    {
        int action_id = current_path_action_[id];
        auto node = current_path_[id];
        node->q_value_[action_id] = (node->n_visit_[action_id] * node->q_value_[action_id] + v) / (node->n_visit_[action_id] + 1);
        node->n_visit_[action_id] += 1;
        node->tot_visit_++;
    }
}
