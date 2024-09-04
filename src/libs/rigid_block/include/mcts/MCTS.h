//
// Created by ziqwang on 16.08.24.
//

#ifndef MCTSTREE_H
#define MCTSTREE_H
#include <iostream>
#include <map>
#include <memory>
#include "MCTSNode.h"
namespace mcts {
    class MCTS
    {
    friend class MCTS_Graphviz;

    //tree search variables
    private:

        std::weak_ptr<MCTSNode> root_, current_root_;

        std::vector<std::shared_ptr<MCTSNode>> nodes_;

        std::map<std::vector<uint32_t>, std::weak_ptr<MCTSNode>> mapping_;

    public:

        std::vector<std::shared_ptr<MCTSNode>> stored_nodes;

        void save(const std::vector<std::shared_ptr<MCTSNode>> &nodes);

        void reload();

    public:

        void set_root(std::shared_ptr<MCTSNode> node);

        std::shared_ptr<MCTSNode> new_node(const MCTSNode &node);

        std::tuple<bool, std::vector<std::shared_ptr<MCTSNode>>, std::vector<int>> find_leaf();

        bool execute(int act);

        void update(double v,
                    double discount,
                    const std::vector<std::shared_ptr<MCTSNode>> &path,
                    const std::vector<int> &path_act);

        void add_child(std::shared_ptr<MCTSNode> parent,
                       std::shared_ptr<MCTSNode> child,
                       int action_id) {
            if(parent && child)
            {
                nodes_[parent->ind_]->add_child(action_id, nodes_[child->ind_]);
            }
        }

    public:

        std::shared_ptr<MCTSNode> root_node() {return root_.lock();}

        std::shared_ptr<MCTSNode> current_root() {
            return current_root_.lock();
        }

        std::shared_ptr<MCTSNode> child_node(std::shared_ptr<MCTSNode> node, int action_id);

    };
}
#endif //MCTSTREE_H
