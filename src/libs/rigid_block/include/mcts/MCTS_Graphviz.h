//
// Created by ziqwang on 16.08.24.
//

#ifndef MCTS_GRAPHVIZ_H
#define MCTS_GRAPHVIZ_H
#include "MCTS.h"
namespace mcts{
class MCTS_Graphviz
{
public:
    std::shared_ptr<MCTSNode> root_;
    std::map<std::vector<uint32_t>, std::weak_ptr<MCTSNode>> mapping_;
    std::map<std::shared_ptr<MCTSNode>, bool> visited;
public:

    MCTS_Graphviz(const MCTS &tree);

public:
    std::string float_to_string(double val, int precision);

    void save_tree(std::string filename);

    void save_tree_edges(std::shared_ptr<MCTSNode> node, std::ofstream &fout);

    void save_tree_nodes(std::ofstream &fout);
};
}

#endif //MCTS_GRAPHVIZ_H
