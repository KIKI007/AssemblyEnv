#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/eigen/dense.h>
#include <nanobind/stl/unique_ptr.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/eigen/sparse.h>
#include <nanobind/stl/vector.h>

#include <mcts/MCTSNode.h>
#include <mcts/MCTS.h>
#include <mcts/MCTS_Graphviz.h>

#include <iostream>
namespace nb = nanobind;
#include "rigid_block/Assembly.h"

NB_MODULE(py_rigidblock, m)
{
    //nb::set_leak_warnings(false);

    nb::class_<rigid_block::ContactFace>(m, "ContactFace")
        .def(nb::init<>())
        .def_rw("part0", &rigid_block::ContactFace::partIDA)
        .def_rw("part1", &rigid_block::ContactFace::partIDB)
        .def_rw("points", &rigid_block::ContactFace::points)
        .def_rw("normal", &rigid_block::ContactFace::normal)
        .def_static("mesh", &rigid_block::ContactFace::toMesh);

    nb::class_<rigid_block::Assembly>(m, "Assembly")
    .def(nb::init<>())
    .def("from_file", &rigid_block::Assembly::loadFromFile)
    .def("part", &rigid_block::Assembly::getPart)
    .def("add_part", &rigid_block::Assembly::addPart)
    .def("n_part", [&](rigid_block::Assembly &t){return t.blocks_.size();})
    .def("contacts", nb::overload_cast<const std::vector<int> &, double>(&rigid_block::Assembly::computeContacts))
    .def("ground", &rigid_block::Assembly::computeGroundPlane)
    .def_rw("friction", &rigid_block::Assembly::friction_coeff_)
    .def("set_boundary", &rigid_block::Assembly::updateGroundBlocks)
    .def("analyzer", &rigid_block::Assembly::createAnalyzer)
    .def("analyzerGNN", &rigid_block::Assembly::createAnalyzerGNN)
    .def("self_collision", &rigid_block::Assembly::checkSelfCollision);

    nb::class_<rigid_block::Analyzer>(m, "Analyzer")
    .def(nb::init<int, bool>())
    .def("n_var", &rigid_block::Analyzer::n_var)
    .def("n_con_eq", &rigid_block::Analyzer::n_con_eq)
    .def("n_con_fr", &rigid_block::Analyzer::n_con_fr)
    .def("lobnd", &rigid_block::Analyzer::var_lobnd,  nb::rv_policy::take_ownership)
    .def("upbnd", &rigid_block::Analyzer::var_upbnd,  nb::rv_policy::take_ownership)
    .def_rw("matEq", &rigid_block::Analyzer::equalibrium_mat_)
    .def_rw("vecG", &rigid_block::Analyzer::equalibrium_gravity_)
    .def_rw("matFr", &rigid_block::Analyzer::friction_mat_)
    .def_rw("friction", &rigid_block::Analyzer::friction_mu_)
    .def("fdim", &rigid_block::Analyzer::fdim)
    .def("obj_ceoff", &rigid_block::Analyzer::obj_ceoff, nb::rv_policy::take_ownership)
    .def("gnn", &rigid_block::Analyzer::computeGNNRep)
    .def("compute", &rigid_block::Analyzer::compute)
    .def("sample", &rigid_block::Analyzer::sample_disassembly_directions);

    nb::class_<rigid_block::Part>(m, "Part")
    .def(nb::init<>())
    .def_rw("V", &rigid_block::Part::V_)
    .def_rw("F", &rigid_block::Part::F_)
    .def_rw("fixed", &rigid_block::Part::ground_)
    .def_rw("ind", &rigid_block::Part::partID_)
    .def_static("cuboid", &rigid_block::Part::create_cuboid)
    .def_static("polygon", &rigid_block::Part::create_polygon)
    .def_static("mesh", &rigid_block::Part::create_mesh)
    .def("face_center", &rigid_block::Part::center)
    .def("centroid", &rigid_block::Part::centroid)
    .def("volume", &rigid_block::Part::volume)
    .def("ee", &rigid_block::Part::eeAnchor);

    nb::class_<mcts::MCTSNode>(m, "MCTSNode")
    .def(nb::init<bool, double, double,
        const std::vector<int> &,
        const std::vector<double> &,
        const std::vector<double> &,
        const std::vector<int> &>())
    .def_rw("label", &mcts::MCTSNode::label_)
    .def("s", &mcts::MCTSNode::s)
    .def("sa", &mcts::MCTSNode::sa)
    .def("prior", &mcts::MCTSNode::prior)
    .def("noise", &mcts::MCTSNode::noise)
    .def("na", &mcts::MCTSNode::na)
    .def("n", &mcts::MCTSNode::N)
    .def("v", &mcts::MCTSNode::v)
    .def("add_child", &mcts::MCTSNode::add_child)
    .def("update_qa", &mcts::MCTSNode::update_qa)
    .def("update_na", &mcts::MCTSNode::update_na)
    .def("update_noise", &mcts::MCTSNode::update_noise)
    .def_rw("reward", &mcts::MCTSNode::reward_ )
    .def_rw("terminated", &mcts::MCTSNode::terminated_);

    nb::class_<mcts::MCTS>(m, "MCTS")
    .def(nb::init<>())
    .def("new_node", &mcts::MCTS::new_node)
    .def("add_child", &mcts::MCTS::add_child)
    .def("current_root", &mcts::MCTS::current_root)
    .def("set_root", &mcts::MCTS::set_root)
    .def("find_leaf", &mcts::MCTS::find_leaf)
    .def("update", &mcts::MCTS::update)
    .def("save", &mcts::MCTS::save)
    .def("reload", &mcts::MCTS::reload)
    .def("execute", &mcts::MCTS::execute);

    nb::class_<mcts::MCTS_Graphviz>(m, "MCTS_Graphviz")
    .def(nb::init<const mcts::MCTS &>())
    .def("save_tree", &mcts::MCTS_Graphviz::save_tree);
}
