#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/eigen/dense.h>
#include <nanobind/stl/unique_ptr.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/eigen/sparse.h>
#include <nanobind/stl/vector.h>
#include <rigid_block/MCTS.h>

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

    nb::class_<rigid_block::MCTSNode>(m, "MCTSNode")
    .def(nb::init<bool, double,
        const std::vector<int> &,
        const std::vector<std::vector<int>> &,
        const std::vector<double> &,
        const std::vector<double> &,
        const std::vector<bool> &>())
    .def_rw("S", &rigid_block::MCTSNode::state_)
    .def_rw("Sa", &rigid_block::MCTSNode::child_states_)
    .def_rw("Pa", &rigid_block::MCTSNode::prior_)
    .def_rw("noise", &rigid_block::MCTSNode::noise_)
    .def_rw("Na", &rigid_block::MCTSNode::n_visit_)
    .def_rw("N", &rigid_block::MCTSNode::tot_visit_)
    .def_rw("reward", &rigid_block::MCTSNode::reward_ )
    .def_rw("Va", &rigid_block::MCTSNode::valid_action_)
    .def_rw("terminated", &rigid_block::MCTSNode::terminated_);

    nb::class_<rigid_block::MCTS>(m, "MCTS")
    .def(nb::init<int, double, double>())
    .def("root_node", &rigid_block::MCTS::root_node)
    .def("leaf_node", &rigid_block::MCTS::leaf_node)
    .def("leaf_act", &rigid_block::MCTS::path_endAction)
    .def("set_root", &rigid_block::MCTS::set_root)
    .def("set_root_noise", &rigid_block::MCTS::set_root_noise)
    .def("find_leaf", &rigid_block::MCTS::find_leaf)
    .def("update", &rigid_block::MCTS::backward_update)
    .def("expand", &rigid_block::MCTS::expand)
    .def("execute", &rigid_block::MCTS::execute);

}