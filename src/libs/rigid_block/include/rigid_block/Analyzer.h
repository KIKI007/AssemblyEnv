//
// Created by 汪子琦 on 05.09.22.
//

#ifndef ROBO_CRAFT_CONTACTGRAPHFORCE_H
#define ROBO_CRAFT_CONTACTGRAPHFORCE_H
#include "Eigen/Dense"
#include "Eigen/Sparse"
#include <map>

using std::vector;

namespace rigid_block
{

    class ContactPoint{
    public:
        int partIDA;
        int partIDB;
        Eigen::Vector3d contact_point;
        Eigen::Vector3d contact_normal;
        Eigen::Vector3d contact_friction_t1;
        Eigen::Vector3d contact_friction_t2;
    };

    class Analyzer
    {
    public:

        // contact list
        // partIDA, partIDB, normal, contact point
        // id is -1 if the corresponding block is fixed
        std::vector<ContactPoint> contact_points_;

        //blocks' weight
        std::vector<double> mass_;

        //blocks' centroid
        std::vector<Eigen::Vector3d> centroid_;

        // number of part
        int n_part_;

        // friction coefficient
        double friction_mu_;

        //matrix include tension component or not
        bool with_tension_ = false;

    public:

        Eigen::SparseMatrix<double> equalibrium_mat_;

        Eigen::SparseMatrix<double> friction_mat_;

        Eigen::VectorXd equalibrium_gravity_;

    public:

        const double inf = 1E8;

    public:

        Analyzer(int nParts, bool with_tension): n_part_(nParts), with_tension_(with_tension)
        {
            mass_.resize(nParts, 0.0);
            centroid_.resize(nParts, Eigen::Vector3d::Zero());
        }

        Analyzer(const Analyzer &analyzer)
        {
            n_part_ = analyzer.n_part_;
            mass_ = analyzer.mass_;
            centroid_ = analyzer.centroid_;
            contact_points_ = analyzer.contact_points_;
            with_tension_ = analyzer.with_tension_;
        }

        //Get Basic Properties
    public:

        int n_contact(){return contact_points_.size();}

        int n_part(){return n_part_;}

        double mass(int index){return mass_.at(index);}

        Eigen::Vector3d centroid(int index){return centroid_.at(index);}

        ContactPoint contact(int index){return contact_points_.at(index);}

        void computeFrictionDir(const Eigen::Vector3d &n, Eigen::Vector3d &t1, Eigen::Vector3d &t2);

        // Update Equilibrium Problem
    public:

        void updateFrictionCeoff(double mu){friction_mu_ = mu;}

        void addContact(int partIDA, int partIDB, const Eigen::Vector3d &normal, const std::vector<Eigen::Vector3d> &normal_point);

        void updatePart(int partID, double mass, Eigen::Vector3d ct);

        //Compute Equilibrium Matrices
    public:

        //nanobind
        void compute() {
                computeEquilibriumMatrix();
                computeGravity();
                computeFrictionMatrix();
        }

        void computeEquilibriumMatrix();

        void computeFrictionMatrix();

        void computeGravity();

        int n_var() {
            return n_part() * 6 + n_contact() * fdim();
        }

        int n_con_eq() {
            return n_part() * 6;
        }

        int n_con_fr() {
            return n_contact() * 4;
        }

        inline int fdim() {
            if(with_tension_) return 4;
            else return 3;
        }

        //l2 normal coefficent
        Eigen::SparseMatrix<double> obj_ceoff();

        std::tuple<std::vector<int>, std::vector<double>> var_lobnd(const std::vector<int> &status);
        std::tuple<std::vector<int>, std::vector<double>> var_upbnd(const std::vector<int> &status);

    };
}


#endif  //ROBO_CRAFT_CONTACTGRAPHFORCE_H
