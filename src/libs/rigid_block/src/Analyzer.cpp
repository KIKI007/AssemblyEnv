//
// Created by 汪子琦 on 05.09.22.
//

#include "rigid_block/Analyzer.h"
#include <iostream>
#include <random>
namespace rigid_block
{
    void Analyzer::addContact(int partIDA,
                              int partIDB,
                              const Eigen::Vector3d& normal,
                              const std::vector<Eigen::Vector3d>& normal_point)
    {
        ContactPoint contact_point;
        Eigen::Vector3d t1, t2;
        computeFrictionDir(normal, t1, t2);
        for (int id = 0; id < normal_point.size(); id++)
        {
            contact_point.contact_point = normal_point[id];
            contact_point.contact_normal = normal;
            contact_point.contact_friction_t1 = t1;
            contact_point.contact_friction_t2 = t2;
            contact_point.partIDA = partIDA;
            contact_point.partIDB = partIDB;
            contact_points_.push_back(contact_point);
        }

        contact_normals_[partIDA].push_back({partIDB, normal});
        contact_normals_[partIDB].push_back({partIDA, -normal});
    }

    void Analyzer::updatePart(int partID, double mass, Eigen::Vector3d ct)
    {
        if (0 <= partID && partID < n_part())
        {
            mass_[partID] = mass;
            centroid_[partID] = ct;
        }
    }

    std::tuple<Eigen::VectorXi, Eigen::VectorXi, Eigen::MatrixXd> Analyzer::computeGNNRep()
    {
        int nC = contact_points_.size();
        Eigen::VectorXi edge_sta( nC * 2), edge_end(nC * 2);
        Eigen::MatrixXd edge_attr(nC * 2, 9);

        for (int ic = 0; ic < contact_points_.size(); ic++)
        {
            int partIDA = contact_points_[ic].partIDA;
            int partIDB = contact_points_[ic].partIDB;

            Eigen::Vector3d n = contact_points_[ic].contact_normal;
            Eigen::Vector3d r = contact_points_[ic].contact_point;
            Eigen::Vector3d ctA = centroid(partIDA);
            Eigen::Vector3d ctB = centroid(partIDB);

            Eigen::VectorXd dataAB(9), dataBA(9);
            dataAB.segment(0, 3) = r - ctA;
            dataAB.segment(3, 3) = ctB - r;
            dataAB.segment(6, 3) = n;
            edge_sta(ic * 2) = partIDA;
            edge_end(ic * 2) = partIDB;
            edge_attr.row(ic * 2) = dataAB;

            dataBA.segment(0, 3) = r - ctB;
            dataBA.segment(3, 3) = ctA - r;
            dataBA.segment(6, 3) = -n;
            edge_sta(ic * 2 + 1) = partIDB;
            edge_end(ic * 2 + 1) = partIDA;
            edge_attr.row(ic * 2 + 1) = dataBA;
        }

        return {edge_sta, edge_end, edge_attr};
    }

    void Analyzer::computeEquilibriumMatrix()
    {
        std::vector<Eigen::Triplet<double>> equalibrium_mat_triplets_;

        for (int ic = 0; ic < contact_points_.size(); ic++)
        {
            int partIDA = contact_points_[ic].partIDA;
            int partIDB = contact_points_[ic].partIDB;

            Eigen::Vector3d n = contact_points_[ic].contact_normal;
            Eigen::Vector3d r = contact_points_[ic].contact_point;
            Eigen::Vector3d t1 = contact_points_[ic].contact_friction_t1;
            Eigen::Vector3d t2 = contact_points_[ic].contact_friction_t2;

            std::vector<int> partIDs = {partIDA, partIDB};
            std::vector<Eigen::Vector3d> drts;

            //every vector in drts has a corresponding force variable
            if (with_tension_) drts = {n, -n, t1, t2};
            else drts = {n, t1, t2};

            for (int iP = 0; iP < 2; iP++)
            {
                int sign = (iP == 0 ? -1 : 1);
                int partID = partIDs[iP];

                //force
                for (int jdim = 0; jdim < 3; jdim++)
                {
                    int row = partID * 6 + jdim;
                    for (int kdir = 0; kdir < drts.size(); kdir++)
                    {
                        int col = drts.size() * ic + kdir;
                        Eigen::Vector3d fdir = drts[kdir];
                        equalibrium_mat_triplets_.push_back(Eigen::Triplet(row, col, fdir[jdim] * sign));
                    }
                }

                //torque
                for (int jdim = 0; jdim < 3; jdim++)
                {
                    int row = partID * 6 + jdim + 3;
                    for (int kdir = 0; kdir < drts.size(); kdir++)
                    {
                        Eigen::Vector3d m = r.cross(drts[kdir]);
                        int col = drts.size() * ic + kdir;
                        equalibrium_mat_triplets_.push_back(Eigen::Triplet(row, col, m[jdim] * sign));
                    }
                }
            }
        }

        for (int pi = 0; pi < n_part(); pi++)
        {
            for (int jdim = 0; jdim < 6; jdim++)
            {
                int row = pi * 6 + jdim;
                int col = n_contact() * fdim() + pi * 6 + jdim;
                equalibrium_mat_triplets_.push_back({row, col, 1});
            }
        }

        equalibrium_mat_ = Eigen::SparseMatrix<double>(n_con_eq(), n_var());
        equalibrium_mat_.setFromTriplets(equalibrium_mat_triplets_.begin(), equalibrium_mat_triplets_.end());
    }

    void Analyzer::computeFrictionMatrix()
    {
        std::vector<Eigen::Triplet<double>> friction_triplets_;

        for (int ic = 0; ic < contact_points_.size(); ic++)
        {
            int fr_t1 = ic * fdim() + fdim() - 2;
            int fr_t2 = ic * fdim() + fdim() - 1;
            int fn = ic * fdim();
            //|fr| <= fn * coeff

            //fr - fn * coeff <= 0
            friction_triplets_.push_back({ic * 4, fr_t1, 1});
            friction_triplets_.push_back({ic * 4, fn, -friction_mu_});

            //-fn * coeff - fr <= 0
            friction_triplets_.push_back({ic * 4 + 1, fr_t1, -1});
            friction_triplets_.push_back({ic * 4 + 1, fn, -friction_mu_});

            //fr - fn * coeff <= 0
            friction_triplets_.push_back({ic * 4 + 2, fr_t2, 1});
            friction_triplets_.push_back({ic * 4 + 2, fn, -friction_mu_});

            //-fn * coeff - fr <= 0
            friction_triplets_.push_back({ic * 4 + 3, fr_t2, -1});
            friction_triplets_.push_back({ic * 4 + 3, fn, -friction_mu_});
        }

        friction_mat_ = Eigen::SparseMatrix<double>(n_con_fr(), n_var());
        friction_mat_.setFromTriplets(friction_triplets_.begin(), friction_triplets_.end());
    }

    void Analyzer::computeGravity()
    {
        equalibrium_gravity_ = Eigen::VectorXd::Zero(n_part() * 6);
        for (int part_id = 0; part_id < n_part(); part_id++)
        {
            Eigen::Vector3d force(0, 0, -mass_[part_id]);
            Eigen::Vector3d r = centroid_[part_id];
            Eigen::Vector3d torque = r.cross(force);
            equalibrium_gravity_.segment(part_id * 6, 3) = force;
            equalibrium_gravity_.segment(part_id * 6 + 3, 3) = torque;
        }
    }

    Eigen::SparseMatrix<double> Analyzer::obj_ceoff()
    {
        std::vector<Eigen::Triplet<double>> triLists;
        if (with_tension_)
        {
            for (int id = 0; id < n_contact(); id++)
            {
                triLists.push_back({id * fdim() + 1, id * fdim() + 1, 1});
            }
        }
        else
        {
            for (int id = 0; id < n_contact(); id++)
            {
                triLists.push_back({id * fdim(), id * fdim(), 1});
            }
        }
        Eigen::SparseMatrix<double> mat(n_var(), n_var());
        mat.setFromTriplets(triLists.begin(), triLists.end());
        return mat;
    }

    //status
    //0, non-installed
    //1, installed
    //2, fixed
    std::tuple<std::vector<int>, std::vector<double>> Analyzer::var_lobnd(const std::vector<int>& status)
    {
        std::vector<int> vind;
        std::vector<double> lobnd;

        for (int id = 0; id < n_contact(); id++)
        {
            for (int jd = 0; jd < fdim() - 2; jd++)
            {
                vind.push_back(id * fdim() + jd);
                lobnd.push_back(0);
            }
        }

        for (int id = 0; id < status.size(); id++)
        {
            if (status[id] == 1)
            {
                for (int jd = 0; jd < 6; jd++)
                {
                    int fj = n_contact() * fdim() + id * 6 + jd;
                    lobnd.push_back(0);
                    vind.push_back(fj);
                }
            }
        }

        return {vind, lobnd};
    }

    //status
    //0, non-installed
    //1, installed
    //2, fixed
    std::tuple<std::vector<int>, std::vector<double>> Analyzer::var_upbnd(const std::vector<int>& status)
    {
        std::vector<int> vind;
        std::vector<double> upbnd;

        for (int id = 0; id < n_contact(); id++)
        {
            int partIDA = contact_points_[id].partIDA;
            int partIDB = contact_points_[id].partIDB;

            if(status[partIDA] == 0 || status[partIDB] == 0)
            {
                for (int jd = 0; jd < fdim() - 2; jd++)
                {
                    vind.push_back(id * fdim() + jd);
                    upbnd.push_back(0);
                }
            }
        }


        for (int id = 0; id < status.size(); id++)
        {
            if (status[id] == 1)
            {
                for (int jd = 0; jd < 6; jd++)
                {
                    int fj = n_contact() * fdim() + id * 6 + jd;
                    upbnd.push_back(0);
                    vind.push_back(fj);
                }
            }
        }

        return {vind, upbnd};
    }

    Eigen::MatrixXd Analyzer::sample_disassembly_directions(int partID, const std::vector<int> &status, int numSamples) {

        std::vector<Eigen::Vector3d> normals;
        for(int id = 0; id < contact_normals_[partID].size(); id++) {
            auto [partIDB, n] = contact_normals_[partID][id];
            if (status[partIDB] > 0) {
                normals.push_back(-n);
                //std::cout << (-n).transpose() << std::endl;
            }
        }

        // Generate random points inside the unit sphere
        std::random_device rd;
        std::mt19937 rng(rd());

        std::vector<Eigen::Vector3d> cand_drts, drts;
        std::uniform_real_distribution<double> dist(-1.0, 1.0);

        for (int i = 0; i < numSamples; ++i) {
            Eigen::Vector3d drt;
            drt = Eigen::Vector3d(dist(rng), dist(rng), dist(rng));
            if(drt.norm() > 1E-6) {
                drt = drt / drt.norm();
                cand_drts.push_back(drt);
            }
        }

        int N = 32;
        for(auto normal : normals) {
            Eigen::Vector3d xaxis = Eigen::Vector3d(0, 0, 1).cross(normal);
            if(xaxis.norm() < 1E-6) {
                xaxis = Eigen::Vector3d(1, 0, 0).cross(normal);
            }
            xaxis /= xaxis.norm();
            Eigen::Vector3d yaxis = normal.cross(xaxis);
            yaxis /= yaxis.norm();
            for(int j = 0; j < N; j++) {
                double angle = M_PI * 2 / N * (double)j;
                Eigen::Vector3d drt = xaxis * cos(angle) + yaxis * sin(angle);
                cand_drts.push_back(drt);
            }
        }

        for(auto drt : cand_drts)
        {
            bool valid = true;
            for(auto &n : normals) {
                if(drt.dot(n) < -1E-6) {
                    valid = false;
                    break;
                }
            }
            if(valid) {
                drts.push_back(drt);
            }
        }

        Eigen::MatrixXd result = Eigen::MatrixXd(drts.size(), 3);
        for(int id = 0; id < drts.size(); id++) {
            result.row(id) = drts[id];
        }
        return result;

    }

    void Analyzer::computeFrictionDir(const Eigen::Vector3d& n, Eigen::Vector3d& t1, Eigen::Vector3d& t2)
    {
        t1 = n.cross(Eigen::Vector3d(1, 0, 0));
        if (t1.norm() < 1E-4)
        {
            t1 = n.cross(Eigen::Vector3d(0, 1, 0));
        }
        t1.normalize();
        t2 = n.cross(t1);
        t2.normalize();
    }
}
