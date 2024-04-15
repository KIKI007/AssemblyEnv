//
// Created by 汪子琦 on 04.09.22.
//

#include "rigid_block/Assembly.h"
#include "rigid_block/collision.h"
#include <MacTypes.h>

#include "util/PolyPolyBoolean.h"
#include "util/ConvexHull2D.h"

namespace rigid_block
{
    Eigen::Vector3d ContactFace::center()
    {
        Eigen::Vector3d ct(0, 0, 0);
        for(auto pt: points) ct += pt;
        if(points.empty()) return ct;
        else return ct / points.size();
    }

    void ContactFace::scale(double ratio)
    {
        auto ct = center();
        for(auto &pt :points)
        {
            pt = (pt - ct) * ratio + ct;
        }
    }

    std::tuple<Eigen::MatrixXd, Eigen::MatrixXi> ContactFace::toMesh(const std::vector<ContactFace> &contacts)
    {
        Eigen::MatrixXd V;
        Eigen::MatrixXi F;

        int nV = 0;
        int nF = 0;
        for (int id = 0; id < contacts.size(); id++) {
            auto contact = contacts[id];
            nV += contact.points.size();
            nF += contact.points.size() - 2;
        }

        V = Eigen::MatrixXd(nV, 3);
        F = Eigen::MatrixXi(nF, 3);

        int iV = 0;
        int iF = 0;

        for (int id = 0; id < contacts.size(); id++) {
            auto contact = contacts[id];
            int nV = contact.points.size();
            for (int jd = 0; jd < nV; jd++) {
                V.row(jd + iV) = contact.points[jd];
            }
            for (int jd = 2; jd < nV; jd++) {
                F(jd - 2 + iF, 0) = iV;
                F(jd - 2 + iF, 1) = iV + jd - 1;
                F(jd - 2 + iF, 2) = iV + jd;
            }
            iV += nV;
            iF += nV - 2;
        }
        return {V, F};
    }




    std::vector<ContactFace> Assembly::computeContacts(const std::vector<int> &subPartIDs, double scale) {
        std::vector<ContactFace> contacts;
        for (int id = 0; id < subPartIDs.size(); id++) {
            for (int jd = id + 1; jd < subPartIDs.size(); jd++) {
                int partIDA = subPartIDs[id];
                int partIDB = subPartIDs[jd];
                std::vector<ContactFace> list_contacts = computeContacts(blocks_[partIDA], blocks_[partIDB]);
                for (int kd = 0; kd < list_contacts.size(); kd++) {
                    list_contacts[kd].partIDA = id;
                    list_contacts[kd].partIDB = jd;
                    list_contacts[kd].scale(scale);
                }
                contacts.insert(contacts.end(), list_contacts.begin(), list_contacts.end());
            }
        }
        return contacts;
    }

    std::vector<ContactFace> Assembly::computeContacts(std::shared_ptr<Part> block1,
                                                   std::shared_ptr<Part> block2) {
        util::PolyPolyBoolean boolean;

        std::vector<Eigen::Vector3d> points;
        std::vector<Eigen::Vector3d> normals;

        std::vector<ContactFace> contacts;

        for (int f1_i = 0; f1_i < block1->F_.rows(); f1_i++) {
            Eigen::Vector3d n1_i = block1->normal(f1_i);
            Eigen::Vector3d p1_i = block1->center(f1_i);
            for (int f2_j = 0; f2_j < block2->F_.rows(); f2_j++) {
                Eigen::Vector3d n2_j = block2->normal(f2_j);
                Eigen::Vector3d p2_j = block2->center(f2_j);

                //if two faces have the opposite normal
                //and two faces on the same plane
                //compute the intersection area, or the contact area
                if ((n1_i + n2_j).norm() <= error_small_normal_ &&
                    abs((p2_j - p1_i).dot(n1_i)) <= error_small_distance_) {
                    util::PolyPolyBoolean::PolyVector3 poly1_i = block1->face(f1_i);
                    util::PolyPolyBoolean::PolyVector3 poly2_j = block2->face(f2_j);
                    util::PolyPolyBoolean::PolysVector3 outs;
                    boolean.computePolygonsIntersection(poly1_i, poly2_j, outs);

                    for (util::PolyPolyBoolean::PolyVector3 out: outs)
                    {
                        ContactFace face;
                        face.normal= n1_i;
                        for (int kd = 0; kd < out.size(); kd++)
                        {
                            //points.push_back(out[kd]);
                            face.points.push_back(out[kd]);
                        }
                        contacts.push_back(face);
                    }
                }
            }
        }

        simplifyContact(contacts);
        return contacts;
    }

    void Assembly::simplifyContact(std::vector<ContactFace> &contacts)
    {
        std::vector<Eigen::Vector3d> points;
        std::vector<Eigen::Vector3d> normals;
        for(auto& contact: contacts)
        {
            for(auto & pt: contact.points)
            {
                points.push_back(pt);
                normals.push_back(contact.normal);
            }
        }

        ConvexCluster cluster;

        std::vector<std::vector<Eigen::Vector3d>> hull_points;
        std::vector<Eigen::Vector3d> hull_normals;
        cluster.computeConvexHull(points, normals, hull_points, hull_normals);

        contacts.clear();
        for(int id = 0; id < hull_points.size(); id++) {
            ContactFace face;
            face.points = hull_points[id];
            face.normal = hull_normals[id];
            contacts.push_back(face);
        }
    }

    void Assembly::loadFromFile(const std::string &filename) {
        util::readOBJ readObj;
        readObj.loadFromFile(filename);
        blocks_.resize(readObj.objIDs_.size());
        for (int id = 0; id < readObj.Vs_.size(); id++)
        {
            std::shared_ptr<Part> block = std::make_shared<Part>();
            block->V_ = readObj.Vs_[id];
            block->F_ = readObj.Fs_[id];
            block->N_ = readObj.Ns_[id];
            block->partID_ = readObj.objIDs_[id];
            block->ground_ = false;
            blocks_[block->partID_] = block;
        }
    }

    std::shared_ptr<Part> Assembly::getPart(int partID) {
        if(partID >= 0 && partID < blocks_.size()) {
            return blocks_[partID];
        }
        return nullptr;
    }

    void Assembly::updateGroundBlocks(std::shared_ptr<Part> ground_plane, const std::string &option)
    {
        for(auto &block: blocks_) block->ground_ = false;

        if(option == "fix") {
            for (int id = 0; id < blocks_.size(); id++) {
                std::shared_ptr<Part> block = blocks_[id];
                std::vector<ContactFace> contacts;
                contacts = computeContacts(block, ground_plane);
                if (!contacts.empty()) {
                    block->ground_ = true;
                }
            }
        }
        else if(option == "add"){
            blocks_.push_back(ground_plane);
            ground_plane->partID_ = (int) blocks_.size() - 1;
            ground_plane->ground_ = true;
        }
    }

    std::shared_ptr<Part> Assembly::computeGroundPlane()
    {
        Eigen::Vector3d minCoord, maxCoord;
        for (int id = 0; id < blocks_.size(); id++)
            {
            for (int kd = 0; kd < 3; kd++) {
                Eigen::VectorXd col = blocks_[id]->V_.col(kd);

                if (id == 0) minCoord[kd] = col.minCoeff();
                else minCoord[kd] = std::min(col.minCoeff(), minCoord[kd]);

                if (id == 0) maxCoord[kd] = col.maxCoeff();
                else maxCoord[kd] = std::max(col.maxCoeff(), maxCoord[kd]);
            }
        }

        double height = minCoord[2];

        Eigen::Vector3d center = (minCoord + maxCoord) / 2;
        Eigen::Vector3d size = (maxCoord - minCoord) / 2;
        size *= 1.5; //scale the plane 1.5x to cover the structure's base
        minCoord = center - size;
        maxCoord = center + size;

        std::shared_ptr<Part> ground_plane_ = std::make_shared<Part>();
        ground_plane_->V_ = Eigen::MatrixXd(4, 3);
        ground_plane_->V_ << minCoord[0], minCoord[1], height,
                maxCoord[0], minCoord[1], height,
                maxCoord[0], maxCoord[1], height,
                minCoord[0], maxCoord[1], height;

        ground_plane_->F_ = Eigen::MatrixXi(2, 3);
        ground_plane_->F_ << 0, 1, 2,
                0, 2, 3;
        ground_plane_->N_ = Eigen::MatrixXd(4, 3);
        ground_plane_->N_ << 0, 1, 0,
                0, 1, 0,
                0, 1, 0,
                0, 1, 0;

        return ground_plane_;
    }

    std::unique_ptr<Analyzer> Assembly::createAnalyzer(const std::vector<ContactFace> &contacts, bool tension)
    {
        std::unique_ptr<Analyzer> analyzer = std::make_unique<Analyzer>(blocks_.size(), tension);
        analyzer->updateFrictionCeoff(friction_coeff_);

        std::vector<int> partIDs;
        double length = computeAvgDiagnalLength();

        for (int ipart = 0; ipart < blocks_.size(); ipart++) {
            partIDs.push_back(ipart);
            analyzer->updatePart(ipart, blocks_[ipart]->volume() / pow(length, 3.0) / 16, blocks_[ipart]->centroid());
        }

        for (int id = 0; id < contacts.size(); id++) {
            analyzer->addContact(contacts[id].partIDA, contacts[id].partIDB, contacts[id].normal, contacts[id].points);
        }

        analyzer->computeEquilibriumMatrix();
        analyzer->computeGravity();

        return analyzer;
    }

    bool Assembly::checkSelfCollision()
    {
        if(blocks_.empty()) return false;
        std::shared_ptr<Part> back_part = blocks_.back();
        for(int id = 0; id < (int)blocks_.size() - 1; id++)
        {
            Eigen::MatrixXd Vi = blocks_[id]->V_;
            Eigen::MatrixXd Vb = back_part->V_;
            Vi = (Vi.rowwise() - blocks_[id]->centroid().transpose()) * 0.99;
            Vi = Vi.rowwise() + blocks_[id]->centroid().transpose();

            Vb = (Vb.rowwise() - back_part->centroid().transpose()) * 0.99;
            Vb = Vb.rowwise() + back_part->centroid().transpose();
            if(checkCollision(Vi, Vb)) {
                return true;
            }
        }
        return false;
    }

    double Assembly::computeAvgDiagnalLength()
    {
        double length = 0;
        for(int partID = 0; partID < blocks_.size(); partID ++) {
            length += blocks_[partID]->computeDiagnalLength();
        }
        length /= blocks_.size();
        return length;

    }
}
