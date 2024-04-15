//
// Created by 汪子琦 on 04.09.22.
//

#ifndef ROBO_CRAFT_BLOCKASSEMBLY_H
#define ROBO_CRAFT_BLOCKASSEMBLY_H

#include "Part.h"
#include "util/readOBJ.h"
#include "Analyzer.h"
#include <memory>
#include "util/ConvexCluster.h"

namespace rigid_block
{
    class ContactFace
    {
    public:
        int partIDA;
        int partIDB;
        std::vector<Eigen::Vector3d> points;
        Eigen::Vector3d normal;
        static std::tuple<Eigen::MatrixXd, Eigen::MatrixXi> toMesh(const std::vector<ContactFace> &contacts);

        void scale(double ratio);
        Eigen::Vector3d center();
    };

    class Assembly
    {
    public:

        std::vector<std::shared_ptr<Part> > blocks_;

    public:

        const double error_small_normal_ = 1E-4;

        const double error_small_distance_ = 1E-2;

        float friction_coeff_ = 0.5;

    public:

        Assembly()
        {
        }

        Assembly(std::vector<std::shared_ptr<Part> > blocks): blocks_(blocks) {
        }

    public:

        std::vector<ContactFace> computeContacts(std::shared_ptr<Part> block1, std::shared_ptr<Part> block2);

        void simplifyContact(std::vector<ContactFace> &contacts);

        double computeAvgDiagnalLength();

        //nanobind
    public:

        void loadFromFile(const std::string &filename);

        std::shared_ptr<Part> getPart(int partID);

        std::vector<ContactFace> computeContacts(const std::vector<int> &subPartIDs, double scale = 1.0);

        std::shared_ptr<Part> computeGroundPlane();

        void updateGroundBlocks(std::shared_ptr<Part> ground_plane, const std::string &option);

        void addPart(std::shared_ptr<Part> part){part->partID_ = blocks_.size(); blocks_.push_back(part);}

        std::unique_ptr<Analyzer> createAnalyzer(const std::vector<ContactFace> &contacts, bool tension);

        bool checkSelfCollision();

    };
}


#endif  //ROBO_CRAFT_BLOCKASSEMBLY_H
