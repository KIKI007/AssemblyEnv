//
// Created by 汪子琦 on 04.09.22.
//

#ifndef PART_H
#define PART_H
#include "Eigen/Dense"
#include <memory>
#include <vector>
namespace rigid_block {
    class Part {
    public:

        //vertices
        Eigen::MatrixXd V_;

        //faces
        Eigen::MatrixXi F_;

        //vertices' normal
        Eigen::MatrixXd N_;

        int partID_;

        //the block is fixed if ground is true
        bool ground_ = false;

    public:

        static std::shared_ptr<Part> create_cuboid(Eigen::Vector3d center, Eigen::Vector3d dimension);

        static std::shared_ptr<Part> create_polygon(const Eigen::MatrixXd &points, double depth);

        static void create_triangles(
            const Eigen::MatrixXd &points,
            Eigen::MatrixXd &V,
            Eigen::MatrixXi &F,
            Eigen::MatrixXi &E,
            std::string option = "a");

    public:

        double volume();

        Eigen::Vector3d centroid();

        //normal of the face #fid
        Eigen::Vector3d normal(int fid);

        //center of the face #fid
        Eigen::Vector3d center(int fid);

        std::vector<Eigen::Vector3d> face(int fid);

        double computeDiagnalLength();

        int nF(){return F_.rows();}

        Eigen::Vector3d color() {
            if(ground_) {
                return  Eigen::Vector3d(0, 0, 0);
            }
            else {
                return Eigen::Vector3d(1, 1, 1);
            }
        }
    };
}


#endif  //PART_H
