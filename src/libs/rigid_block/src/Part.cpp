//
// Created by 汪子琦 on 04.09.22.
//
#include "util/ConvexCluster.h"
#include "rigid_block/Part.h"

#include "iostream"
#include <algorithm>
#include <unistd.h>
#include <triangle.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include "util/PolyPolyBoolean.h"

namespace rigid_block {

    std::shared_ptr<Part> Part::create_polygon(Eigen::MatrixXd& points, double depth)
    {
        Eigen::MatrixXd V;
        Eigen::MatrixXi F, E;
        create_triangles(points, V, F, E, "pzQ");
        int nV = V.rows();
        int nP = points.rows();

        Eigen::MatrixXd vertices(nV * 2 + nP * 2, 3);
        Eigen::MatrixXi faces(F.rows() * 2 + nP * 2, 3);
        vertices.setZero();
        faces.setZero();
        for(int id = 0; id < nV; id++)
        {
            double y = V(id, 0);
            double z = V(id, 1);
            double x = depth;
            vertices.row(id) = Eigen::RowVector3d(0, y, z);
            vertices.row(id + nV) = Eigen::RowVector3d(x, y, z);
        }
        for(int id = 0; id < points.rows(); id++)
        {
            double y = points(id, 0);
            double z = points(id, 1);
            double x = depth;
            vertices.row(id + nV * 2) = Eigen::RowVector3d(0, y, z);
            vertices.row(id + points.rows() + nV * 2) = Eigen::RowVector3d(x, y, z);
        }

        //bottom/top face
        for(int id = 0; id < F.rows(); id++)
        {
            int a =  F(id, 0);
            int b =  F(id, 1);
            int c =  F(id, 2);
            faces.row(id * 2) = Eigen::RowVector3i(a, c, b);
            faces.row(id * 2 + 1) = Eigen::RowVector3i(a + nV, b + nV, c + nV);
        }

        for(int id = 0; id < nP; id++)
        {
            int a = 2 * nV + id;
            int b = 2 * nV + (id + 1) % nP;
            int c = a + nP;
            int d = b + nP;
            faces.row(id * 2 + F.rows() * 2) = Eigen::RowVector3i(a, b, c);
            faces.row(id * 2 + F.rows() * 2 + 1) = Eigen::RowVector3i(b, d, c);
        }

        std::shared_ptr<Part> part = std::make_shared<Part>();
        part->V_ = vertices;
        part->F_ = faces;
        part->partID_ = -1;
        part->ground_ = false;

        return part;
    }

    void Part::create_triangles(Eigen::MatrixXd& points, Eigen::MatrixXd& V, Eigen::MatrixXi& F, Eigen::MatrixXi& E, std::string option)
    {
        //boundary represented by points can be non-convex
        //need triangulation
        util::PolyPolyBoolean poly;
        poly.cleanPath2(points);

        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> P(points.rows(), 2);
        for(int id = 0; id < points.rows(); id++) P.row(id) = points.row(id);
        int N = points.rows();

        triangulateio in, out;

        out.pointlist = NULL;
        out.trianglelist = NULL;
        out.segmentlist = NULL;
        out.segmentmarkerlist = NULL;
        out.pointmarkerlist = NULL;

        //points
        in.numberofpointattributes = 0;
        in.pointattributelist = NULL;
        in.numberofpoints = N;
        in.pointlist = P.data();

        //triangle
        in.trianglelist = NULL;
        in.numberoftriangles = 0;
        in.numberoftriangleattributes = 0;
        in.triangleattributelist = NULL;
        std::vector<int> pointmarkerlist;   pointmarkerlist.resize(N * 2, 1);
        in.pointmarkerlist = pointmarkerlist.data();

        //holes
        in.numberofholes = 0;
        in.numberofregions = 0;
        in.holelist = NULL;

        //segment
        Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> segment(N, 2);
        for(int id = 0; id < N; id ++)
        {
            segment(id, 0) = id;
            segment(id, 1) = (id + 1) % N;
        }

        in.numberofsegments = N;
        in.segmentlist = segment.data();
        std::vector<int> segmentmarkerlist; segmentmarkerlist.resize(N, 1);
        in.segmentmarkerlist = segmentmarkerlist.data();

        size_t len = strlen(option.c_str())+1;
        char *str = new char [len]; // allocate for string and ending \0
        strcpy(str, option.c_str());
        triangulate(str, &in, &out, 0);

        V = Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, 2, Eigen::RowMajor>>(out.pointlist, out.numberofpoints, 2);
        E = Eigen::Map<Eigen::Matrix<int, Eigen::Dynamic, 2, Eigen::RowMajor>>(out.segmentlist, out.numberofsegments, 2);
        F = Eigen::Map<Eigen::Matrix<int, Eigen::Dynamic, 3, Eigen::RowMajor>>(out.trianglelist, out.numberoftriangles, 3);
        // Cleanup in
        //free(in.pointlist);
        //free(in.pointmarkerlist);
        //free(in.segmentlist);
        //free(in.segmentmarkerlist);
        if(in.holelist) free(in.holelist);
        // Cleanup out
        free(out.pointlist);
        free(out.trianglelist);
        free(out.segmentlist);
        free(out.segmentmarkerlist);
        free(out.pointmarkerlist);
    }

    std::shared_ptr<Part> Part::create_cuboid(Eigen::Vector3d center, Eigen::Vector3d dimensions) {
        Eigen::MatrixXd vertices(8, 3); // 8 vertices, 3 coordinates each

        double halfX = dimensions.x() / 2.0f;
        double halfY = dimensions.y() / 2.0f;
        double halfZ = dimensions.z() / 2.0f;

        // Define vertices relative to center
        vertices << center.x() - halfX, center.y() - halfY, center.z() - halfZ, // 0
                    center.x() + halfX, center.y() - halfY, center.z() - halfZ, // 1
                    center.x() + halfX, center.y() + halfY, center.z() - halfZ, // 2
                    center.x() - halfX, center.y() + halfY, center.z() - halfZ, // 3
                    center.x() - halfX, center.y() - halfY, center.z() + halfZ, // 4
                    center.x() + halfX, center.y() - halfY, center.z() + halfZ, // 5
                    center.x() + halfX, center.y() + halfY, center.z() + halfZ, // 6
                    center.x() - halfX, center.y() + halfY, center.z() + halfZ; // 7

        Eigen::MatrixXi faces(12, 3); // 12 faces, 3 vertices each

        // Define faces using vertex indices
        faces << 0, 1, 2, // Front
                 0, 2, 3,
                 4, 5, 1, // Back
                 4, 1, 0,
                 7, 6, 5, // Top
                 7, 5, 4,
                 3, 2, 6, // Bottom
                 3, 6, 7,
                 0, 3, 7, // Left
                 0, 7, 4,
                 1, 5, 6, // Right
                 1, 6, 2;
        for(int ir = 0; ir < faces.rows(); ir++) std::swap(faces(ir, 1), faces(ir, 2));

        std::shared_ptr<Part> part = std::make_shared<Part>();
        part->V_ = vertices;
        part->F_ = faces;
        part->partID_ = -1;
        part->ground_ = false;

        return part;
    }

    double Part::volume() {
        double volume = 0;

        // Accumulate volume value for each triangle
        for (size_t i = 0; i < F_.rows(); i++)
        {
            Eigen::Vector3d v0, v1, v2;
            v0 = V_.row(F_(i, 0));
            v1 = V_.row(F_(i, 1));
            v2 = V_.row(F_(i, 2));

            Eigen::Vector3d crossVec = -1.0f * (v2 - v0).cross(v1 - v0);
            double dotP = v0.dot(crossVec);
            volume += dotP;
        }

        volume = volume / 6.0;
        return volume;
    }
    Eigen::Vector3d Part::centroid()
    {
        Eigen::Vector3d centroid = Eigen::Vector3d(0, 0, 0);

        // Save the 3 major axes
        Eigen::Vector3d axes[3];
        axes[0] = Eigen::Vector3d(1, 0, 0);
        axes[1] = Eigen::Vector3d(0, 1, 0);
        axes[2] = Eigen::Vector3d(0, 0, 1);

        // Accumulate centroid value for each major axes
        for (int i = 0; i < 3; i++)
        {
            Eigen::Vector3d axis = axes[i];

            for (size_t j = 0; j < F_.rows(); j++)
            {
                Eigen::Vector3d v2, v1, v0;
                v2 = V_.row(F_(j, 2));
                v1 = V_.row(F_(j, 1));
                v0 = V_.row(F_(j, 0));

                Eigen::Vector3d crossVec = -1.0f * (v2 - v0).cross(v1 - v0);

                centroid[i] += (1 / 24.0f) * (crossVec.dot(axis)) *
                               (pow((v0 + v1).dot(axis), 2) +
                                pow((v1 + v2).dot(axis), 2) +
                                pow((v2 + v0).dot(axis), 2));
            }
        }

        // Compute volume and centroid
        double v = volume();

        if(v > 1E-6){
            centroid = centroid / (2.0 * v);
        }
        else{
            centroid = Eigen::Vector3d(0, 0, 0);
        }

        return centroid;
    }

    Eigen::Vector3d Part::normal(int fid) {
        if(fid >= 0 && fid < F_.rows()){
            Eigen::Vector3d v0 = V_.row(F_(fid, 0));
            Eigen::Vector3d v1 = V_.row(F_(fid, 1));
            Eigen::Vector3d v2 = V_.row(F_(fid, 2));
            Eigen::Vector3d normal = ((v1 - v0).cross(v2 - v0));
            return normal.normalized();
        }
        return Eigen::Vector3d(0, 0, 0);
    }

    Eigen::Vector3d Part::center(int fid) {
        if(fid >= 0 && fid < F_.rows()){
            Eigen::Vector3d v0 = V_.row(F_(fid, 0));
            Eigen::Vector3d v1 = V_.row(F_(fid, 1));
            Eigen::Vector3d v2 = V_.row(F_(fid, 2));
            return (v0 + v1 + v2) / 3;
        }
        return Eigen::Vector3d(0, 0, 0);
    }

    std::vector<Eigen::Vector3d> Part::face(int fid){
        if(fid >= 0 && fid < F_.rows()){
            Eigen::Vector3d v0 = V_.row(F_(fid, 0));
            Eigen::Vector3d v1 = V_.row(F_(fid, 1));
            Eigen::Vector3d v2 = V_.row(F_(fid, 2));
            return {v0, v1, v2};
        }
        return {};
    }

    double Part::computeDiagnalLength() {
        Eigen::Vector3d minCoord(std::numeric_limits<double>::max(), std::numeric_limits<double>::max(), std::numeric_limits<double>::max());
        Eigen::Vector3d maxCoord(std::numeric_limits<double>::lowest(), std::numeric_limits<double>::lowest(), std::numeric_limits<double>::lowest());
        for(int ir = 0; ir < V_.rows(); ir++) {
            Eigen::Vector3d pt = V_.row(ir);
            minCoord = minCoord.cwiseMin(pt);
            maxCoord = maxCoord.cwiseMax(pt);
        }
        return (maxCoord - minCoord).norm() / 2;
    }
}

