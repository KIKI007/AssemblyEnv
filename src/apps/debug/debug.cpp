//
// Created by Ziqi Wang on 11.04.2024.
//

#include "rigid_block/Part.h"
#include <iostream>
int main(){
    Eigen::MatrixXd points(8, 2);
    rigid_block::Part::create_polygon(points, 10);
}