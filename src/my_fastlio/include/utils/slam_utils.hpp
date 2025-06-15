#pragma once

#include <ceres/ceres.h>
#include <Eigen/Core>

namespace slam_utils
{

using V3T = Eigen::Vector3d;
using QuaT = Eigen::Quaterniond;

struct ImuInterpData
{
    V3T w,a;
    V3T ba, bw;
    double time;
};


ImuInterpData ImuLinearInterp(ImuInterpData data0, ImuInterpData data1, double t);

} // namespace slam_utils


