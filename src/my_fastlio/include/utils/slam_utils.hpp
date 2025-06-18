#pragma once

#include <Eigen/Core>
#include <Eigen/src/Geometry/Quaternion.h>
#include <chrono>

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

class TimerHelper {
public:
    // 开始计时
    static std::chrono::high_resolution_clock::time_point start() {
        return std::chrono::high_resolution_clock::now();
    }
    
    // 结束计时并返回耗时(毫秒)
    static double end(const std::chrono::high_resolution_clock::time_point& start_time) {
        auto end_time = std::chrono::high_resolution_clock::now();
        return double(std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count()) / 1000000.0f;
    }
};


ImuInterpData ImuLinearInterp(ImuInterpData data0, ImuInterpData data1, double t);

} // namespace slam_utils


