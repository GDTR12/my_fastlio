#include <sophus/so3.hpp>
#include <Eigen/Core>

#define SO3Exp(x) Sophus::SO3d::exp((x))

#define SO3Log(X) Sophus::SO3d::log((X))

#define SO3Hat(x) Sophus::SO3d::hat((x))

#define SO3Jr(x) Sophus::SO3d::leftJacobian((-x))

#define SO3Jl(x) Sophus::SO3d::leftJacobian((x))

using V3T = Eigen::Vector3d;
using V4T = Eigen::Vector4d;
using QuaT = Eigen::Quaterniond;
using M3T = Eigen::Matrix3d;
using M2T = Eigen::Matrix2d;
using M4T = Eigen::Matrix4d;
using SO3d = Sophus::SO3d;

#define MP_EN 
#define MP_PROC_NUM 30
