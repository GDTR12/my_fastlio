#pragma once
#include <Eigen/Dense>
#include <memory>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include "ikd_Tree.h"

namespace my_fastlio
{
    
using V3T = Eigen::Vector3d;
using V4T = Eigen::Vector4d;
using QuaT = Eigen::Quaterniond;
using M3T = Eigen::Matrix3d;
using M2T = Eigen::Matrix2d;
using M4T = Eigen::Matrix4d;
using PointXYZ = pcl::PointXYZ;
using PointXYZI = pcl::PointXYZ;
using PointXYZRGB = pcl::PointXYZRGB;
using CloudXYZ = pcl::PointCloud<PointXYZ>;
using CloudXYZI = pcl::PointCloud<PointXYZI>;
using CloudXYZRGB = pcl::PointCloud<PointXYZRGB>;
using CloudXYZPtr = pcl::PointCloud<PointXYZ>::Ptr;
using CloudXYZIPtr = pcl::PointCloud<PointXYZI>::Ptr;
using CloudXYZRGBPtr = pcl::PointCloud<PointXYZRGB>::Ptr;
using CloudXYZConstPtr = pcl::PointCloud<PointXYZ>::ConstPtr;
using CloudXYZIConstPtr = pcl::PointCloud<PointXYZI>::ConstPtr;
using CloudXYZRGBConstPtr = pcl::PointCloud<PointXYZRGB>::ConstPtr;

using PointT = pcl::PointXYZINormal;
using CloudT = pcl::PointCloud<PointT>;
using CloudPtr = pcl::PointCloud<PointT>::Ptr;
using CloudConstPtr = pcl::PointCloud<PointT>::ConstPtr;

using PointIKDT = ikdTree_PointType;


struct ImuData
{
    double time;
    V3T w; // angular velocity
    V3T a; // linear acceleration
};

struct LidarData
{
    double time = 0.0;
    CloudPtr cloud = pcl::make_shared<CloudT>();
};

// class MyFastLIOParam
// {
// private:
//     MyFastLIOParam(){}
// public:

//     static MyFastLIOParam& createInstance()
//     {
//         static MyFastLIOParam instance;
//         return instance;
//     }

//     ~MyFastLIOParam() = default;

//     std::string imu_topic;
//     std::string lidar_topic;
//     Eigen::Matrix3d R_ItoL;
//     Eigen::Vector3d t_ItoL;
    
    
//     Eigen::Matrix4d getTransformMatrix()
//     {
//         Eigen::Matrix4d T_ItoL = Eigen::Matrix4d::Identity();
//         T_ItoL.block<3, 3>(0, 0) = R_ItoL;
//         T_ItoL.block<3, 1>(0, 3) = t_ItoL;
//         return T_ItoL;
//     }
// };


} // namespace my_fastlio
