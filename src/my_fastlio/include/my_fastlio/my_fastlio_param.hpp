#pragma once
#include <Eigen/Dense>
#include <memory>
#include <pcl/make_shared.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <vector>
// #include "ikd_Tree.h"
#include "pcl/pcl_macros.h"
#include "voxel_map/voxel_map_util.hpp"
#include "common_define.hpp"

namespace my_fastlio
{
    
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

// using PointIKDT = ikdTree_PointType;

using PointVMap = voxelmap::pointWithCov;
using CloudVMap = std::vector<PointVMap>;
using CloudVmapPtr = pcl::shared_ptr<CloudVMap>;

struct ImuData
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
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
