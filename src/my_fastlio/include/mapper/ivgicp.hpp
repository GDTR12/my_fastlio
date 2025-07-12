#pragma once

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <Eigen/Core> 
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <unordered_map>
#include <vector>
#include <pcl/kdtree/kdtree_flann.h>
#include "ikd_Tree.h"

namespace gicp {

typedef pcl::PointXYZ Point;
// template<typename Point>
class VGICP 
{
public:

    struct GICPVoxelIndex : public Eigen::Vector3i{
    public:
        GICPVoxelIndex(const Eigen::Vector3i& other) : Eigen::Vector3i(other) {}

        GICPVoxelIndex(){*this = Eigen::Vector3i::Zero();}

        bool operator==(const GICPVoxelIndex& other) const {
            return this->x() == other.x() && this->y() == other.y() && this->z() == other.z();
        }

        GICPVoxelIndex& operator=(const Eigen::Vector3i& other){
            if (this != &other){
                this->x() = other.x();
                this->y() = other.y();
                this->z() = other.z();
            }
            return *this;
        }

        Eigen::Vector3d getCenter(float voxel_size) const {
            return (Eigen::Vector3d(this->x(), this->y(), this->z()) + 0.5 * Eigen::Vector3d::Ones()) * voxel_size;
        }

    };

    struct VoxelIndexHash {
        std::size_t operator()(const GICPVoxelIndex& idx) const {
            return ((std::hash<int>()(idx.x()) ^ 
                    (std::hash<int>()(idx.y()) << 1)) >> 1) ^ 
                    (std::hash<int>()(idx.z()) << 1);
        }
    };

    typedef GICPVoxelIndex VoxelIndex;

    typedef pcl::PointCloud<Point> Cloud;

    typedef typename pcl::PointCloud<Point>::Ptr CloudPtr;

    struct Voxel{
        Eigen::Vector3d mean = Eigen::Vector3d::Zero();
        Eigen::Matrix3d cov = Eigen::Matrix3d::Zero();
        Eigen::Vector3d sum = Eigen::Vector3d::Zero();
        Eigen::Matrix3d outer = Eigen::Matrix3d::Zero();
        size_t N = 0;
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    };

    struct SinglePointCov{
        Eigen::Vector3d point = Eigen::Vector3d::Zero();
        Eigen::Matrix3d cov = Eigen::Matrix3d::Zero();
        size_t N = 0;
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    };

    struct SingleAssociation: public SinglePointCov{
        VoxelIndex voxel_idx;
        bool associated = true;
    };

    VGICP();

    ~VGICP();


    VoxelIndex point2Index(const Eigen::Vector3d& point, float voxel_size){
        return Eigen::Vector3i((point / voxel_size).array().floor().cast<int>());
    }

    VoxelIndex point2Index(const Point& point, float voxel_size){
        return point2Index(Eigen::Vector3d(point.x, point.y, point.z), voxel_size);
    }

    void setSourceCloud(CloudPtr cloud, bool voxel_downsample=false, float voxel_downsample_size=1.0);

    void setSourceTransformation(const Eigen::Matrix4d& pose, bool update_association = true);

    // void computeErrorAndCovariance(Eigen::Matrix<double, Eigen::Dynamic, 1>& error, Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>& Rinv);
    bool empty(){return voxel_map.empty();}

    void pushSourceIntoMap();

    void removeOutRadiusVoxel(const Point& center);

    const std::vector<SingleAssociation>& getAssociation(){return association;}

    std::unordered_map<VoxelIndex, Voxel, VoxelIndexHash>&getVoxeles(){return voxel_map;}

    float getVoxelSize(){return voxel_size;}

    const Voxel& operator[](const VoxelIndex& idx){
        if (voxel_map.find(idx) == voxel_map.end()){
            throw std::runtime_error("voxel doesn't exist!");
        }
        return voxel_map[idx];
    }
    
private:

    void computeSourceCloudCovariance();

    Voxel& getAssociatedVoxel(const Eigen::Vector3d& point);

    std::unordered_map<VoxelIndex, Voxel, VoxelIndexHash> voxel_map;

    Eigen::Matrix4d src_pose = Eigen::Matrix4d::Identity();

    CloudPtr src_cloud;

    pcl::KdTreeFLANN<Point> src_kdtree;

    // KD_TREE<ikdTree_PointType>::PointVector cloud_map_center;
    KD_TREE<ikdTree_PointType> ikdtree_map_center;

    std::vector<SinglePointCov> points_cov;
    std::vector<SingleAssociation> association;

    float localmap_radius = 200;

    int cov_nearist_k = 10;

    int omp_num_threads = 8;

    float voxel_size = 0.5;

    float search_radius = 1.5;

    // float min_point_density = 10 * voxel_size * voxel_size * voxel_size;
    float min_point_density = 5;

};


}

