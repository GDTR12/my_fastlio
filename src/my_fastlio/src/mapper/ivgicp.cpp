#include "mapper/ivgicp.hpp"
#include "ikd_Tree.h"
#include "pcl/impl/point_types.hpp"
#include "pcl/kdtree/kdtree_flann.h"
#include <list>
#include <stdexcept>
#include <pcl/filters/voxel_grid.h>
#include <omp.h>
#include <unordered_set>
#include <vector>

namespace gicp {


VGICP::VGICP()
{}


VGICP::~VGICP()
{}

void VGICP::setSourceCloud(CloudPtr input, bool voxel_downsample, float voxel_downsample_size)
{
    CloudPtr cloud = input;
    if (voxel_downsample){
        CloudPtr downcloud(new Cloud);
        std::unordered_map<VoxelIndex, Voxel, VoxelIndexHash> voxeles;
        for (int i = 0; i < cloud->size(); i++)
        {
            auto id = point2Index(cloud->at(i), voxel_downsample_size);
            Eigen::Vector3d point = cloud->at(i).getVector3fMap().cast<double>();
            voxeles[id].sum += point;
            voxeles[id].outer += point * point.transpose();
            voxeles[id].N += 1;
        }
        points_cov.resize(voxeles.size());
        downcloud->resize(voxeles.size());
        int i = 0;
        for (const auto& [id, voxel] : voxeles)
        {
            Point p;
            p.getVector3fMap() = (voxel.sum / voxel.N).cast<float>();
            downcloud->at(i) = p;
            points_cov[i].point = p.getVector3fMap().cast<double>();
            points_cov[i].N = voxel.N;
            if (voxel.N < 6){
                points_cov[i].cov.setZero();
            }else{
                points_cov[i].cov = (voxel.outer - voxel.sum * voxel.mean.transpose()) / float(voxel.N);
            }
            i++;
        }
        cloud = downcloud;
   }
    src_cloud = cloud;
    src_kdtree.setInputCloud(src_cloud);
    // computeSourceCloudCovariance();
}

// void VGICP::computeSourceCloudCovariance()
// {
//     if (src_cloud == nullptr){
//         throw  std::runtime_error("Source Cloud is null");
//     }
//     if (src_cloud->empty()){
//         throw  std::runtime_error("Source Cloud is empty");
//     }

//     points_cov.clear();
//     points_cov.resize(src_cloud->size());

//     #pragma omp parallel for num_threads(omp_num_threads) schedule(static, 8)
//     for (int i = 0; i < src_cloud->size(); i++)
//     {
//         std::vector<int> indices;
//         std::vector<float> dists;
//         src_kdtree.nearestKSearch(src_cloud->at(i), cov_nearist_k, indices, dists);

//         Eigen::Matrix<double, 3, Eigen::Dynamic> neighbors(3, indices.size());
//         for (int j = 0; j < indices.size(); j++){
//             neighbors.col(j) = src_cloud->at(indices[j]).getVector3fMap().template cast<double>();
//         }
//         // points_cov[i].point = neighbors.rowwise().mean().eval();
//         points_cov[i].point = src_cloud->at(i).getVector3fMap().cast<double>();
//         neighbors.colwise() -= neighbors.rowwise().mean().eval();
//         points_cov[i].cov = neighbors * neighbors.transpose() / neighbors.cols();
//         // points_cov[i].cov = Eigen::Matrix3d::Zero();
//     }
// }

void VGICP::setSourceTransformation(const Eigen::Matrix4d& pose, bool update_association)
{
    if (pose.isIdentity()){
        return;
    }
    src_pose = pose;
    if (update_association && !voxel_map.empty()){
        association.clear();
        association.reserve(points_cov.size());
        std::vector<SingleAssociation> asso_tmp(points_cov.size());
        #pragma omp parallel for num_threads(8)
        for (int i = 0; i < points_cov.size(); i++)
        {
            SingleAssociation& asso = asso_tmp[i];
            Eigen::Vector3d p_global = src_pose.block<3, 3>(0,0) * points_cov[i].point + src_pose.topRightCorner(3, 1);
            ikdTree_PointType p_search(p_global.x(), p_global.y(), p_global.z());
            KD_TREE<ikdTree_PointType>::PointVector points_in_radius;
            std::vector<float> dists;
            ikdtree_map_center.Nearest_Search(p_search, 1, points_in_radius, dists);
            if (points_in_radius.empty()){
                asso.associated = false;
                continue;
            }
            if (dists.front() > search_radius){
                asso.associated = false;
                continue;
            }
            
            VoxelIndex idx = point2Index((Eigen::Vector3d() << 
                                points_in_radius.front().x, 
                                points_in_radius.front().y, 
                                points_in_radius.front().z).finished(), 
                                voxel_size);

            if (voxel_map[idx].N < min_point_density){
                asso.associated = false;
                continue;
            } 
            asso.voxel_idx = idx;
            asso.point = points_cov[i].point;
            asso.cov = points_cov[i].cov;
        }

        for (int i = 0; i < asso_tmp.size(); i++)
        {
            if (asso_tmp[i].associated == true){
                association.push_back(asso_tmp[i]);
            }
        }
    }
}

void VGICP::pushSourceIntoMap()
{
    if (src_cloud == nullptr){
        return;
    }
    if (src_cloud->empty()){
        return;
    }

    Eigen::Matrix3d R = src_pose.block<3, 3>(0,0);
    Eigen::Vector3d t = src_pose.topRightCorner(3, 1);
    std::unordered_set<VoxelIndex, VoxelIndexHash> indices_update;
    KD_TREE<ikdTree_PointType>::PointVector centers_new;

    auto t1 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < points_cov.size(); i++)
    {
        Eigen::Vector3d point = R * points_cov[i].point + t;
        VoxelIndex idx = point2Index(point, voxel_size);

        if (voxel_map.find(idx) != voxel_map.end()){
            if (voxel_map[idx].N > 100)continue;
        }else{
            Eigen::Vector3d center = idx.getCenter(voxel_size);
            ikdTree_PointType center_;
            center_.x = center.x();
            center_.y = center.y();
            center_.z = center.z();
            centers_new.push_back(center_);
        }
        voxel_map[idx].sum += point;
        voxel_map[idx].outer += point * point.transpose();
        voxel_map[idx].N += 1;
        indices_update.insert(idx);
    }

    auto t2 = std::chrono::high_resolution_clock::now();
    for (const auto& idx : indices_update)
    {
        voxel_map[idx].mean = voxel_map[idx].sum / float(voxel_map[idx].N);
        voxel_map[idx].cov = (voxel_map[idx].outer - voxel_map[idx].sum * voxel_map[idx].mean.transpose()) / float(voxel_map[idx].N);
    }
    auto t3 = std::chrono::high_resolution_clock::now();
    if (ikdtree_map_center.size() == 0){
        ikdtree_map_center.Build(centers_new);
    }else{
        ikdtree_map_center.Add_Points(centers_new, false);
    }
    auto t4 = std::chrono::high_resolution_clock::now();
    std::cout << indices_update.size() << ", " <<
        float(std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count()) / float(1e3) << ", "  <<
        float(std::chrono::duration_cast<std::chrono::microseconds>(t3-t2).count()) / float(1e3) << ", " << 
        float(std::chrono::duration_cast<std::chrono::microseconds>(t4-t3).count()) / float(1e3) << std::endl;
}

}



