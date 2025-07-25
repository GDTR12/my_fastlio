#pragma once
#include <functional>
#include <iostream>
#include <Eigen/Core>
#include <memory>
#include <optional>
#include <pcl/pcl_macros.h>
#include <pcl/point_cloud.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <deque>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <condition_variable>
#include <sophus/se3.hpp>
#include <unordered_map>
#include <vector>

#include "Eigen/src/Core/Matrix.h"
#include "ikd_Tree.h"
#include "ivgicp.hpp"
#include "manifold/manifold.hpp"
#include "manifold/s2.hpp"
#include "manifold/so3.hpp"
#include "my_fastlio/my_fastlio_param.hpp"
#include "ieskf/ieskf.hpp"

#include "pcl/common/transforms.h"
#include "pcl/make_shared.h"
#include "pcl/pcl_base.h"
#include "voxel_map/common_lib.h"
#include "voxel_map/voxel_map_util.hpp"

#define getM(X, I) std::get<(IDX_##I)>((X.data))

namespace my_fastlio
{

constexpr char rot_str[] = "r";
constexpr char g_str[] = "g";

enum StateIDX
{
    IDX_p,
    IDX_R,
    IDX_R_ItoL,
    IDX_p_ItoL,
    IDX_v,
    IDX_bw,
    IDX_ba,
    IDX_g,
};

enum ControlIDX
{
    IDX_w,
    IDX_a
};

enum NoiseIDX
{
    IDX_nw,
    IDX_na,
    IDX_nbw,
    IDX_nba,
};

using namespace ieskf;

namespace mfd = manifold;


class MyFastLIO
{
public:
    typedef mfd::Manifolds<mfd::Vector<3>, mfd::SO3, mfd::SO3, mfd::Vector<3>, mfd::Vector<3>, mfd::Vector<3>, mfd::Vector<3>,
                           mfd::S2<98090, 10000, mfd::CLOSE_Z>> StateType;

    typedef mfd::Manifolds<mfd::Vector<3>, mfd::Vector<3>> ControlType;

    using IESKF = ieskf::IESKF<StateType, ControlType,12>;

    struct MeasurePack
    {
        std::shared_ptr<std::deque<ImuData>> imu_lst = std::make_shared<std::deque<ImuData>>();
        std::shared_ptr<LidarData> cloud = std::make_shared<LidarData>();
    };

    enum ObservationMethod
    {
        POINT_TO_PLANE  = 0,
        POINT_TO_VOXELMAP,
        METHOD_VGICP
    };


    struct LIOdometryInfo
    {
        double time;
        Sophus::SE3d pose;
        CloudXYZPtr cloud;
        std::shared_ptr<std::deque<ImuData>> imu_lst;
    };

    struct CallbackInfo
    {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        double time;
        Sophus::SE3d pose;
        V3T vel;
        V3T ba, bg;
        V3T grav;
        CloudPtr map;
        CloudXYZPtr vgicp_center;
        std::shared_ptr<VMap> vmap;
        pcl::IndicesPtr filtered_indices;
        std::shared_ptr<std::vector<voxelmap::ptpl>> ptpl;
    };


    MyFastLIO();
    ~MyFastLIO();

    bool imuAsyncPushImu(std::shared_ptr<ImuData> data);

    bool lidarAsyncPushLidar(std::shared_ptr<LidarData> data);

    void start();

    MyFastLIO& setR_ItoL(const M3T& R_ItoL)
    {
        R_ItoL_ = Sophus::SO3d(R_ItoL); 
        getM(kf.normal_state, R_ItoL) = R_ItoL_.value();
        return *this;
    }

    MyFastLIO& setp_ItoL(const V3T& t_ItoL)
    {
        p_ItoL_ = t_ItoL; 
        getM(kf.normal_state, p_ItoL) = p_ItoL_.value();
        return *this;
    }

    MyFastLIO& setNoiseConvariance(const IESKF::NoiseConvarianceType& noise_cov)
    {
        kf.noise_cov = noise_cov;
        inited_noise_cov_ = true;
        return *this;
    }


    MyFastLIO& setErrorConvariance(const IESKF::ErrorStateConvarianceType& error_cov)
    {
        kf.error_cov = error_cov;
        inited_error_cov_ = true;
        return *this;
    }

    MyFastLIO& setUpdateCallback(std::function<void(std::shared_ptr<CallbackInfo>)> callback)
    {
        updateCallback = callback;
        return *this;
    }

    MyFastLIO& setInitializeGravity(const V3T& g)
    {
        getM(kf.normal_state, g) = g;
        inited_gravity_ = true;
        return *this;
    }


private:

    // typedef KD_TREE<PointT> KDTree;
    bool isReady()
    {
        if (R_ItoL_.has_value() &&
            p_ItoL_.has_value() && 
            inited_error_cov_ && 
            inited_noise_cov_){
            
            T_ItoL_ = Sophus::SE3d(R_ItoL_->unit_quaternion(), *p_ItoL_);
            return true;
        }
        return false;
    }

    void lioThread();

    void handlePack();

    // typedef PointXYZ PointType;
    template<typename PointType>
    void pclCloud2VoxelMapCloud(pcl::shared_ptr<pcl::PointCloud<PointType>> cloud, CloudVmapPtr& vcloud)
    {
        if (vcloud == nullptr){
            vcloud = pcl::make_shared<CloudVMap>();
        }
        if (cloud == nullptr){
            throw std::runtime_error("Input cloudptr is nullptr!");
        }
        vcloud->resize(cloud->size());
        omp_set_num_threads(8);
        #pragma omp parallel
        {
            #pragma omp for nowait
            for (int i = 0; i < cloud->size(); i++)
            {
                V3T p = cloud->at(i).getVector3fMap().template cast<double>();
                PointVMap& vp = vcloud->at(i);
                p.z() = std::abs(p.z()) < 1e-7 ? 0.0001 : p.z();
                M3T cov;
                voxelmap::calcBodyCov(p, lidar_range_cov, lidar_angle_cov, cov);
                vp.point = p;
                vp.cov = cov;
            }
        }
    }

    template<typename PointType>
    void transformPCLCloud2GlobalVMap(pcl::shared_ptr<pcl::PointCloud<PointType>> cloud, CloudVmapPtr& global_cloud)
    {
        if (global_cloud == nullptr){
            global_cloud = pcl::make_shared<CloudVMap>();
        }
        CloudVmapPtr local_cloud(new CloudVMap), local_cloud2(new CloudVMap);
        Sophus::SE3d T_WtoIi(getM(kf.normal_state, R), getM(kf.normal_state, p));
        pclCloud2VoxelMapCloud(cloud, local_cloud);
        M3T cov_R = kf.error_cov.block<3,3>(IDX_R_ItoL * 3, IDX_R_ItoL * 3);
        M3T cov_p = kf.error_cov.block<3,3>(IDX_p_ItoL * 3, IDX_p_ItoL * 3);

        transformVoxelCloud(local_cloud, local_cloud2, T_ItoL_.rotationMatrix(), T_ItoL_.translation(), cov_R, cov_p);
        cov_R = kf.error_cov.block<3,3>(IDX_R * 3, IDX_R * 3);
        cov_p = kf.error_cov.block<3,3>(IDX_p * 3, IDX_p * 3);
        transformVoxelCloud(local_cloud2, global_cloud, T_WtoIi.rotationMatrix(), T_WtoIi.translation(), cov_R, cov_p);
    }

    void accumalateVMap(CloudVmapPtr& vmap0, CloudVmapPtr& vmap1);

    // using PointType = pcl::PointXYZ;
    template<typename PointType>
    void transformPCLCloud2GlobalKDTreeMap(pcl::shared_ptr<pcl::PointCloud<PointType>> cloud, CloudKdTreePtr& global_cloud)
    {
        if (cloud == nullptr){
            throw std::runtime_error("Input cloud ptr is null!");
        }
        if (global_cloud == nullptr){
            global_cloud = pcl::make_shared<CloudKdTree>();
        }

        Sophus::SE3d T_WtoIi(getM(kf.normal_state, R), getM(kf.normal_state, p));
        Sophus::SE3d T_ItoL(getM(kf.normal_state, R_ItoL), getM(kf.normal_state, p_ItoL));
        Sophus::SE3d T_WtoLi = T_WtoIi * T_ItoL;
        global_cloud->resize(cloud->size());
        for (int i = 0; i < cloud->size(); i++){
            Eigen::Vector3d p = T_WtoLi * cloud->at(i).getVector3fMap().template cast<double>();
            global_cloud->at(i).x = p.x();
            global_cloud->at(i).y = p.y();
            global_cloud->at(i).z = p.z();
        }
    }

    inline void transformVoxelPoint(const PointVMap& in, PointVMap& out, const M3T& R, const V3T& t, const M3T& cov_R, const M3T& cov_p);

    void transformVoxelCloud(CloudVmapPtr in, CloudVmapPtr& out, const M3T& R, const V3T& t, const M3T& cov_R, const M3T& cov_p);

    // void transformPCLCloud2GlobalVMap(CloudPtr cloud, CloudVmapPtr& vcloud);

    // bool esti_plane(V4T &pca_result, const KDTree::PointVector &point, const float &threshold);

    void buildmapAndUpdate(std::shared_ptr<MeasurePack> meas);

    void lidarCompensation(std::shared_ptr<MeasurePack> meas, const std::deque<std::pair<double, Sophus::SE3d>>& tmp_odo);

    void imuPropagate(const ImuData& imu0, const ImuData& imu1);

    void computeFxAndFw(IESKF::ErrorPropagateFx& fx, IESKF::ErrorPropagateFw& fw, const StateType& X, const IESKF::ErrorStateType& delta_x,const ControlType& u, const double dt);

    void computeHxAndRinvWithCov(std::vector<voxelmap::ptpl>& ptpl, IESKF::ObserveResult& zk, IESKF::ObserveMatrix& H, IESKF::ObserveCovarianceType& R, const StateType& Xk,const StateType& X, bool with_cov = true);

    // void computeHxAndRinvKDTree(std::vector<voxelmap::ptpl>& ptpls, IESKF::ObserveResult& zk, IESKF::ObserveMatrix& H, IESKF::ObserveCovarianceType& R, const StateType& Xk,const StateType& X);

    void computeHxAndRinvVGICP(IESKF::ObserveResult& zk, IESKF::ObserveMatrix& H, IESKF::ObserveCovarianceType& R, const StateType& Xk,const StateType& X, float cauchy_dist=-1.0);

    void staticStateIMUInitialize();

    IESKF kf;

    std::shared_ptr<std::thread> main_thread_;

    std::deque<std::shared_ptr<ImuData>> imu_data_queue_;
    std::deque<std::shared_ptr<LidarData>> lidar_data_queue_;

    std::deque<std::shared_ptr<MeasurePack>> meas_pack_;

     

    std::shared_ptr<MeasurePack> last_meas_;
    std::mutex meas_pack_mutex_, imu_data_mutex_;
    std::condition_variable meas_pack_cond_;


    static constexpr double GRAVITY = 9.81;
    double imu_acc_scale = 1.0;
    std::optional<Sophus::SO3d> R_ItoL_;
    std::optional<Eigen::Vector3d> p_ItoL_;
    Sophus::SE3d T_ItoL_;
    bool inited_noise_cov_ = false;
    bool inited_error_cov_ = false;
    bool inited_gravity_ = false;

    /* update callback info */
    std::optional<std::function<void(std::shared_ptr<CallbackInfo>)>> updateCallback;

    CloudPtr map_for_publish = pcl::make_shared<CloudT>();
    pcl::IndicesPtr map_indices;
    std::shared_ptr<std::vector<voxelmap::ptpl>> ptpl_cb = std::make_shared<std::vector<voxelmap::ptpl>>();
    std::shared_ptr<VMap> vmap = std::make_shared<VMap>();
    CloudXYZPtr vgicp_center = CloudXYZPtr(new CloudXYZ);
    std::shared_ptr<CallbackInfo> ret_info = std::shared_ptr<CallbackInfo>(new CallbackInfo);

    KDTree kdtree;
    CloudVmapPtr init_vmap = pcl::make_shared<CloudVMap>();

    gicp::VGICP vgicp;

    // TODO: 超参数放在 config 中
    ObservationMethod observation_method = METHOD_VGICP;

    double lidar_range_cov = 1e-4, lidar_angle_cov = 1e-6;
    double max_voxel_size = 3.0;
    int max_layer = 4.0;
    std::vector<int> layer_size = std::vector<int>({20, 10, 10, 5, 5});
    int max_point_size = 200; int max_cov_point_size = 200; 
    double plane_threshold_kdtree = 0.05;
    double plane_threshold_vmap = 0.01;

    int MAX_H_DIMENSION = 1800;

    // 超参数
    int NUM_MAX_ITERATIONS = 5;

    double vgpicp_pre_map_time = 2.5;

    const int frame_residual_count = 2000;
    static constexpr int plane_N_search  = 5;

};

}

