#include "my_fastlio/my_fastlio.hpp"
#include "Eigen/src/Core/GlobalFunctions.h"
#include "Eigen/src/Core/Matrix.h"
#include "Eigen/src/Core/util/Memory.h"
#include "Eigen/src/Geometry/Quaternion.h"
#include "ieskf/ieskf.hpp"
#include "manifold/manifold.hpp"
#include "manifold/so3.hpp"
#include "my_fastlio/my_fastlio_param.hpp"
#include "pcl/common/transforms.h"
#include "pcl/filters/voxel_grid.h"
#include "pcl/impl/point_types.hpp"
#include "pcl/io/pcd_io.h"
#include "pcl/make_shared.h"
#include <cmath>
#include <cstdlib>
#include <functional>
#include <memory>
#include <mutex>
#include <sophus/se3.hpp>
#include <sophus/so3.hpp>
#include <stdexcept>
#include <pcl/registration/ndt.h>
#include <pcl/filters/random_sample.h>
#include <utility>
#include <omp.h>
#include <vector>
#include "utils/slam_utils.hpp"
#include "voxel_map/voxel_map_util.hpp"


namespace my_fastlio
{


MyFastLIO::MyFastLIO()
{

    kf.computeFxAndFw = std::bind(&MyFastLIO::computeFxAndFw, this, 
                                  std::placeholders::_1, 
                                  std::placeholders::_2, 
                                  std::placeholders::_3, 
                                  std::placeholders::_4, 
                                  std::placeholders::_5,
                                  std::placeholders::_6);

    MyFastLIO::IESKF::NoiseConvarianceType noise_cov;
    MyFastLIO::IESKF::ErrorStateConvarianceType error_cov;
    error_cov.setZero();
    error_cov.block<3,3>(0,0) = 1e-5 * M3T::Identity();
    error_cov.block<3,3>(3,3) = 1e-5 * M3T::Identity();
    error_cov.block<3,3>(6,6) = 1e-5 * M3T::Identity();
    error_cov.block<3,3>(9,9) = 1e-5 * M3T::Identity();
    error_cov.block<3,3>(12, 12) = 1e-4 * M3T::Identity();
    error_cov.block<3,3>(15, 15) = 1e-5 * M3T::Identity();
    error_cov.block<3,3>(18, 18) = 1e-5 * M3T::Identity();
    error_cov.block<2,2>(21, 21) = 1e-7 * M2T::Identity();
    noise_cov.setZero();
    noise_cov.block<3,3>(0,0) = 1e-4 * M3T::Identity();
    noise_cov.block<3,3>(3,3) = 1e-4 * M3T::Identity();
    noise_cov.block<3,3>(6,6) = 1e-5 * M3T::Identity();
    noise_cov.block<3,3>(9,9) = 1e-5 * M3T::Identity();

    (*this).setErrorConvariance(error_cov)
           .setNoiseConvariance(noise_cov);


    // Initialize the data queues and mutexes
    imu_data_queue_.clear();
    meas_pack_mutex_.unlock();
}

MyFastLIO::~MyFastLIO()
{
    std::unique_lock<std::mutex> lock1(meas_pack_mutex_);
    meas_pack_.clear();
    lock1.unlock();
}

bool MyFastLIO::imuAsyncPushImu(std::shared_ptr<ImuData> data)
{
    std::unique_lock<std::mutex> lock(imu_data_mutex_);
    data->a *= imu_acc_scale;
    imu_data_queue_.push_back(data);
    lock.unlock();
    // std::cout << "imu: " << data->a.transpose() << std::endl;
    return true;
}

bool MyFastLIO::lidarAsyncPushLidar(std::shared_ptr<LidarData> lidar_)
{
    lidar_data_queue_.push_back(lidar_);
    std::unique_lock<std::mutex> lock(meas_pack_mutex_);
    while (lidar_data_queue_.size() > 0){
        std::shared_ptr<LidarData> lidar = lidar_data_queue_.front();
        if (imu_data_queue_.back()->time < lidar->time) return true;

        auto meas = std::make_shared<MeasurePack>();
        while (!imu_data_queue_.empty())
        {
            auto& imu = imu_data_queue_.front();
            if (imu->time > lidar->time) break;
            meas->imu_lst->push_back(*imu);
            imu_data_queue_.pop_front();
        }
        if (meas->imu_lst->size() > 0){
            meas->cloud = lidar;
            meas_pack_.push_back(meas);
        }
        lidar_data_queue_.pop_front();
    }
    lock.unlock();
    meas_pack_cond_.notify_one();
    return true;
}

void MyFastLIO::imuPropagate(const ImuData& imu0, const ImuData& imu1)
{
    ImuData imu;
    imu.a = 0.5 * (imu0.a + imu1.a);
    imu.w = 0.5 * (imu0.w + imu1.w);
    double dt = imu1.time - imu0.time;

    IESKF::ErrorStateType delta;
    delta.setZero();
    StateType& X = kf.normal_state;
    const V3T& a = imu.a;
    const V3T& w = imu.w;
    delta.segment<3>(IDX_p * 3) = getM(X, v) +  0.5 * dt * (getM(X, R).matrix() * (a - getM(X, ba)) + getM(X, g));
    delta.segment<3>(IDX_R * 3) = w - getM(X, bw);
    delta.segment<3>(IDX_v * 3) = getM(X, R).matrix() * (a - getM(X, ba)) + getM(X, g);
    // delta *= dt;
    ControlType control;
    getM(control, w) = w;
    getM(control, a) = a;
    kf.predict(delta, control, dt);
}

void MyFastLIO::handlePack()
{

    std::unique_lock<std::mutex> lock_pack(meas_pack_mutex_);
    meas_pack_cond_.wait(lock_pack, [this] { return !meas_pack_.empty(); });
    std::shared_ptr<MeasurePack> meas = meas_pack_.front();
    meas_pack_.pop_front();
    lock_pack.unlock();

    std::deque<std::pair<double, Sophus::SE3d>> imu_odo;

    ImuData last_imu, curr_imu;
    auto start = slam_utils::TimerHelper::start();
    if (last_meas_ != nullptr){
        last_imu = last_meas_->imu_lst->back();
        curr_imu = meas->imu_lst->front();

        imu_odo.push_back(std::make_pair(last_imu.time, Sophus::SE3d(getM(kf.normal_state, R), getM(kf.normal_state, p))));
        imuPropagate(last_imu, curr_imu);
        imu_odo.push_back(std::make_pair(curr_imu.time, Sophus::SE3d(getM(kf.normal_state, R), getM(kf.normal_state, p))));
    }else{
        imu_odo.push_back(std::make_pair(meas->imu_lst->front().time, Sophus::SE3d(getM(kf.normal_state, R), getM(kf.normal_state, p))));
    }

    for (int i = 0; i < meas->imu_lst->size() - 1; i++)
    {
        curr_imu = meas->imu_lst->at(i+1);
        last_imu = meas->imu_lst->at(i);
        imuPropagate(last_imu, curr_imu);
        imu_odo.push_back(std::make_pair(curr_imu.time, Sophus::SE3d(getM(kf.normal_state, R), getM(kf.normal_state, p))));
    }
    std::cout << "[forward] imu propagate cost: " << slam_utils::TimerHelper::end(start) << " ms" << std::endl; 

    // lidarCompensation(meas, imu_odo);

    buildmapAndUpdate(meas);

    last_meas_ = meas;
}


void MyFastLIO::lidarCompensation(std::shared_ptr<MeasurePack> meas, const std::deque<std::pair<double, Sophus::SE3d>>& tmp_odo)
{
    auto& cloud = meas->cloud->cloud;
    pcl::io::savePCDFile("/root/workspace/my_fastlio/build/bagsave/before.pcd", *cloud);
    auto getPose = [&tmp_odo](double t, Sophus::SE3d& ret){
        int idx = 0;
        for (int i = 0; i < tmp_odo.size() - 1; i++){
            double t0 = tmp_odo[i].first, t1 = tmp_odo[i + 1].first;
            double slerp_coeff = (t - t0) / (t1 - t0);
            if (t > t0 && t < t1){
                auto& se3_0 = tmp_odo[i].second;
                auto& se3_2 = tmp_odo[i + 1].second;
                QuaT so3 = se3_0.unit_quaternion().slerp(slerp_coeff, se3_2.unit_quaternion());
                V3T trans = (1 - slerp_coeff) * tmp_odo[i].second.translation() + slerp_coeff * tmp_odo[i + 1].second.translation();
                ret = Sophus::SE3d(so3, trans);
                return true;
            }
        }
        return false;
    };
    Sophus::SE3d T_IktoI0 = Sophus::SE3d(getM(kf.normal_state, R), getM(kf.normal_state, p)).inverse();
    Sophus::SE3d T_ItoL(getM(kf.normal_state, R_ItoL), getM(kf.normal_state, p_ItoL));
    Sophus::SE3d T_LtoI = T_ItoL.inverse();
    int count = 0;
    V3T ava_change; ava_change.setZero();
    auto t1 = std::chrono::high_resolution_clock::now();
    omp_set_num_threads(8);  // 实测8线程处理较快
    #pragma omp parallel
    {
        V3T local_change = V3T::Zero();
        int local_count = 0;

        #pragma omp for nowait
        for (int i = 0; i < cloud->points.size(); i++)
        {
            auto& p_inj = cloud->points[i];
            double t = meas->cloud->time - p_inj.curvature;
            Sophus::SE3d T_I0toIj;
            if (!getPose(t, T_I0toIj)){
                continue;
            }
            V3T p_ink = T_LtoI * (T_IktoI0 * T_I0toIj) * T_ItoL * V3T(p_inj.getVector3fMap().cast<double>());
            local_change += (p_ink - p_inj.getVector3fMap().cast<double>()).cwiseAbs();
            p_inj.x = p_ink.x();
            p_inj.y = p_ink.y();
            p_inj.z = p_ink.z();
            local_count++;
        }

        #pragma omp critical
        {
            ava_change += local_change;
            count += local_count;
        }

    }

    auto t2 = std::chrono::high_resolution_clock::now();
    auto add_duration = std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count();      
    std::cout << "[backward] undistort " << count << " points(" << float(add_duration)/1e3 <<" ms): " << double(count) / double(cloud->points.size()) * 100 << "% avarage change: " << ava_change.transpose() / double(count) << std::endl;
    pcl::io::savePCDFile("/root/workspace/my_fastlio/build/bagsave/after.pcd", *cloud);
    // #ifdef _OPENMP
    // std::cout << "OpenMP enabled with " << omp_get_max_threads() << " threads\n";
    // #else
    // std::cout << "OpenMP NOT enabled\n";
    // #endif
}

// TODO:
void MyFastLIO::computeFxAndFw(IESKF::ErrorPropagateFx& fx, IESKF::ErrorPropagateFw& fw, const StateType& X, const IESKF::ErrorStateType& delta_x, const ControlType& u, const double dt)
{
    fx.setIdentity();
    fw.setZero();
    fx.block<3,3>(IDX_p * 3, IDX_v * 3) = dt * M3T::Identity();
    fx.block<3,3>(IDX_R * 3, IDX_bw * 3) = -dt * SO3Jr(dt * (getM(u, w) - getM(X, bw)));
    fx.block<3,3>(IDX_R * 3, IDX_R * 3) = SO3Exp(-dt * (getM(u, w) - getM(X, bw))).matrix();
    fx.block<3,3>(IDX_v * 3, IDX_R * 3) = -dt * getM(X, R).matrix() * SO3Hat(getM(u, a) - getM(X, ba));
    fx.block<3,3>(IDX_v * 3, IDX_ba * 3) =  -dt * getM(X, R).matrix();
    // fx.block<3,2>(IDX_v * 3, IDX_g * 3) = Eigen::Matrix<double, 3, 2>::Zero();
    Eigen::Matrix<double,2,1> delta_g = delta_x.segment<2>(IDX_g * 3);
    // TODO:
    fx.block<3,2>(IDX_v * 3, IDX_g * 3) = dt * std::get<7>(X.data).boxplusJacobian(delta_g);
    // fx.block<3,2>(IDX_v * 3, IDX_g * 3).setZero();

    fw.block<3,3>(IDX_R * 3, IDX_nw * 3) = -dt * SO3Jr(dt * (getM(u, w) - getM(X, bw)));
    fw.block<3,3>(IDX_v * 3, IDX_na * 3) = -dt * getM(X, R).matrix();
    fw.block<3,3>(IDX_bw * 3, IDX_nbw * 3) = dt * M3T::Identity();
    fw.block<3,3>(IDX_ba * 3, IDX_nba * 3) = dt * M3T::Identity();
}


void MyFastLIO::start()
{
    if (isReady()){
        main_thread_ = std::make_shared<std::thread>(&MyFastLIO::lioThread, this);
        pthread_setname_np(main_thread_->native_handle(), "myfastlio_thread");
        main_thread_->detach();
    }else{
        throw std::runtime_error("Parameters in lio didn't completely initialized!");
    }

    std::cout << "\n============================================= LIO Start =============================================\n";
    std::cout << "Param R_ItoL: " << R_ItoL_->unit_quaternion().coeffs().transpose() << std::endl;
    std::cout << "Param t_ItoL" << p_ItoL_->transpose() << std::endl;
    // std::cout << "Param noise cov:\n" << kf.noise_cov << std::endl;
    // std::cout << "Param error cov:\n" << kf.error_cov << std::endl;
    std::cout << std::endl;
}


void MyFastLIO::staticStateIMUInitialize()
{
    std::unique_lock<std::mutex> lock(meas_pack_mutex_);
    meas_pack_cond_.wait(lock, [this]{
        if (meas_pack_.empty()){
            return false;
        }
        if (meas_pack_.back()->imu_lst->back().time - meas_pack_.front()->imu_lst->front().time > 1.5){
            return true;
        }else{
            return false;
        }
    });
    double start_t = meas_pack_.front()->imu_lst->front().time;
    std::vector<ImuData> imu_lst;
    bool imu_full = false;
    for (auto& meas : meas_pack_)
    {
        for(auto& imu : *meas->imu_lst)
        {
            if (imu.time - start_t > 1.5){
                imu_full = true;
                break;
            }
            imu_lst.push_back(imu);
        }
        if (imu_full) break;
    }

    V3T ava_acc(V3T::Zero()), ava_gyro(V3T::Zero());
    for (auto& imu : imu_lst)
    {
        ava_acc += imu.a;
        ava_gyro += imu.w;
    }
    ava_gyro /= float(imu_lst.size());
    ava_acc /= float(imu_lst.size());

    imu_acc_scale = GRAVITY / ava_acc.norm();

    for (auto& meas : meas_pack_)
    {
        for(auto& imu : *meas->imu_lst)
        {
            imu.a *= imu_acc_scale;
        }
    }
    lock.unlock();
    std::unique_lock<std::mutex> lock_(imu_data_mutex_);
    for (auto& imu : imu_data_queue_)
    {
        imu->a *= imu_acc_scale;
    }
    lock_.unlock();


    getM(kf.normal_state, bw) = ava_gyro;
    getM(kf.normal_state, g) = V3T(0,0,-9.81);
    std::cout << "avarage acc: " << ava_acc.transpose() << std::endl;
    Eigen::Quaterniond R0 = Eigen::Quaterniond::FromTwoVectors(ava_acc, Eigen::Vector3d::UnitZ());
    getM(kf.normal_state, R) = Sophus::SO3d(R0.matrix());
    std::cout << (R0 * ava_acc).transpose() << std::endl;

    std::cout << "Bias gyro: " << ava_gyro.transpose() << std::endl;
    std::cout << "R0: " << R0.coeffs().transpose() << std::endl;
}

// bool MyFastLIO::esti_plane(V4T &pca_result, const KDTree::PointVector &point, const float &threshold)
// {
//     Eigen::Matrix<double, plane_N_search, 3> A;
//     Eigen::Matrix<double, plane_N_search, 1> b;
//     A.setZero();
//     b.setOnes();
//     b *= -1.0f;

//     for (int j = 0; j < plane_N_search; j++)
//     {
//         A(j,0) = point[j].x;
//         A(j,1) = point[j].y;
//         A(j,2) = point[j].z;
//     }

//     Eigen::Matrix<double, 3, 1> normvec = A.colPivHouseholderQr().solve(b);

//     double n = normvec.norm();
//     pca_result(0) = normvec(0) / n;
//     pca_result(1) = normvec(1) / n;
//     pca_result(2) = normvec(2) / n;
//     pca_result(3) = 1.0 / n;

//     for (int j = 0; j < plane_N_search; j++)
//     {
//         if (fabs(pca_result(0) * point[j].x + pca_result(1) * point[j].y + pca_result(2) * point[j].z + pca_result(3)) > threshold)
//         {
//             return false;
//         }
//     }
//     return true;
// }




inline void MyFastLIO::transformVoxelPoint(const PointVMap& in, PointVMap& out, const M3T& R, const V3T& t, const M3T& cov_R, const M3T& cov_p)
{
    M3T R_hat_p = R * SO3Hat(in.point);
    out.point = R * in.point + t;
    out.cov = R * in.cov * R.transpose() + R_hat_p * cov_R * R_hat_p.transpose() + cov_p;
}

void MyFastLIO::transformVoxelCloud(CloudVmapPtr in, CloudVmapPtr& out, const M3T& R, const V3T& t, const M3T& cov_R, const M3T& cov_p)
{
    if (in == nullptr){
        throw std::runtime_error("Input cloudptr is nullptr!");
    }
    if (out == nullptr){
        out = pcl::make_shared<CloudVMap>();
    }

    out->resize(in->size());
    omp_set_num_threads(8);
    #pragma omp parallel
    {
        #pragma omp for nowait
        for (int i = 0; i < in->size(); i++)
        {
            PointVMap& p = in->at(i);
            PointVMap& vp = out->at(i);
            transformVoxelPoint(p, vp, R, t, cov_R, cov_p);
        }
    }
}



void MyFastLIO::computeHxAndR(std::vector<voxelmap::ptpl>& ptpl, IESKF::ObserveResult& zk, IESKF::ObserveMatrix& H, IESKF::ObserveCovarianceType& Rk_inv, const StateType& Xk,const StateType& X)
{
    // std::cout << "[update] avarage scan matched num: " << ptpl.size() << ", cost: " << slam_utils::TimerHelper::end(start) << " ms" << std::endl;

    H.resize(ptpl.size(), StateType::DOF);
    H.setZero();
    Rk_inv.resize(ptpl.size(), ptpl.size());
    Rk_inv.setZero();
    zk.resize(ptpl.size(), 1);

    omp_set_num_threads(4);

    M3T R = getM(Xk, R).matrix();
    V3T p = getM(Xk, p).matrix();
    M3T R_ItoL = getM(Xk, R_ItoL).matrix();
    V3T p_ItoL = getM(Xk, p_ItoL).matrix();

    // 实测单核心更快
    // #pragma omp parallel
    {
        // #pragma omp for nowait
        for(int i = 0; i < ptpl.size(); i++)
        {
            Eigen::Matrix<double, 1, 3> uT = ptpl[i].normal.transpose();
            Eigen::Matrix<double, 1, 3> uTR = uT * R;
            V3T& pl = ptpl[i].point;
            V3T& q = ptpl[i].center;
            H.block<1, 3>(i, IDX_p * 3) = uT;
            H.block<1, 3>(i, IDX_R * 3) = -uTR * SO3Hat(R_ItoL * pl + p_ItoL);
            // H.block<1, 3>(i, IDX_R_ItoL * 3) = -uTR * R_ItoL * SO3Hat(pl);
            // H.block<1, 3>(i, IDX_p_ItoL * 3) = uTR;
            // if (i < 10) std::cout << H.row(i) << std::endl;

            Eigen::Matrix<double, 1, 6> Huq;
            V3T vec = (R * (R_ItoL * pl + p_ItoL) + p - q);
            Huq.block<1,3>(0, 0) = vec.transpose();
            Huq.block<1,3>(0, 3) = uT;
            Eigen::Matrix<double, 1, 3> Hp = uTR * R_ItoL;

            Rk_inv(i,i) = Huq * ptpl[i].plane_cov * Huq.transpose();
            Rk_inv(i, i) += Hp * ptpl[i].point_cov * Hp.transpose();
            Rk_inv(i, i) = 1 * 1e9 / Rk_inv(i, i);

            zk(i, 0) = uT * vec;
        }
    
    }
    // std::cout << "[update] measure " << ptpl.size() << " points (res " << zk.norm() << "), cost: " << slam_utils::TimerHelper::end(start) << " ms" << std::endl;
    // std::cout << zk.transpose() << std::endl;
    // std::cout << Rk_inv.diagonal().transpose() << std::endl;
}




void MyFastLIO::buildmapAndUpdate(std::shared_ptr<MeasurePack> meas)
{
    auto start = slam_utils::TimerHelper::start();
    static double start_time = 0;
    if (vmap->empty()){
        CloudPtr cloud(new CloudT);
        std::unique_lock<std::mutex> lock(meas_pack_mutex_);
        for(int i = 0; i < meas_pack_.size(); i++){
            if (i == 0){
                start_time = meas_pack_[i]->cloud->time;
            }
            if (meas_pack_[i]->cloud->time - 1.5 > start_time){
                break;
            }
            CloudT tmp;
            pcl::transformPointCloud(*meas_pack_[i]->cloud->cloud, tmp, T_ItoL_.matrix().cast<float>());
            *cloud += tmp;
        }
        // pcl::io::savePCDFile("/root/workspace/my_fastlio/build/bagsave/pre_map.pcd", *cloud);

        CloudVmapPtr global_cloud;
        transformPCLCloud2GlobalVMap(cloud, global_cloud);
        std::cout << "cloud to global vmap cost: " << slam_utils::TimerHelper::end(start) << std::endl;
        buildVoxelMap(*global_cloud, max_voxel_size, max_layer, layer_size,
                      max_point_size, max_cov_point_size, plane_threshold,
                      *vmap);
        map_for_publish = meas->cloud->cloud;
        start_time = meas->cloud->time;
    // }else if (meas->cloud->time - start_time < 1.8){
    //     CloudVmapPtr global_cloud;
    //     transformPCLCloud2GlobalVMap(meas->cloud->cloud, global_cloud);
    //     voxelmap::updateVoxelMap(*global_cloud, max_voxel_size, max_layer, layer_size,
    //                   max_point_size, max_cov_point_size, plane_threshold,
    //                   *vmap);
    }else{
        auto start = slam_utils::TimerHelper::start();
        CloudXYZPtr filtered_cloud(new CloudXYZ);
        // pcl::RandomSample<PointXYZ> sample;
        pcl::VoxelGrid<pcl::PointXYZ> sample;
        CloudXYZPtr xyz_cloud(new CloudXYZ);
        pcl::copyPointCloud(*meas->cloud->cloud, *xyz_cloud);
        // sample.setSample(frame_residual_count);
        sample.setLeafSize(0.3, 0.3, 0.3);
        sample.setInputCloud(xyz_cloud);
        sample.filter(*filtered_cloud);
        map_indices = sample.getIndices();
        map_for_publish = meas->cloud->cloud;

        std::cout << "[update] filter num: " << filtered_cloud->size() << ", cost: " << slam_utils::TimerHelper::end(start) << " ms" << std::endl;
        

        CloudVmapPtr local_cloud;
        Sophus::SE3d T_WtoIi(getM(kf.normal_state, R), getM(kf.normal_state, p));
        Sophus::SE3d T_ItoL(getM(kf.normal_state, R_ItoL), getM(kf.normal_state, p_ItoL)); 
        Sophus::SE3d T_WtoLi = T_WtoIi * T_ItoL;
        pclCloud2VoxelMapCloud(filtered_cloud, local_cloud, (T_WtoIi * T_ItoL).matrix());

        std::vector<Eigen::Vector3d> point_world(filtered_cloud->size());
        for(int i = 0 ; i < filtered_cloud->size(); i++){
            point_world[i] = T_WtoLi * filtered_cloud->at(i).getVector3fMap().cast<double>();
        }

        
        std::vector<voxelmap::ptpl>& ptpl = *ptpl_cb;
        std::vector<Eigen::Vector3d> non_match;
        voxelmap::BuildResidualListOMP(*vmap, max_voxel_size, 1, max_layer, *local_cloud, point_world, ptpl, non_match);


        kf.computeHxAndRinv = [this](IESKF::ObserveResult& zk, IESKF::ObserveMatrix& H, IESKF::ObserveCovarianceType& Rinv, const StateType& Xk,const StateType& X){
            this->computeHxAndR(*ptpl_cb, zk, H, Rinv, Xk, X);
        };
        IESKFUpdateInfo info = kf.update(NUM_MAX_ITERATIONS, 1e-4);
        std::cout << "[update] cost change: " << info.begin_cost << " -> " << info.end_cost << ", change: " << info.begin_cost - info.end_cost << std::endl;
        std::cout << "[update] iteration time: " << info.cost_time << " ms" << std::endl;
       
        auto start_vmap_update = slam_utils::TimerHelper::start();
        CloudVmapPtr cloud_global(new CloudVMap);
        transformPCLCloud2GlobalVMap(filtered_cloud, cloud_global);
        voxelmap::updateVoxelMap(*cloud_global, max_voxel_size, 
                max_layer, layer_size, max_point_size, 
                max_cov_point_size, plane_threshold, *vmap);
        std::cout << "[update] voxmap update cost: " << slam_utils::TimerHelper::end(start_vmap_update) << " ms" << std::endl;
        
        std::cout << "[update] total time: " << slam_utils::TimerHelper::end(start) << " ms" << std::endl;
    }

}

void MyFastLIO::lioThread()
{
    staticStateIMUInitialize();
    while (true)
    {
        handlePack();
        if (updateCallback){
            std::shared_ptr<CallbackInfo> ret_info(new CallbackInfo);
            ret_info->time = last_meas_->cloud->time;
            ret_info->pose = Sophus::SE3d(getM(kf.normal_state, R), getM(kf.normal_state, p));
            ret_info->vel = getM(kf.normal_state, v);
            ret_info->map = map_for_publish;
            ret_info->filtered_indices = map_indices;
            ret_info->vmap = vmap;
            ret_info->ptpl = ptpl_cb;
            updateCallback.value()(ret_info);
        }
    }
}
    
} // namespace my_fastlio


