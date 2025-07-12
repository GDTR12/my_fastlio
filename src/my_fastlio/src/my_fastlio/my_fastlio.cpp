#include "my_fastlio/my_fastlio.hpp"
#include "Eigen/src/Core/GlobalFunctions.h"
#include "Eigen/src/Core/Matrix.h"
#include "Eigen/src/Core/util/Constants.h"
#include "Eigen/src/Core/util/Memory.h"
#include "Eigen/src/Geometry/Quaternion.h"
#include "Eigen/src/SparseCore/SparseUtil.h"
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
#include <cstddef>
#include <cstdlib>
#include <functional>
#include <memory>
#include <mutex>
#include <sophus/se3.hpp>
#include <sophus/so3.hpp>
#include <stdexcept>
#include <pcl/registration/ndt.h>
#include <pcl/filters/random_sample.h>
#include <string>
#include <thread>
#include <utility>
#include <omp.h>
#include <vector>
#include "utils/slam_utils.hpp"
#include "voxel_map/common_lib.h"
#include "voxel_map/voxel_map_util.hpp"
#include "ikd_Tree.h"
#include "mapper/ivgicp.hpp"

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
    error_cov.block<3,3>(0,0) = 1e-3 * M3T::Identity();
    error_cov.block<3,3>(3,3) = 1e-3 * M3T::Identity();
    error_cov.block<3,3>(6,6) = 1e-5 * M3T::Identity();
    error_cov.block<3,3>(9,9) = 1e-5 * M3T::Identity();
    error_cov.block<3,3>(12, 12) = 1e-3 * M3T::Identity();
    error_cov.block<3,3>(15, 15) = 1e-2 * M3T::Identity();
    error_cov.block<3,3>(18, 18) = 1e-2 * M3T::Identity();
    error_cov.block<2,2>(21, 21) = 1e-7 * M2T::Identity();
    noise_cov.setZero();
    noise_cov.block<3,3>(0,0) = 1e-2 * M3T::Identity();
    noise_cov.block<3,3>(3,3) = 1e-2 * M3T::Identity();
    noise_cov.block<3,3>(6,6) = 1e-3 * M3T::Identity();
    noise_cov.block<3,3>(9,9) = 1e-3 * M3T::Identity();

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
    // if (lidar_data_queue_.size() < 2) return true;

    std::unique_lock<std::mutex> lock(meas_pack_mutex_);
    while (lidar_data_queue_.size() > 1){
        std::shared_ptr<LidarData> lidar = lidar_data_queue_.front();
        double end_time = lidar_data_queue_[1]->time;
        if (end_time > imu_data_queue_.back()->time) return true;
        
        auto meas = std::make_shared<MeasurePack>();
        while (!imu_data_queue_.empty()){
            auto& imu = imu_data_queue_.front();
            if (imu->time > end_time) break;
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
    meas_pack_cond_.notify_all();
    return true;
    
    // lidar_data_queue_.push_back(lidar_);
    // std::unique_lock<std::mutex> lock(meas_pack_mutex_);
    // while (lidar_data_queue_.size() > 0){
    //     std::shared_ptr<LidarData> lidar = lidar_data_queue_.front();
    //     if (imu_data_queue_.back()->time < lidar->time) return true;

    //     auto meas = std::make_shared<MeasurePack>();
    //     while (!imu_data_queue_.empty())
    //     {
    //         auto& imu = imu_data_queue_.front();
    //         if (imu->time > lidar->time) break;
    //         meas->imu_lst->push_back(*imu);
    //         imu_data_queue_.pop_front();
    //     }
    //     if (meas->imu_lst->size() > 0){
    //         meas->cloud = lidar;
    //         meas_pack_.push_back(meas);
    //     }
    //     lidar_data_queue_.pop_front();
    // }
    // lock.unlock();
    // meas_pack_cond_.notify_one();
    // return true;
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

    lidarCompensation(meas, imu_odo);

    buildmapAndUpdate(meas);

    last_meas_ = meas;
}


void MyFastLIO::lidarCompensation(std::shared_ptr<MeasurePack> meas, const std::deque<std::pair<double, Sophus::SE3d>>& tmp_odo)
{
    auto& cloud = meas->cloud->cloud;
    // pcl::io::savePCDFile("/root/workspace/my_fastlio/build/bagsave/before.pcd", *cloud);
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
            double t = meas->cloud->time + p_inj.curvature;
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


    // pcl::io::savePCDFile("/root/workspace/my_fastlio/build/bagsave/after.pcd", *cloud);
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
    Eigen::Quaterniond R0 = Eigen::Quaterniond::FromTwoVectors(ava_acc, V3T::UnitZ());
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

void MyFastLIO::accumalateVMap(CloudVmapPtr& source, CloudVmapPtr& target)
{
    if (target == nullptr){
        throw std::runtime_error("target is null ptr");
        return;
    }
    for (const auto& p : *source){
        target->push_back(p);
    }
}


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

void eigenSparseBlockSet(std::vector<Eigen::Triplet<double>>& triplets, const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>& mat, size_t start_row, size_t start_col)
{
    for(size_t i = 0; i < mat.rows(); i++)
    {
        for (size_t j = 0; j < mat.cols(); j++)
        {
            triplets.emplace_back(start_row + i, start_col + j, mat(i, j));
        }
    }
}

void MyFastLIO::computeHxAndRinvWithCov(std::vector<voxelmap::ptpl>& ptpl, IESKF::ObserveResult& zk, IESKF::ObserveMatrix& H, IESKF::ObserveCovarianceType& Rk_inv, const StateType& Xk,const StateType& X, bool with_cov)
{
    H.resize(ptpl.size(), StateType::DOF);
    // H.setZero();
    Rk_inv.resize(ptpl.size(), ptpl.size());
    // Rk_inv.setZero();
    zk.resize(ptpl.size(), 1);

    omp_set_num_threads(4);

    M3T R = getM(Xk, R).matrix();
    V3T p = getM(Xk, p).matrix();
    M3T R_ItoL = getM(Xk, R_ItoL).matrix();
    V3T p_ItoL = getM(Xk, p_ItoL).matrix();
    std::vector<Eigen::Triplet<double>> triplets_H;
    std::vector<Eigen::Triplet<double>> triplets_R;

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
            // H.block<1, 3>(i, IDX_p * 3) = uT;
            // H.block<1, 3>(i, IDX_R * 3) = -uTR * SO3Hat(R_ItoL * pl + p_ItoL);
            // H.block<1, 3>(i, IDX_R_ItoL * 3) = -uTR * R_ItoL * SO3Hat(pl);
            // H.block<1, 3>(i, IDX_p_ItoL * 3) = uTR;
            eigenSparseBlockSet(triplets_H, uT, i, IDX_p * 3);
            eigenSparseBlockSet(triplets_H, -uTR * SO3Hat(R_ItoL * pl + p_ItoL), i, IDX_R * 3);
            // eigenSparseBlockSet(triplets_H, -uTR * R_ItoL * SO3Hat(pl), i, IDX_R_ItoL * 3);
            // eigenSparseBlockSet(triplets_H, uTR, i, IDX_p_ItoL * 3);

            // if (i < 10) std::cout << H.row(i) << std::endl;

            if (with_cov){
                Eigen::Matrix<double, 1, 6> Huq;
                V3T vec = (R * (R_ItoL * pl + p_ItoL) + p - q);
                Huq.block<1,3>(0, 0) = vec.transpose();
                Huq.block<1,3>(0, 3) = uT;
                Eigen::Matrix<double, 1, 3> Hp = uTR * R_ItoL;

                double R_inv_value = Huq * ptpl[i].plane_cov * Huq.transpose();
                R_inv_value += Hp * ptpl[i].point_cov * Hp.transpose();
                triplets_R.emplace_back(i, i, 1.0 / R_inv_value);
                // Rk_inv(i, i) = Huq * ptpl[i].plane_cov * Huq.transpose();
                // Rk_inv(i, i) += Hp * ptpl[i].point_cov * Hp.transpose();
                // Rk_inv(i, i) = 1 / Rk_inv(i, i);

            }else{
                // Rk_inv(i, i) = 1e4;
                triplets_R.emplace_back(i, i, 1e4);
            }

            zk(i, 0) = uT * (R * (R_ItoL * pl + p_ItoL) + p - q);
        }
        Rk_inv.setFromTriplets(triplets_R.begin(), triplets_R.end());
        H.setFromTriplets(triplets_H.begin(), triplets_H.end());
        // double avarage_R = Rk_inv.diagonal().lpNorm<1>() / Rk_inv.rows();
        // Rk_inv *= 1e4 / avarage_R;
        // std::cout << "\n==============================================\n"; 
        // std::cout << Rk_inv.diagonal().lpNorm<1>() / Rk_inv.rows();
        // std::cout << "\n==============================================\n"; 
    
    }
}


void MyFastLIO::computeHxAndRinvVGICP(IESKF::ObserveResult& zk, IESKF::ObserveMatrix& H, IESKF::ObserveCovarianceType& Rinv, const StateType& Xk,const StateType& X, float cauchy_loss)
{

    auto t_1 = std::chrono::high_resolution_clock::now();
    Sophus::SE3d T_WtoIi(getM(Xk, R), getM(Xk, p));
    Sophus::SE3d T_ItoL(getM(Xk, R_ItoL),  getM(Xk, p_ItoL));
    Sophus::SE3d T_WtoLi = T_WtoIi * T_ItoL;
    vgicp.setSourceTransformation(T_WtoLi.matrix(), true);
    const auto& association = vgicp.getAssociation();

    if (cauchy_loss > 0.0){
        zk.resize(association.size(), 1);
    }else{
        zk.resize(3 * association.size(), 1);
    }

    // for (int i = 0; i < association.size(); i++){
    //     // std::cout << "(" << association[i].N << ", " << association[i].mean.trace() << ", " << association[i].cov.trace() << ") ";
    //     std::cout << "(" << association[i].cov.array() << ") ";
    // }
    // std::cout << std::endl;

    H.resize(zk.rows(), StateType::DOF);
    Rinv.resize(zk.rows(), zk.rows());
    
    // std::vector<M3T> Rinv_cache(zk.rows());
    // std::vector<Eigen::Matrix<double, 3, 12>> H_cache(zk.rows());
    // std::vector<V3T> zk_cache(zk.rows());
    M3T zero_mat = M3T::Zero();

    std::vector<Eigen::Triplet<double>> triplets_H;
    std::vector<Eigen::Triplet<double>> triplets_R;

    M3T R = getM(Xk, R).matrix();
    V3T p = getM(Xk, p);
    M3T R_ItoL = getM(Xk, R_ItoL).matrix();
    V3T p_ItoL = getM(Xk, p_ItoL);
    M3T R_WtoL = R * R_ItoL;
    auto& ptpls = *ptpl_cb;
    ptpls.clear();
    ptpls.resize(association.size());
    if (cauchy_loss > 0){
        // #pragma omp parallel for num_threads(8) schedule(static, 8)
        for (int i = 0; i < association.size(); i ++)
        {
            const gicp::VGICP::Voxel& voxel = vgicp[association[i].voxel_idx];
            V3T p_inIi = R_ItoL * association[i].point + p_ItoL;
            V3T res = voxel.mean - R * p_inIi - p;
            M3T cov = voxel.cov + R_WtoL * association[i].cov * R_WtoL.transpose();
            M3T cov_inv = cov.inverse();
            double error = res.transpose() * cov_inv * res;
            double weight = 1.0f / (1.0f + error / (cauchy_loss * cauchy_loss));
            zk(i, 0) = error;
            // Rinv.block<3,3>(i * 3,i * 3) = weight * cov.inverse();
            // H.block<3, 3>(i * 3, IDX_p * 3) = -M3T::Identity();
            // H.block<3, 3>(i * 3, IDX_R * 3) = R * SO3Hat(R_ItoL * association[i].point + p_ItoL);
            // H.block<3, 3>(i * 3, IDX_R_ItoL) = R * R_ItoL * SO3Hat(association[i].mean);
            // H.block<3, 3>(i * 3, IDX_p_ItoL) = -R_WtoL;

            triplets_R.emplace_back(i, i, 1.0 * weight);
            // eigenSparseBlockSet(triplets_R, cov.ldlt().solve(M3T::Identity()), i * 3, i * 3);
            eigenSparseBlockSet(triplets_H, -2 * res.transpose() * cov_inv * M3T::Identity(), i, IDX_p * 3);
            eigenSparseBlockSet(triplets_H, 2 * res.transpose() * cov_inv * R * SO3Hat(p_inIi), i, IDX_R * 3);
            // eigenSparseBlockSet(triplets_H, R * R_ItoL * SO3Hat(association[i].point), i * 3, IDX_R_ItoL);
            // eigenSparseBlockSet(triplets_H, -R_WtoL, i * 3, IDX_p_ItoL);
        }
    }else{

        auto t0 = std::chrono::high_resolution_clock::now();
        // #pragma omp parallel for num_threads(8) schedule(static)
        for (int i = 0; i < association.size(); i ++)
        {
            const gicp::VGICP::Voxel& voxel = vgicp[association[i].voxel_idx];
            V3T p_inIi = R_ItoL * association[i].point + p_ItoL;
            V3T res = voxel.mean - R * p_inIi - p;
            M3T cov = voxel.cov + R_WtoL * association[i].cov * R_WtoL.transpose();
            zk.segment<3>(i * 3) = res;
            // Rinv.block<3,3>(i * 3,i * 3) = cov.ldlt().solve(M3T::Identity());
            // H.block<3, 3>(i * 3, IDX_p * 3) = -M3T::Identity();
            // H.block<3, 3>(i * 3, IDX_R * 3) = R * SO3Hat(p_inIi);
            // H.block<3, 3>(i * 3, IDX_R_ItoL) = R * R_ItoL * SO3Hat(association[i].point);
            // H.block<3, 3>(i * 3, IDX_p_ItoL) = -R_WtoL;
            // if (i % 200 == 0) std::cout << Rinv.block<3,3>(i * 3,i * 3) << "\n" << std::endl;

            // zk_cache[i] = res;
            // Rinv_cache[i] = cov.ldlt().solve(M3T::Identity());
            // H_cache[i] << -M3T::Identity(), R * SO3Hat(R_ItoL * association[i].point + p_ItoL), zero_mat, zero_mat;

            eigenSparseBlockSet(triplets_R, cov.ldlt().solve(M3T::Identity()), i * 3, i * 3);
            eigenSparseBlockSet(triplets_H, -M3T::Identity(), i * 3, IDX_p * 3);
            eigenSparseBlockSet(triplets_H, R * SO3Hat(p_inIi), i * 3, IDX_R * 3);
            // eigenSparseBlockSet(triplets_H, R * R_ItoL * SO3Hat(association[i].point), i * 3, IDX_R_ItoL);
            // eigenSparseBlockSet(triplets_H, -R_WtoL, i * 3, IDX_p_ItoL);

            // auto& ptpl = ptpls[i];
            // ptpl.center = voxel.mean;
            // ptpl.point = association[i].point;
            // Eigen::JacobiSVD<M3T> svd(cov, Eigen::ComputeFullU);
            // ptpl.normal = svd.matrixU().col(2);  // 对应最小奇异值;

        }

        // std::cout << "time: " << 
        //     float(std::chrono::duration_cast<std::chrono::microseconds>(t0-t_1).count()) / float(1e3) << 
        //     ", " << float(std::chrono::duration_cast<std::chrono::microseconds>(t1-t0).count()) / float(1e3) << 
        //     ", " << float(std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count()) / float(1e3) << std::endl;
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    // for (int i = 0; i < association.size(); i ++)
    // {
    //     zk.segment<3>(i * 3) = zk_cache[i];
    //     Rinv.block<3,3>(i * 3,i * 3) = Rinv_cache[i];
    //     H.block<3, 12>(i * 3, 0) = H_cache[i];
    // }
    Rinv.setFromTriplets(triplets_R.begin(), triplets_R.end());
    H.setFromTriplets(triplets_H.begin(), triplets_H.end());
    auto t2 = std::chrono::high_resolution_clock::now();
}


/* 利用 ikd-tree 构建初始地图 */
// void MyFastLIO::computeHxAndRinvKDTree(std::vector<voxelmap::ptpl>& ptpls, IESKF::ObserveResult& zk, IESKF::ObserveMatrix& H, IESKF::ObserveCovarianceType& Rk_inv, const StateType& Xk,const StateType& X)
// {
//     H.resize(ptpls.size(), StateType::DOF);
//     H.setZero();
//     Rk_inv.resize(ptpls.size(), ptpls.size());
//     Rk_inv.setZero();
//     zk.resize(ptpls.size(), 1);

//     omp_set_num_threads(4);

//     M3T R = getM(Xk, R).matrix();
//     V3T p = getM(Xk, p).matrix();
//     M3T R_ItoL = getM(Xk, R_ItoL).matrix();
//     V3T p_ItoL = getM(Xk, p_ItoL).matrix();
//     // 实测单核心更快
//     // #pragma omp parallel
//     {
//         // #pragma omp for nowait
//         for(int i = 0; i < ptpls.size(); i++)
//         {
//             Eigen::Matrix<double, 1, 3> uT = ptpls[i].normal.transpose();
//             Eigen::Matrix<double, 1, 3> uTR = uT * R;
//             V3T& pl = ptpls[i].point;
//             V3T& q = ptpls[i].center;
//             H.block<1, 3>(i, IDX_p * 3) = uT;
//             H.block<1, 3>(i, IDX_R * 3) = -uTR * SO3Hat(R_ItoL * pl + p_ItoL);
//             // H.block<1, 3>(i, IDX_R_ItoL * 3) = -uTR * R_ItoL * SO3Hat(pl);
//             // H.block<1, 3>(i, IDX_p_ItoL * 3) = uTR;
//             // if (i < 10) std::cout << H.row(i) << std::endl;

//             // Eigen::Matrix<double, 1, 6> Huq;
//             // V3T vec = (R * (R_ItoL * pl + p_ItoL) + p - q);
//             // Huq.block<1,3>(0, 0) = vec.transpose();
//             // Huq.block<1,3>(0, 3) = uT;
//             // Eigen::Matrix<double, 1, 3> Hp = uTR * R_ItoL;


//             Rk_inv(i, i) = 1e4;

//             // zk(i, 0) = uT * vec;
//             zk(i, 0) = uT * (R * (R_ItoL * pl + p_ItoL) + p) + ptpls[i].d;
//             // if (i == 0) std::cout << zk(i,0) << " " << ptpls[i].d << std::endl;
//         }
    
//     }
// }


void MyFastLIO::buildmapAndUpdate(std::shared_ptr<MeasurePack> meas)
{
    auto start = slam_utils::TimerHelper::start();
    static double start_time = 0;
    std::cout << "==============================================================================" << std::endl;
    std::cout << "[measure] Time at " << meas->cloud->time - start_time << " s" << std::endl;
    static int count = 0;

    if (observation_method == POINT_TO_PLANE){
        CloudXYZPtr filtered_cloud(new CloudXYZ);
        pcl::VoxelGrid<pcl::PointXYZ> sample;
        CloudXYZPtr xyz_cloud(new CloudXYZ);
        pcl::copyPointCloud(*meas->cloud->cloud, *xyz_cloud);
        sample.setLeafSize(0.5, 0.5, 0.5);
        sample.setInputCloud(xyz_cloud);
        sample.filter(*filtered_cloud);

        if (kdtree.size() == 0){
            CloudKdTreePtr global_cloud;
            transformPCLCloud2GlobalKDTreeMap(filtered_cloud, global_cloud);
            kdtree.Build(*global_cloud);
            start_time = meas->cloud->time;
        }else{

            CloudKdTreePtr target_cloud(new CloudKdTree);
            transformPCLCloud2GlobalKDTreeMap(filtered_cloud, target_cloud);
            CloudXYZPtr kdtree_add_cloud(new CloudXYZ);

            std::vector<voxelmap::ptpl>& ptpls = *ptpl_cb;
            ptpls.clear();
            for(int i = 0; i < target_cloud->size(); i++){
                const PointXYZ& local_p = filtered_cloud->at(i);
                const PointKdTree& global_p = target_cloud->at(i);
                
                CloudKdTree nears;
                std::vector<float> dists;
                
                kdtree.Nearest_Search(global_p, 5, nears, dists);
                if (nears.size() < 5 || dists.back() > 5) continue;
                Eigen::Matrix<double, 4, 1> plane;
                PointVector pts;
                V3T center = V3T::Zero();
                for(auto p : nears){
                    PointType pt;
                    pt.x = p.x;
                    pt.y = p.y;
                    pt.z = p.z;
                    pts.push_back(pt);
                    center += V3T(p.x, p.y, p.z);
                }
                center /= nears.size();
                double score;
                if (esti_plane<double>(plane, pts, plane_threshold_kdtree, score)){
                    voxelmap::ptpl ptpl;
                    ptpl.normal = plane.head(3);
                    ptpl.center = center;
                    ptpl.point = V3T(local_p.x, local_p.y, local_p.z);
                    // ptpl.d = (V3T(global_p.x, global_p.y, global_p.z) - center).dot(ptpl.normal);
                    ptpl.d = plane(3);
                    ptpl.score = score;
                    float pd2 = plane(0) * global_p.x + plane(1) * global_p.y + plane(2) * global_p.z + plane(3);
                    double s = 1 - 0.9 * fabs(pd2) / sqrt(ptpl.point.norm());
                    if(s < 0.9 ){
                        continue;
                    }
                    ptpls.push_back(ptpl);
                    PointXYZ p_kd;
                    p_kd.x = local_p.x; p_kd.y = local_p.y; p_kd.z = local_p.z;
                    kdtree_add_cloud->push_back(p_kd);
                }
            }

            std::cout << "[measure] " << "ori: " << filtered_cloud->size() << ", ptpl: " << ptpls.size() << ", add: " << kdtree_add_cloud->size() << std::endl;

            /* 构建 Hx 和 R */
            kf.computeHxAndRinv = [this](IESKF::ObserveResult& zk, IESKF::ObserveMatrix& H, IESKF::ObserveCovarianceType& Rinv, const StateType& Xk,const StateType& X){
                this->computeHxAndRinvWithCov(*ptpl_cb, zk, H, Rinv, Xk, X, false);
            };
            // std::cout << "cov: \n" << kf.error_cov << std::endl;
            auto info = kf.update(NUM_MAX_ITERATIONS, 1e-8);

            std::cout << "[update] cost change: " << info.begin_cost << " -> " << info.end_cost << ", change: " << info.begin_cost - info.end_cost << std::endl;
            std::cout << "[update] average iter time: " << info.avarave_iter_time << " ms" << std::endl;
            std::cout << "[update] total iteration time(" << info.iter_times << " nums): " << info.cost_time << " ms" << std::endl;

            /* ikdtree 添加点云 */
            CloudKdTreePtr global_cloud(new CloudKdTree);
            transformPCLCloud2GlobalKDTreeMap(kdtree_add_cloud, global_cloud);
            kdtree.Add_Points(*global_cloud, true);

            // if (meas->cloud->time - start_time >= 70 && meas->cloud->time - start_time <= 75){
            //     CloudXYZ map_tosave;
            //     map_tosave.height = 1;
            //     map_tosave.header.frame_id = "map";
            //     KDTree::PointVector(). swap(kdtree.PCL_Storage);
            //     kdtree.flatten(kdtree.Root_Node, kdtree.PCL_Storage, NOT_RECORD);
            //     // map_tosave.points = kdtree.PCL_Storage;
            //     for(int i = 0; i < kdtree.PCL_Storage.size(); i++){
            //         PointXYZ p;
            //         p.x = kdtree.PCL_Storage[i].x;
            //         p.y = kdtree.PCL_Storage[i].y;
            //         p.z = kdtree.PCL_Storage[i].z;
            //         map_tosave.push_back(p);
            //     }
            //     map_tosave.width = kdtree.PCL_Storage.size();
            //     pcl::io::savePCDFile("/root/workspace/my_fastlio/build/bagsave/kdmap.pcd", map_tosave);
            // }

            map_for_publish = meas->cloud->cloud;

            // CloudVmapPtr global_vmap(new CloudVMap);
            // transformPCLCloud2GlobalVMap(kdtree_add_cloud, global_vmap);
            // accumalateVMap(global_vmap, init_vmap);
        }

    }else if (observation_method == POINT_TO_VOXELMAP){

        // auto start = slam_utils::TimerHelper::start();
        // CloudXYZPtr filtered_cloud(new CloudXYZ);
        // pcl::VoxelGrid<pcl::PointXYZ> sample;
        // CloudXYZPtr xyz_cloud(new CloudXYZ);
        // pcl::copyPointCloud(*meas->cloud->cloud, *xyz_cloud);
        // sample.setLeafSize(0.8, 0.8, 0.8);
        // sample.setInputCloud(xyz_cloud);
        // sample.filter(*filtered_cloud);

        // map_for_publish = meas->cloud->cloud;

        // CloudVmapPtr local_cloud;
        // Sophus::SE3d T_WtoIi(getM(kf.normal_state, R), getM(kf.normal_state, p));
        // Sophus::SE3d T_ItoL(getM(kf.normal_state, R_ItoL), getM(kf.normal_state, p_ItoL)); 
        // Sophus::SE3d T_WtoLi = T_WtoIi * T_ItoL;
        // pclCloud2VoxelMapCloud(filtered_cloud, local_cloud);

        // std::vector<V3T> point_world(filtered_cloud->size());
        // for(int i = 0 ; i < filtered_cloud->size(); i++){
        //     point_world[i] = T_WtoLi * filtered_cloud->at(i).getVector3fMap().cast<double>();
        // }

        
        // std::vector<voxelmap::ptpl>& ptpl = *ptpl_cb;
        // ptpl.clear();
        // std::vector<V3T> non_match;
        // voxelmap::BuildResidualListOMP(*vmap, max_voxel_size, 2, max_layer, *local_cloud, point_world, ptpl, non_match);

        // std::cout << "[measure] " << "ori: " << filtered_cloud->size() << ", ptpl: " << ptpl.size() << std::endl;

        // kf.computeHxAndRinv = [this](IESKF::ObserveResult& zk, IESKF::ObserveMatrix& H, IESKF::ObserveCovarianceType& Rinv, const StateType& Xk,const StateType& X){
        //     this->computeHxAndRinvWithCov(*ptpl_cb, zk, H, Rinv, Xk, X);
        // };
        // IESKFUpdateInfo info = kf.update(NUM_MAX_ITERATIONS, 1e-4);
        // std::cout << "[update] cost change: " << info.begin_cost << " -> " << info.end_cost << ", change: " << info.begin_cost - info.end_cost << std::endl;
        // std::cout << "[update] iteration time: " << info.cost_time << " ms" << std::endl;
       
        // auto start_vmap_update = slam_utils::TimerHelper::start();
        // CloudVmapPtr cloud_global(new CloudVMap);
        // transformPCLCloud2GlobalVMap(filtered_cloud, cloud_global);
        // voxelmap::updateVoxelMap(*cloud_global, max_voxel_size, 
        //         max_layer, layer_size, max_point_size, 
        //         max_cov_point_size, plane_threshold_vmap, *vmap);
        // std::cout << "[update] voxmap update cost: " << slam_utils::TimerHelper::end(start_vmap_update) << " ms" << std::endl;
        
        // std::cout << "[update] total time: " << slam_utils::TimerHelper::end(start) << " ms" << std::endl;
    }else if (observation_method == METHOD_VGICP){
        CloudXYZPtr filtered_cloud(new CloudXYZ);
        pcl::VoxelGrid<pcl::PointXYZ> sample;
        CloudXYZPtr xyz_cloud(new CloudXYZ);
        CloudXYZPtr tmp_filter_cloud(new CloudXYZ);
        pcl::copyPointCloud(*meas->cloud->cloud, *xyz_cloud);
        sample.setLeafSize(0.5, 0.5, 0.5);
        sample.setInputCloud(xyz_cloud);
        sample.filter(*filtered_cloud);
        
        static bool vgicp_pre_map_inited = false;

        if (kdtree.size() == 0){
            CloudKdTreePtr global_cloud;
            transformPCLCloud2GlobalKDTreeMap(filtered_cloud, global_cloud);
            kdtree.Build(*global_cloud);
            start_time = meas->cloud->time;
        }else if (!vgicp_pre_map_inited){

            CloudKdTreePtr target_cloud(new CloudKdTree);
            transformPCLCloud2GlobalKDTreeMap(filtered_cloud, target_cloud);
            CloudXYZPtr kdtree_add_cloud(new CloudXYZ);

            std::vector<voxelmap::ptpl>& ptpls = *ptpl_cb;
            ptpls.clear();
            for(int i = 0; i < target_cloud->size(); i++){
                const PointXYZ& local_p = filtered_cloud->at(i);
                const PointKdTree& global_p = target_cloud->at(i);
                
                CloudKdTree nears;
                std::vector<float> dists;
                
                kdtree.Nearest_Search(global_p, 5, nears, dists);
                if (nears.size() < 5 || dists.back() > 5) continue;
                Eigen::Matrix<double, 4, 1> plane;
                PointVector pts;
                V3T center = V3T::Zero();
                for(auto p : nears){
                    PointType pt;
                    pt.x = p.x;
                    pt.y = p.y;
                    pt.z = p.z;
                    pts.push_back(pt);
                    center += V3T(p.x, p.y, p.z);
                }
                center /= nears.size();
                double score;
                if (esti_plane<double>(plane, pts, plane_threshold_kdtree, score)){
                    voxelmap::ptpl ptpl;
                    ptpl.normal = plane.head(3);
                    ptpl.center = center;
                    ptpl.point = V3T(local_p.x, local_p.y, local_p.z);
                    // ptpl.d = (V3T(global_p.x, global_p.y, global_p.z) - center).dot(ptpl.normal);
                    ptpl.d = plane(3);
                    ptpl.score = score;
                    float pd2 = plane(0) * global_p.x + plane(1) * global_p.y + plane(2) * global_p.z + plane(3);
                    double s = 1 - 0.9 * fabs(pd2) / sqrt(ptpl.point.norm());
                    if(s < 0.9 ){
                        continue;
                    }
                    ptpls.push_back(ptpl);
                    PointXYZ p_kd;
                    p_kd.x = local_p.x; p_kd.y = local_p.y; p_kd.z = local_p.z;
                    kdtree_add_cloud->push_back(p_kd);
                }
            }

            std::cout << "[measure] " << "ori: " << filtered_cloud->size() << ", ptpl: " << ptpls.size() << ", add: " << kdtree_add_cloud->size() << std::endl;

            /* 构建 Hx 和 R */
            kf.computeHxAndRinv = [this](IESKF::ObserveResult& zk, IESKF::ObserveMatrix& H, IESKF::ObserveCovarianceType& Rinv, const StateType& Xk,const StateType& X){
                this->computeHxAndRinvWithCov(*ptpl_cb, zk, H, Rinv, Xk, X, false);
            };
            // std::cout << "cov: \n" << kf.error_cov << std::endl;
            auto info = kf.update(NUM_MAX_ITERATIONS, 1e-8);

            std::cout << "[update] cost change (" << info.iter_times << " iter nums) :" << info.begin_cost << " -> " << info.end_cost << ", change: " << info.begin_cost - info.end_cost << std::endl;
            std::cout << "[update] total measure time: " << info.measure_time * info.iter_times << " ms, avarage: " << info.measure_time << " ms" << std::endl; 
            std::cout << "[update] total iteration time: " << info.cost_time << " ms, average: " << info.avarave_iter_time << " ms" << std::endl;

            /* ikdtree 添加点云 */
            CloudKdTreePtr global_cloud(new CloudKdTree);
            transformPCLCloud2GlobalKDTreeMap(kdtree_add_cloud, global_cloud);
            kdtree.Add_Points(*global_cloud, true);
            map_for_publish = meas->cloud->cloud;

            if (meas->cloud->time - start_time > vgpicp_pre_map_time){
                CloudXYZPtr vgicp_premap(new CloudXYZ);
                vgicp_premap->height = 1;
                vgicp_premap->header.frame_id = "map";
                KDTree::PointVector(). swap(kdtree.PCL_Storage);
                kdtree.flatten(kdtree.Root_Node, kdtree.PCL_Storage, NOT_RECORD);
                // map_tosave.points = kdtree.PCL_Storage;
                for(int i = 0; i < kdtree.PCL_Storage.size(); i++){
                    PointXYZ p;
                    p.x = kdtree.PCL_Storage[i].x;
                    p.y = kdtree.PCL_Storage[i].y;
                    p.z = kdtree.PCL_Storage[i].z;
                    vgicp_premap->push_back(p);
                }
                vgicp_premap->width = kdtree.PCL_Storage.size();

                // Sophus::SE3d T_WtoIi(getM(kf.normal_state, R), getM(kf.normal_state, p));
                // Sophus::SE3d T_ItoL(getM(kf.normal_state, R_ItoL),  getM(kf.normal_state, p_ItoL));
                // Sophus::SE3d T_WtoLi = T_WtoIi * T_ItoL;
                vgicp.setSourceCloud(vgicp_premap);
                vgicp.setSourceTransformation(Eigen::Matrix4d::Identity(), false);
                vgicp.pushSourceIntoMap();
                vgicp_pre_map_inited = true;
            }

            // auto& voxeles = vgicp.getVoxeles();
            // std::cout << "voxels count:" << voxeles.size() << std::endl;
            // CloudXYZ cloud_vgicp;
            // for (const auto& [idx, voxel] : voxeles)
            // {
            //     pcl::PointXYZ p;
            //     p.getVector3fMap() = idx.cast<float>();
            //     cloud_vgicp.push_back(p);
            // }
        }else{
            
            auto start_set_source = slam_utils::TimerHelper::start();
            auto& voxels = vgicp.getVoxeles();
            float vsize = vgicp.getVoxelSize();
            vgicp_center->clear();
            for (const auto& [idx, voxel]: voxels){
                pcl::PointXYZ cent_p;
                cent_p.getVector3fMap() = idx.getCenter(vsize).cast<float>();
                vgicp_center->push_back(cent_p);
            }
            vgicp.setSourceCloud(filtered_cloud);
            std::cout << "[preprocessing] compute cloud convariance time: " << slam_utils::TimerHelper::end(start_set_source) << " ms" << std::endl;
            // if (vgicp.empty()){
            //     Sophus::SE3d T_WtoIi(getM(kf.normal_state, R), getM(kf.normal_state, p));
            //     Sophus::SE3d T_ItoL(getM(kf.normal_state, R_ItoL),  getM(kf.normal_state, p_ItoL));
            //     Sophus::SE3d T_WtoLi = T_WtoIi * T_ItoL;
            //     vgicp.setSourceTransformation(Eigen::Matrix4d::Identity(), false);
            //     vgicp.pushSourceIntoMap();
            //     vgicp_pre_map_inited = true;
            // }else{
                
                auto start_construct_asso = slam_utils::TimerHelper::start();
                Sophus::SE3d T_WtoIi_(getM(kf.normal_state, R), getM(kf.normal_state, p));
                Sophus::SE3d T_ItoL_(getM(kf.normal_state, R_ItoL),  getM(kf.normal_state, p_ItoL));
                Sophus::SE3d T_WtoLi_ = T_WtoIi_ * T_ItoL_;
                vgicp.setSourceTransformation(T_WtoLi_.matrix(), false);
                std::cout << "[update] association time: " << slam_utils::TimerHelper::end(start_construct_asso) << " ms" << std::endl;

                kf.computeHxAndRinv = [this](IESKF::ObserveResult &zk, IESKF::ObserveMatrix &H, IESKF::ObserveCovarianceType &R, const StateType &Xk, const StateType &X){
                    this->computeHxAndRinvVGICP(zk, H, R, Xk, X);
                };
                auto info = kf.update(NUM_MAX_ITERATIONS, 1e-7);
                std::cout << "[update] observation num: " << info.H_rows / 3 << std::endl;
                std::cout << "[update] cost change (" << info.iter_times << " iter nums): " << info.begin_cost << " -> " << info.end_cost << ", change: " << info.begin_cost - info.end_cost << std::endl;
                std::cout << "[update] total construct H and R time: " << info.measure_time * info.iter_times << " ms, avarage: " << info.measure_time << " ms" << std::endl; 
                std::cout << "[update] total iteration time: " << info.cost_time << " ms, average: " << info.avarave_iter_time << " ms" << std::endl;

                auto start_push_points = slam_utils::TimerHelper::start();
                Sophus::SE3d T_WtoIi(getM(kf.normal_state, R), getM(kf.normal_state, p));
                Sophus::SE3d T_ItoL(getM(kf.normal_state, R_ItoL),  getM(kf.normal_state, p_ItoL));
                Sophus::SE3d T_WtoLi = T_WtoIi * T_ItoL;
                // Sophus::SE3d T_WtoLi = T_ItoL;
                vgicp.setSourceTransformation(T_WtoLi.matrix(), false);
                vgicp.pushSourceIntoMap();
                std::cout << "[mapping] total time: " << slam_utils::TimerHelper::end(start_push_points) << " ms" << std::endl;;

                // auto& voxeles = vgicp.getVoxeles();
                // std::cout << "voxels count:" << voxeles.size() << std::endl;
                // CloudXYZ cloud_vgicp;
                // for (const auto& [idx, voxel] : voxeles)
                // {
                //     pcl::PointXYZ p;
                //     p.getVector3fMap() = idx.cast<float>();
                //     cloud_vgicp.push_back(p);
                // }

                map_for_publish = meas->cloud->cloud;
            // }

        }

     }
    std::cout << std::endl;

}

void MyFastLIO::lioThread()
{
    staticStateIMUInitialize();
    while (true)
    {
        handlePack();
        if (updateCallback){
            auto start_callback = slam_utils::TimerHelper::start();
            ret_info->time = last_meas_->cloud->time;
            Sophus::SE3d T_WtoIi = Sophus::SE3d(getM(kf.normal_state, R), getM(kf.normal_state, p));
            Sophus::SE3d T_WtoLi = T_WtoIi * Sophus::SE3d(getM(kf.normal_state, R_ItoL), getM(kf.normal_state, p_ItoL));
            ret_info->pose = T_WtoLi;
            // ret_info->pose = Sophus::SE3d(getM(kf.normal_state, R), getM(kf.normal_state, p));
            ret_info->vel = getM(kf.normal_state, v);
            ret_info->map = map_for_publish;
            ret_info->filtered_indices = map_indices;
            ret_info->vmap = vmap;
            ret_info->ptpl = ptpl_cb;
            ret_info->vgicp_center = vgicp_center;
            updateCallback.value()(ret_info);
            // std::cout << "q: " << getM(kf.normal_state, R).unit_quaternion().coeffs().transpose() << std::endl;
            // std::cout << "p: " << getM(kf.normal_state, p).transpose() << std::endl;
            // std::cout << "q_ItoL: " << getM(kf.normal_state, R_ItoL).unit_quaternion().coeffs().transpose() << std::endl;
            // std::cout << "p_ItoL: " << getM(kf.normal_state, p_ItoL).transpose() << std::endl;
            // std::cout << "v: " << getM(kf.normal_state, v).transpose() << std::endl;
            // std::cout << "bw: " << getM(kf.normal_state, bw).transpose() << std::endl;
            // std::cout << "ba: " << getM(kf.normal_state, ba).transpose() << std::endl;
            // std::cout << "g: " << getM(kf.normal_state, g).transpose() << std::endl;
            // exit(0);
            std::cout << "[user] callback time:" << slam_utils::TimerHelper::end(start_callback) << " ms" << std::endl;
            std::this_thread::sleep_for(1ms);
        }
    }
}
    
} // namespace my_fastlio


