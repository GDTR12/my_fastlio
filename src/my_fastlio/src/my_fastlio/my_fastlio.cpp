#include "my_fastlio/my_fastlio.hpp"
#include "Eigen/src/Core/Matrix.h"
#include "Eigen/src/Core/util/Memory.h"
#include "Eigen/src/Geometry/Quaternion.h"
#include "ieskf/ieskf.hpp"
#include "manifold/so3.hpp"
#include "my_fastlio/my_fastlio_param.hpp"
#include "pcl/common/io.h"
#include "pcl/common/transforms.h"
#include "pcl/impl/point_types.hpp"
#include "pcl_conversions/pcl_conversions.h"
#include <functional>
#include <iomanip>
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
    error_cov.setIdentity();
    error_cov.block<3,3>(6,6) = 1e-5 * M3T::Identity();
    error_cov.block<3,3>(9,9) = 1e-5 * M3T::Identity();
    error_cov.block<3,3>(15, 15) = 1e-4 * M3T::Identity();
    error_cov.block<3,3>(18, 18) = 1e-3 * M3T::Identity();
    error_cov.block<2,2>(21, 21) = 1e-5 * M2T::Identity();
    noise_cov.setIdentity();
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
    std::cout << "imu propagate cost: " << slam_utils::TimerHelper::end(start) << " ms" << std::endl; 

    lidarCompensation(meas, imu_odo);

    last_meas_ = meas;
}


void MyFastLIO::lidarCompensation(std::shared_ptr<MeasurePack> meas, const std::deque<std::pair<double, Sophus::SE3d>>& tmp_odo)
{
    auto& cloud = meas->cloud->cloud;
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
    std::cout << "undistort " << count << " points(" << float(add_duration)/1e3 <<" ms): " << double(count) / double(cloud->points.size()) * 100 << "% avarage change: " << ava_change.transpose() / double(count) << std::endl;
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
    fx.block<3,3>(IDX_R * 3, IDX_bw * 3) = -dt * Sophus::SO3d::leftJacobianInverse(dt * (getM(u, w) - getM(X, bw)));
    fx.block<3,3>(IDX_R * 3, IDX_R * 3) = Sophus::SO3d::exp(-dt * (getM(u, w) - getM(X, bw))).matrix();
    fx.block<3,3>(IDX_v * 3, IDX_R * 3) = -dt * getM(X, R).matrix() * Sophus::SO3d::hat(getM(u, a) - getM(X, ba));
    fx.block<3,3>(IDX_v * 3, IDX_ba * 3) =  -dt * getM(X, R).matrix();
    // fx.block<3,2>(IDX_v * 3, IDX_g * 3) = Eigen::Matrix<double, 3, 2>::Zero();
    Eigen::Matrix<double,2,1> delta_g = delta_x.segment<2>(IDX_g * 3);
    fx.block<3,2>(IDX_v * 3, IDX_g * 3) = dt * std::get<7>(X.data).boxplusJacobian(delta_g);

    fw.block<3,3>(IDX_R * 3, IDX_nw * 3) = -dt * Sophus::SO3d::leftJacobianInverse(dt * (getM(u, w) - getM(X, bw)));
    fw.block<3,3>(IDX_v * 3, IDX_na * 3) = -dt * getM(X, R).matrix();
    fw.block<3,3>(IDX_bw * 3, IDX_nbw * 3) = -dt * M3T::Identity();
    fw.block<3,3>(IDX_ba * 3, IDX_nba * 3) = -dt * M3T::Identity();
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

bool MyFastLIO::esti_plane(V4T &pca_result, const KDTree::PointVector &point, const float &threshold)
{
    Eigen::Matrix<double, plane_N_search, 3> A;
    Eigen::Matrix<double, plane_N_search, 1> b;
    A.setZero();
    b.setOnes();
    b *= -1.0f;

    for (int j = 0; j < plane_N_search; j++)
    {
        A(j,0) = point[j].x;
        A(j,1) = point[j].y;
        A(j,2) = point[j].z;
    }

    Eigen::Matrix<double, 3, 1> normvec = A.colPivHouseholderQr().solve(b);

    double n = normvec.norm();
    pca_result(0) = normvec(0) / n;
    pca_result(1) = normvec(1) / n;
    pca_result(2) = normvec(2) / n;
    pca_result(3) = 1.0 / n;

    for (int j = 0; j < plane_N_search; j++)
    {
        if (fabs(pca_result(0) * point[j].x + pca_result(1) * point[j].y + pca_result(2) * point[j].z + pca_result(3)) > threshold)
        {
            return false;
        }
    }
    return true;
}
void MyFastLIO::buildmapAndUpdate()
{
    CloudPtr new_cloud(new CloudT);
    Sophus::SE3d T_L0toLi(getM(kf.normal_state, R), getM(kf.normal_state, p));
    pcl::transformPointCloud(*last_meas_->cloud->cloud, *new_cloud, T_L0toLi.matrix().cast<float>());
    if (map->empty()){
        *map += *new_cloud;
        ikd_tree.Build(map->points);
    }else{
        auto start = slam_utils::TimerHelper::start();
        CloudXYZPtr filtered_cloud(new CloudXYZ);
        pcl::RandomSample<PointXYZ> sample;
        CloudXYZPtr xyz_cloud(new CloudXYZ);
        pcl::copyPointCloud(*new_cloud, *xyz_cloud);
        sample.setSample(frame_residual_count);
        sample.setInputCloud(xyz_cloud);
        sample.filter(*filtered_cloud);
        // std::cout << "sample:" << filtered_cloud->size() << ", cost: " << slam_utils::TimerHelper::end(start) << " ms" << std::endl;

        for (auto p : *filtered_cloud)
        {
            PointT center; 
            pcl::copyPoint(p, center);
            KDTree::PointVector point_lst;
            std::vector<float> dists;
            ikd_tree.Nearest_Search(center, plane_N_search, point_lst, dists);
            
            V4T plane_coeffs;
            if (esti_plane(plane_coeffs, point_lst, 0.1)){

            }
        }
        // *map += *new_cloud;
        // ikd_tree.Add_Points(new_cloud->points, false);
    }

}

void MyFastLIO::lioThread()
{
    staticStateIMUInitialize();
    while (true)
    {
        handlePack();
        buildmapAndUpdate();
        if (updateCallback){
            std::shared_ptr<CallbackInfo> ret_info(new CallbackInfo);
            ret_info->time = last_meas_->cloud->time;
            ret_info->pose = Sophus::SE3d(getM(kf.normal_state, R), getM(kf.normal_state, p));
            ret_info->vel = getM(kf.normal_state, v);
            ret_info->map = map;
            updateCallback.value()(ret_info);
        }
        // std::unique_lock<std::mutex> lock_lidar(lidar_data_mutex_);
        // lidar_data_cond_.wait(lock_lidar, [this] { return !lidar_data_queue_.empty(); });
        
        // CloudPtr lidar_data = lidar_data_queue_.front();
        // lidar_data_queue_.pop_front();
        // lock_lidar.unlock();
        // std::cout << "Processing lidar data: " << lidar_data->header.stamp << std::endl;

        // // Process the IMU data
        // std::cout << "Processing IMU data: " << imu_data->time << std::endl;
        // std::cout << "Angular velocity: " << imu_data->w.transpose() << std::endl;
        // std::cout << "Linear acceleration: " << imu_data->a.transpose() << std::endl;
    }
}
    
} // namespace my_fastlio


