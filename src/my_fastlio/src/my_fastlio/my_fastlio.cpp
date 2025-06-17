#include "my_fastlio/my_fastlio.hpp"
#include "Eigen/src/Core/Matrix.h"
#include "ieskf/ieskf.hpp"
#include "manifold/so3.hpp"
#include "my_fastlio/my_fastlio_param.hpp"
#include "pcl/common/io.h"
#include "pcl/common/transforms.h"
#include "pcl_conversions/pcl_conversions.h"
#include <functional>
#include <memory>
#include <mutex>
#include <sophus/se3.hpp>
#include <sophus/so3.hpp>
#include <stdexcept>
#include <pcl/registration/ndt.h>
#include <pcl/filters/random_sample.h>
#include <utility>



namespace my_fastlio
{


MyFastLIO::MyFastLIO()
{

    kf.computeFxAndFw = std::bind(&MyFastLIO::computeFxAndFw, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);

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
    delta *= dt;
    kf.predict(delta);
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

    lidarCompensation(meas, imu_odo);

    last_meas_ = meas;
}


void MyFastLIO::lidarCompensation(std::shared_ptr<MeasurePack> meas, const std::deque<std::pair<double, Sophus::SE3d>>& tmp_odo)
{
    CloudT cloud;
    // std::sort(cloud.points.begin(), cloud.points.end(), [](){

    // });
    
}

// TODO:
void MyFastLIO::computeFxAndFw(IESKF::ErrorPropagateFx& fx, IESKF::ErrorPropagateFw& fw, const StateType& X)
{
    fx.setIdentity();
    fw.setIdentity();
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


