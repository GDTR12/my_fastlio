#include "my_fastlio/my_fastlio.hpp"
#include "ieskf/ieskf.hpp"
#include "my_fastlio/my_fastlio_param.hpp"
#include <functional>
#include <memory>
#include <mutex>
#include <sophus/se3.hpp>
#include <stdexcept>



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
    imu_data_mutex_.unlock();


}

MyFastLIO::~MyFastLIO()
{
    std::unique_lock<std::mutex> lock1(imu_data_mutex_);
    meas_pack_.clear();
    lock1.unlock();

    std::unique_lock<std::mutex> lock2(imu_data_mutex_);
    imu_data_queue_.clear();
    lock2.unlock();
}

bool MyFastLIO::imuAsyncPushImu(std::shared_ptr<ImuData> data)
{
    std::unique_lock<std::mutex> lock(imu_data_mutex_);
    imu_data_queue_.push_back(data);
    lock.unlock();
    imu_data_cond_.notify_one();
    return true;
}

bool MyFastLIO::lidarAsyncPushLidar(std::shared_ptr<LidarData> lidar)
{
    std::unique_lock<std::mutex> lock2(imu_data_mutex_);
    imu_data_cond_.wait(lock2, [this, &lidar]{return imu_data_queue_.back()->time > lidar->time;});
    std::unique_lock<std::mutex> lock(meas_pack_mutex_);
    auto meas = std::make_shared<MeasurePack>();
    while (!imu_data_queue_.empty())
    {
        auto& imu = imu_data_queue_.front();
        if (imu->time > lidar->time) break;
        meas->imu_lst.push_back(imu);
        imu_data_queue_.pop_front();
    }
    meas->cloud = lidar;
    meas_pack_.push_back(meas);
    lock.unlock();
    lock2.unlock();
    meas_pack_cond_.notify_one();
    return true;
}

void MyFastLIO::imuPropagate()
{
    IESKF::ErrorStateType delta; 
    delta.setZero();
    StateType& X = kf.normal_state;

    std::unique_lock<std::mutex> lock_pack(meas_pack_mutex_);
    meas_pack_cond_.wait(lock_pack, [this] { return !meas_pack_.empty(); });
    std::shared_ptr<MeasurePack> meas = meas_pack_.front();
    meas_pack_.pop_front();
    lock_pack.unlock();

    double last_t = -1;
    int idx;
    if (last_meas_ != nullptr){
        last_t = last_meas_->imu_lst.back()->time;
        idx = 0;
    }else{
        // 第一个 pack 包
        last_t = meas->imu_lst.front()->time;
        idx = 1;
    }
    for (int i = idx; i < meas->imu_lst.size(); i++)
    {
        auto curr_imu = meas->imu_lst[i];
        double dt = curr_imu->time - last_t;
        const V3T& a = curr_imu->a;
        const V3T& w = curr_imu->w;
        delta.segment<3>(IDX_p * 3) = getM(X, v) +  0.5 * dt * (getM(X, R_ItoL).matrix() * (a - getM(X, ba)) + getM(X, g));
        delta.segment<3>(IDX_R * 3) = w - getM(X, bw);
        delta.segment<3>(IDX_v * 3) = getM(X, R).matrix() * (a - getM(X, ba)) + getM(X, g);
        kf.predict(delta);
    }
    last_meas_ = meas;
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
    std::cout << "Param noise cov:\n" << kf.noise_cov << std::endl;
    std::cout << "Param error cov:\n" << kf.error_cov << std::endl;
    std::cout << std::endl;
}

void MyFastLIO::lioThread()
{
    while (true)
    {
        imuPropagate();

        if (updateCallback){
            std::shared_ptr<CallbackInfo> ret_info(new CallbackInfo);
            std::cout << last_meas_->cloud << std::endl;
            ret_info->time = last_meas_->cloud->time;
            ret_info->pose = Sophus::SE3d(getM(kf.normal_state, R), getM(kf.normal_state, p));
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


