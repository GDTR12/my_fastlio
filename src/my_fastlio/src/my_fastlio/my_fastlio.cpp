#include "my_fastlio/my_fastlio.hpp"


namespace my_fastlio
{

MyFastLIO::MyFastLIO()
{
    // Initialize the data queues and mutexes
    lidar_data_queue_.clear();
    imu_data_queue_.clear();
    lidar_data_mutex_.unlock();
    imu_data_mutex_.unlock();

    main_thread_ = std::make_shared<std::thread>(&MyFastLIO::lioThread, this);
    pthread_setname_np(main_thread_->native_handle(), "myfastlio_thread");
    main_thread_->detach();
}

MyFastLIO::~MyFastLIO()
{
    std::unique_lock<std::mutex> lock(lidar_data_mutex_);
    lidar_data_queue_.clear();
    lock.unlock();
    
    std::unique_lock<std::mutex> lock2(imu_data_mutex_);
    imu_data_queue_.clear();
    lock2.unlock();
}

bool MyFastLIO::imuAsyncPushImu(std::shared_ptr<const ImuData> data)
{
    std::unique_lock<std::mutex> lock(imu_data_mutex_);
    imu_data_queue_.push_back(data);
    lock.unlock();
    imu_data_cond_.notify_one();        
    return true;
}

bool MyFastLIO::lidarAsyncPushLidar(CloudPtr& data)
{
    std::unique_lock<std::mutex> lock(lidar_data_mutex_);
    lidar_data_queue_.push_back(data);
    lock.unlock();
    lidar_data_cond_.notify_one();        
    return true;
}


void MyFastLIO::lioThread()
{

    while (true)
    {
        // std::unique_lock<std::mutex> lock_lidar(lidar_data_mutex_);
        // lidar_data_cond_.wait(lock_lidar, [this] { return !lidar_data_queue_.empty(); });
        
        // CloudPtr lidar_data = lidar_data_queue_.front();
        // lidar_data_queue_.pop_front();
        // lock_lidar.unlock();
        // std::cout << "Processing lidar data: " << lidar_data->header.stamp << std::endl;

        // std::unique_lock<std::mutex> lock_imu(imu_data_mutex_);
        // imu_data_cond_.wait(lock_imu, [this] { return !imu_data_queue_.empty(); });
        // std::shared_ptr<const ImuData> imu_data = imu_data_queue_.front();
        // imu_data_queue_.pop_front();
        // lock_imu.unlock();
        // // Process the IMU data
        // std::cout << "Processing IMU data: " << imu_data->time << std::endl;
        // std::cout << "Angular velocity: " << imu_data->w.transpose() << std::endl;
        // std::cout << "Linear acceleration: " << imu_data->a.transpose() << std::endl;
    }
}
    
} // namespace my_fastlio


