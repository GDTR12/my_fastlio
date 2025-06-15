#pragma once
#include <iostream>
#include <Eigen/Core>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <deque>
#include <mutex>
#include <thread>
#include <condition_variable>

#include "manifold/manifold.hpp"
#include "manifold/s2.hpp"
#include "manifold/so3.hpp"
#include "my_fastlio/my_fastlio_param.hpp"
#include "ieskf/ieskf.hpp"


namespace my_fastlio
{

constexpr char rot_str[] = "r";
constexpr char g_str[] = "g";

using namespace ieskf;

namespace mfd = manifold;

class MyFastLIO
{
public:
    typedef mfd::Manifolds<mfd::SO3, mfd::Vector<3>, mfd::Vector<3>, mfd::Vector<3>, mfd::Vector<3>,
                           mfd::S2, mfd::SO3, mfd::Vector<3>> StateType;
    MyFastLIO();
    ~MyFastLIO();

    bool imuAsyncPushImu(std::shared_ptr<const ImuData> data);
    bool lidarAsyncPushLidar(CloudPtr& data);

private:

    void lioThread();

    ieskf::IESKF<StateType> ieskf;

    std::shared_ptr<std::thread> main_thread_;

    std::deque<CloudPtr> lidar_data_queue_;
    std::deque<std::shared_ptr<const ImuData>> imu_data_queue_;
    std::mutex lidar_data_mutex_, imu_data_mutex_;
    std::condition_variable lidar_data_cond_, imu_data_cond_;
};

}

