#pragma once
#include <functional>
#include <iostream>
#include <Eigen/Core>
#include <memory>
#include <optional>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <deque>
#include <mutex>
#include <thread>
#include <condition_variable>
#include <sophus/se3.hpp>

#include "manifold/manifold.hpp"
#include "manifold/s2.hpp"
#include "manifold/so3.hpp"
#include "my_fastlio/my_fastlio_param.hpp"
#include "ieskf/ieskf.hpp"

#define getM(X, I) std::get<(IDX_##I)>((X.data))

namespace my_fastlio
{

constexpr char rot_str[] = "r";
constexpr char g_str[] = "g";

enum
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

using namespace ieskf;

namespace mfd = manifold;

struct LIOdometryInfo
{
    double time;
    Sophus::SE3d pose;
    CloudPtr new_cloud;
};

class MyFastLIO
{
public:
    typedef mfd::Manifolds<mfd::Vector<3>, mfd::SO3, mfd::SO3, mfd::Vector<3>, mfd::Vector<3>, mfd::Vector<3>, mfd::Vector<3>,
                           mfd::S2> StateType;


    struct MeasurePack
    {
        std::deque<std::shared_ptr<ImuData>> imu_lst;
        std::shared_ptr<LidarData> cloud;
    };

    struct CallbackInfo
    {
        double time;
        Sophus::SE3d pose;
    };

    using IESKF = ieskf::IESKF<StateType, 12>;

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


private:

    bool isReady()
    {
        return R_ItoL_.has_value() &&
               p_ItoL_.has_value() && 
               inited_error_cov_ && 
               inited_noise_cov_;
    }

    void lioThread();

    void imuPropagate();

    void imuBackPropagate();

    void computeFxAndFw(IESKF::ErrorPropagateFx& fx, IESKF::ErrorPropagateFw& fw, const StateType& X);


    IESKF kf;

    std::shared_ptr<std::thread> main_thread_;

    std::deque<std::shared_ptr<ImuData>> imu_data_queue_;
    std::deque<std::shared_ptr<MeasurePack>> meas_pack_;
    std::shared_ptr<MeasurePack> last_meas_;
    std::mutex meas_pack_mutex_, imu_data_mutex_;
    std::condition_variable meas_pack_cond_, imu_data_cond_;

    std::optional<std::function<void(std::shared_ptr<CallbackInfo>)>> updateCallback;

    std::optional<Sophus::SO3d> R_ItoL_;
    std::optional<Eigen::Vector3d> p_ItoL_;
    bool inited_noise_cov_;
    bool inited_error_cov_;
};

}

