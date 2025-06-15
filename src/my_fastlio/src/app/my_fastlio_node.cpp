#include <ros/ros.h>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <iostream>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>

#include "manifold/s2.hpp"
#include "manifold/so3.hpp"
#include "my_fastlio/my_fastlio_param.hpp"
#include "my_fastlio/my_fastlio.hpp"
#include "livox_ros_driver/CustomMsg.h"
#include "manifold/manifold.hpp"

using namespace my_fastlio;

MyFastLIO lio;


void paramFetch(ros::NodeHandle& nh)
{
    MyFastLIOParam& param = MyFastLIOParam::createInstance();
    nh.getParam("imu_topic", param.imu_topic);
    nh.getParam("lidar_topic", param.lidar_topic);

    std::vector<double> R_ItoL_vec(4);
    std::vector<double> t_ItoL_vec(3);
    nh.getParam("extrinsic/rotation", R_ItoL_vec);
    nh.getParam("extrinsic/translation", R_ItoL_vec);

    QuaT q_ItoL(R_ItoL_vec[0], R_ItoL_vec[1], R_ItoL_vec[2], R_ItoL_vec[3]);
    param.R_ItoL = q_ItoL.toRotationMatrix();
    param.t_ItoL = Eigen::Vector3d(t_ItoL_vec[0], t_ItoL_vec[1], t_ItoL_vec[2]);
}


void imuCallback(const sensor_msgs::ImuConstPtr& imu_msg)
{
    std::shared_ptr<ImuData> imu_data(new ImuData);
    imu_data->time = imu_msg->header.stamp.toSec();
    imu_data->w = Eigen::Vector3d(imu_msg->angular_velocity.x, imu_msg->angular_velocity.y, imu_msg->angular_velocity.z);
    imu_data->a = Eigen::Vector3d(imu_msg->linear_acceleration.x, imu_msg->linear_acceleration.y, imu_msg->linear_acceleration.z);
    
    lio.imuAsyncPushImu(imu_data);
}

void lidarPointCloud2Callback(const sensor_msgs::PointCloud2ConstPtr& lidar_msg)
{
    // ROS_INFO("Lidar data received");
    // CloudXYZPtr cloud(new CloudXYZ);
    // pcl::fromROSMsg(*lidar_msg, *cloud);

    // cloud->header.stamp = lidar_msg->header.stamp.toSec();
    // cloud->header.frame_id = lidar_msg->header.frame_id;

    // lio.lidarAsyncPushLidar(cloud);
}


void lidarLivoxCallback(const livox_ros_driver::CustomMsgConstPtr& lidar_msg)
{
    CloudPtr cloud(new CloudT);
    cloud->header.stamp = lidar_msg->header.stamp.toSec();
    cloud->header.frame_id = lidar_msg->header.frame_id;
    cloud->width = lidar_msg->point_num;
    cloud->height = 1;
    cloud->is_dense = false;
    cloud->header.seq = 0;
    cloud->points.reserve(lidar_msg->point_num);
    for (size_t i = 1; i < lidar_msg->point_num; ++i)
    {
        if (lidar_msg->points[i].x * lidar_msg->points[i].x + lidar_msg->points[i].y * lidar_msg->points[i].y + lidar_msg->points[i].z * lidar_msg->points[i].z > 0.1
            && (abs(lidar_msg->points[i].x - lidar_msg->points[i-1].x) > 1e-7 || abs(lidar_msg->points[i].y - lidar_msg->points[i-1].y) > 1e-7 || abs(lidar_msg->points[i].z - lidar_msg->points[i-1].z) > 1e-7))
        {
            PointT point;
            point.x = lidar_msg->points[i].x;
            point.y = lidar_msg->points[i].y;
            point.z = lidar_msg->points[i].z;
            point.intensity = lidar_msg->points[i].reflectivity;
            point.curvature = lidar_msg->points[i].offset_time / float(1e9);
            cloud->points.push_back(point);
        }
    }
    lio.lidarAsyncPushLidar(cloud);
}

int main(int argc, char** argv) 
{
    ros::init(argc, argv, "my_fastlio_node");
    ros::NodeHandle nh;
    paramFetch(nh);

    MyFastLIOParam& param = MyFastLIOParam::createInstance();

    ros::Subscriber imu_sub = nh.subscribe(param.imu_topic, 1000, imuCallback);

    std::string lidar_type;
    nh.getParam("lidar_type", lidar_type);
    std::cout << "Lidar type: " << lidar_type << std::endl;

    ros::Subscriber lidar_sub;

    if (lidar_type == "livox"){
        lidar_sub = nh.subscribe(param.lidar_topic, 1000, lidarLivoxCallback);
    }else{
        lidar_sub = nh.subscribe(param.lidar_topic, 1000, lidarPointCloud2Callback);
    }



    while(ros::ok())
    {
        ros::spinOnce();
    }

    ros::shutdown();

    return 0;
}


