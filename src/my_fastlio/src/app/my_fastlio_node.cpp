#include <memory>
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
#include "ros/publisher.h"
#include "ros/rate.h"
#include "ros/time.h"
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>

using namespace my_fastlio;

MyFastLIO lio;
std::string imu_topic;
std::string lidar_topic;
M3T R_ItoL;
V3T t_ItoL;
ros::Publisher pub_lidar_source;
ros::Publisher pub_odometry;
ros::Publisher pub_path;
ros::Publisher pub_map;
nav_msgs::Path path;

void paramFetch(ros::NodeHandle& nh)
{
    nh.getParam("imu_topic", imu_topic);
    nh.getParam("lidar_topic", lidar_topic);

    std::vector<double> R_ItoL_vec(4);
    std::vector<double> t_ItoL_vec(3);
    nh.getParam("extrinsic/rotation", R_ItoL_vec);
    nh.getParam("extrinsic/translation", t_ItoL_vec);

    QuaT q_ItoL(R_ItoL_vec[0], R_ItoL_vec[1], R_ItoL_vec[2], R_ItoL_vec[3]);
    R_ItoL = q_ItoL.toRotationMatrix();
    t_ItoL = Eigen::Vector3d(t_ItoL_vec[0], t_ItoL_vec[1], t_ItoL_vec[2]);
    std::cout << "imu topic: " << imu_topic << std::endl;
    std::cout << "lidar topic: " << lidar_topic << std::endl;
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

void lioUpdateCallback(std::shared_ptr<MyFastLIO::CallbackInfo> info)
{
    nav_msgs::Odometry odo;
    odo.pose.pose.position.x = info->pose.translation().x();
    odo.pose.pose.position.y = info->pose.translation().y();
    odo.pose.pose.position.z = info->pose.translation().z();
    odo.pose.pose.orientation.w = info->pose.unit_quaternion().w();
    odo.pose.pose.orientation.x = info->pose.unit_quaternion().x();
    odo.pose.pose.orientation.y = info->pose.unit_quaternion().y();
    odo.pose.pose.orientation.z = info->pose.unit_quaternion().z();
    odo.header.frame_id = "map";
    odo.header.stamp = ros::Time::now();
    pub_odometry.publish(odo);

    path.header.frame_id = "map";
    path.header.stamp = ros::Time::now();
    auto& pose = path.poses.emplace_back();
    pose.header.frame_id = "map";
    pose.header.stamp = ros::Time::now();
    pose.pose.position.x = info->pose.translation().x();
    pose.pose.position.y = info->pose.translation().y();
    pose.pose.position.z = info->pose.translation().z();

    pose.pose.orientation.w = info->pose.unit_quaternion().w();
    pose.pose.orientation.x = info->pose.unit_quaternion().x();
    pose.pose.orientation.y = info->pose.unit_quaternion().y();
    pose.pose.orientation.z = info->pose.unit_quaternion().z();
    pub_path.publish(path);

    sensor_msgs::PointCloud2 map_msg;
    if (info->map->size() > 0){
        pcl::toROSMsg(*info->map, map_msg);
        map_msg.header.frame_id = "map";
        map_msg.header.stamp = ros::Time::now();
        pub_map.publish(map_msg);
    }

    // std::cout << "update: " << "time: " << info->time << "\nq: " << info->pose.unit_quaternion().coeffs().transpose() << "\nt: " << info->pose.translation().transpose() << std::endl;
    // std::cout << "velocity: " << info->vel.transpose() << std::endl;
}


void lidarLivoxCallback(const livox_ros_driver::CustomMsgConstPtr& lidar_msg)
{
    std::shared_ptr<my_fastlio::LidarData> lidar_data(new my_fastlio::LidarData);
    lidar_data->time = lidar_msg->header.stamp.toSec();
    CloudPtr cloud = lidar_data->cloud;
    cloud->header.frame_id = lidar_msg->header.frame_id;
    cloud->width = lidar_msg->point_num;
    cloud->height = 1;
    cloud->is_dense = false;
    cloud->header.seq = 0;
    cloud->points.reserve(lidar_msg->point_num);
    for (size_t i = 0; i < lidar_msg->point_num; ++i)
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
    cloud->width = cloud->points.size();
    lio.lidarAsyncPushLidar(lidar_data);
    sensor_msgs::PointCloud2 ret_lidar_msg;
    pcl::toROSMsg(*cloud, ret_lidar_msg);
    ret_lidar_msg.header.frame_id = "map";
    ret_lidar_msg.header.stamp = ros::Time::now();
    pub_lidar_source.publish(ret_lidar_msg);
}

int main(int argc, char** argv) 
{
    ros::init(argc, argv, "my_fastlio_node");
    ros::NodeHandle nh;
    paramFetch(nh);

    ros::Subscriber imu_sub = nh.subscribe(imu_topic, 1000, imuCallback);
    pub_lidar_source = nh.advertise<sensor_msgs::PointCloud2>("/my_fastlio/lidar_source", 10);
    pub_odometry = nh.advertise<nav_msgs::Odometry>("/my_fastlio/odometry", 100);
    pub_path = nh.advertise<nav_msgs::Path>("/my_fastlio/path", 100);
    pub_map = nh.advertise<sensor_msgs::PointCloud2>("/my_fastlio/map", 10);

    std::string lidar_type;
    nh.getParam("lidar_type", lidar_type);
    std::cout << "Lidar type: " << lidar_type << std::endl;

    ros::Subscriber lidar_sub;

    if (lidar_type == "livox"){
        lidar_sub = nh.subscribe(lidar_topic, 1000, lidarLivoxCallback);
    }else{
        lidar_sub = nh.subscribe(lidar_topic, 1000, lidarPointCloud2Callback);
    }

    lio.setR_ItoL(R_ItoL)
       .setp_ItoL(t_ItoL)
       .setUpdateCallback(lioUpdateCallback)
       .setInitializeGravity(V3T(0,0,-9.8));
    lio.start();

    ros::Rate rate(10);
    while(ros::ok())
    {
        ros::spinOnce();
        rate.sleep();
    }

    ros::shutdown();

    return 0;
}


