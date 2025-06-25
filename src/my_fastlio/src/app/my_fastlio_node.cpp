#include <cstddef>
#include <cstdlib>
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
#include "pcl/common/transforms.h"
#include "pcl/impl/point_types.hpp"
#include "pcl/point_cloud.h"
#include "ros/publisher.h"
#include "ros/rate.h"
#include "ros/time.h"
#include "visualization_msgs/Marker.h"
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <visualization_msgs/MarkerArray.h>

using namespace my_fastlio;

typedef visualization_msgs::MarkerArray MsgVmap;
MyFastLIO lio;
std::string imu_topic;
std::string lidar_topic;
M3T R_ItoL;
V3T t_ItoL;
ros::Publisher pub_lidar_source;
ros::Publisher pub_odometry;
ros::Publisher pub_path;
ros::Publisher pub_map;
ros::Publisher pub_vmap;
ros::Publisher pub_ptpl_p;
ros::Publisher pub_ptpl_normal;
nav_msgs::Path path;
bool is_pub_voxelmap;

void paramFetch(ros::NodeHandle& nh)
{
    nh.getParam("imu_topic", imu_topic);
    nh.getParam("lidar_topic", lidar_topic);

    std::vector<double> R_ItoL_vec(4);
    std::vector<double> t_ItoL_vec(3);
    nh.getParam("extrinsic/rotation", R_ItoL_vec);
    nh.getParam("extrinsic/translation", t_ItoL_vec);
    nh.getParam("publish_voxmap", is_pub_voxelmap);

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

void pubSinglePlane(visualization_msgs::MarkerArray &plane_pub,
                    const std::string plane_ns, const voxelmap::Plane &single_plane,
                    const float alpha, const Eigen::Vector3d rgb) {
  visualization_msgs::Marker plane;
  plane.header.frame_id = "map";
  plane.header.stamp = ros::Time();
  plane.ns = plane_ns;
  plane.id = single_plane.id;
  plane.type = visualization_msgs::Marker::CYLINDER;
  plane.action = visualization_msgs::Marker::ADD;
  plane.pose.position.x = single_plane.center[0];
  plane.pose.position.y = single_plane.center[1];
  plane.pose.position.z = single_plane.center[2];
  geometry_msgs::Quaternion q;
  voxelmap::CalcVectQuation(single_plane.x_normal, single_plane.y_normal,
                  single_plane.normal, q);
  plane.pose.orientation = q;
  plane.scale.x = 3 * sqrt(single_plane.max_eigen_value);
  plane.scale.y = 3 * sqrt(single_plane.mid_eigen_value);
  plane.scale.z = 2 * sqrt(single_plane.min_eigen_value);
  plane.color.a = alpha;
  plane.color.r = rgb(0);
  plane.color.g = rgb(1);
  plane.color.b = rgb(2);
  plane.lifetime = ros::Duration();
  plane_pub.markers.push_back(plane);
}

void pubVoxelMap(const std::unordered_map<VOXEL_LOC, voxelmap::OctoTree *> &voxel_map,
                 const int pub_max_voxel_layer,
                 const ros::Publisher &plane_map_pub) {
  double max_trace = 0.25;
  double pow_num = 0.2;
  ros::Rate loop(500);
  float use_alpha = 0.8;
  visualization_msgs::MarkerArray voxel_plane;
  voxel_plane.markers.reserve(1000000);
  std::vector<voxelmap::Plane> pub_plane_list;
  for (auto iter = voxel_map.begin(); iter != voxel_map.end(); iter++) {
    GetUpdatePlane(iter->second, pub_max_voxel_layer, pub_plane_list);
  }
  for (size_t i = 0; i < pub_plane_list.size(); i++) {
    V3D plane_cov = pub_plane_list[i].plane_cov.block<3, 3>(0, 0).diagonal();
    double trace = plane_cov.sum();
    if (trace >= max_trace) {
      trace = max_trace;
    }
    trace = trace * (1.0 / max_trace);
    trace = pow(trace, pow_num);
    uint8_t r, g, b;
    voxelmap::mapJet(trace, 0, 1, r, g, b);
    Eigen::Vector3d plane_rgb(r / 256.0, g / 256.0, b / 256.0);
    double alpha;
    if (pub_plane_list[i].is_plane) {
      alpha = use_alpha;
    } else {
      alpha = 0;
    }
    pubSinglePlane(voxel_plane, "plane", pub_plane_list[i], alpha, plane_rgb);
  }
  plane_map_pub.publish(voxel_plane);
  loop.sleep();
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
    static int id_cloud = 0;
    if (info->map != nullptr) {
        if (info->map->size() > 0){
            CloudPtr filtered_pcd(new CloudT);
            pcl::transformPointCloud(*info->map, *filtered_pcd, info->pose.matrix().cast<float>());
            pcl::toROSMsg(*filtered_pcd, map_msg);
            map_msg.header.frame_id = "map";
            map_msg.header.seq = id_cloud++;
            map_msg.header.stamp = ros::Time::now();
            pub_map.publish(map_msg);
        }
    }
    if (is_pub_voxelmap){
        MsgVmap msg_map;
        if (info->vmap != nullptr){
            pubVoxelMap(*info->vmap, 4.0, pub_vmap);
        }
    }
    if (info->filtered_indices != nullptr)
        std::cout << "remain cloud size: " << info->filtered_indices->size() << std::endl;

    if (info->ptpl != nullptr){
        sensor_msgs::PointCloud2 msg_ptpl_p;
        MsgVmap msg_ptpl_normal;
        pcl::PointCloud<pcl::PointXYZRGB> pcd;
        int id = 0;
        for (auto& ptpl : *info->ptpl)
        {
            float r = float(rand()) / float(RAND_MAX);
            float g = float(rand()) / float(RAND_MAX);
            float b = float(rand()) / float(RAND_MAX);
            pcl::PointXYZRGB p_cent;
            p_cent.getVector3fMap() = ptpl.center.cast<float>();
            p_cent.getRGBVector3i() = (255.0 * Eigen::Matrix<float, 3, 1>(r, g, b)).cast<int>();
            pcd.push_back(p_cent);
            
            pcl::PointXYZRGB p_pcd;
            p_pcd.getVector3fMap() = (info->pose * ptpl.point).cast<float>();
            
            p_pcd.getRGBVector3i() = (255.0 * Eigen::Matrix<float, 3, 1>(r, g, b)).cast<int>();
            pcd.push_back(p_pcd);

            visualization_msgs::Marker normal;
            normal.header.frame_id = "map";  // 或者你当前使用的坐标系
            normal.header.stamp = ros::Time::now();
            normal.ns = "normals";
            normal.id = id; 
            normal.type = visualization_msgs::Marker::LINE_STRIP;
            normal.action = visualization_msgs::Marker::ADD;
            normal.pose.orientation.w = 1;
            normal.pose.orientation.x = 0;
            normal.pose.orientation.y = 0;
            normal.pose.orientation.z = 0;

            normal.scale.x = 0.03;

            normal.color.r = r;
            normal.color.g = g;
            normal.color.b = b;
            normal.color.a = 1;

            Eigen::Vector3f n = ptpl.normal.cast<float>();            // 法向量，单位向量
            geometry_msgs::Point start, end;
            start.x = p_cent.x;
            start.y = p_cent.y;
            start.z = p_cent.z;

            float scale = 0.4;  // 法向长度
            end.x = p_cent.x + scale * n.x();
            end.y = p_cent.y + scale * n.y();
            end.z = p_cent.z + scale * n.z();

            normal.points.push_back(start);
            normal.points.push_back(end);
            msg_ptpl_normal.markers.push_back(normal);
            id++;
        }
        pcl::toROSMsg(pcd, msg_ptpl_p);
        msg_ptpl_p.header.frame_id = "map";
        msg_ptpl_p.header.stamp = ros::Time::now();

        pub_ptpl_p.publish(msg_ptpl_p);
        pub_ptpl_normal.publish(msg_ptpl_normal);
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
    pub_vmap = nh.advertise<MsgVmap>("/my_fastlio/vmap", 10);
    pub_ptpl_p = nh.advertise<sensor_msgs::PointCloud2>("/my_fastlio/ptpl_p", 10);
    pub_ptpl_normal = nh.advertise<MsgVmap>("/my_fastlio/ptpl_normal", 10);

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


