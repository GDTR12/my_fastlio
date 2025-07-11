#include "voxel_map/voxel_map_util.hpp"
#include "Eigen/src/Core/Matrix.h"
#include <iostream>
#include <vector>

namespace voxelmap {

void mapJet(double v, double vmin, double vmax, uint8_t &r, uint8_t &g,
            uint8_t &b) {
  r = 255;
  g = 255;
  b = 255;

  if (v < vmin) {
    v = vmin;
  }

  if (v > vmax) {
    v = vmax;
  }

  double dr, dg, db;

  if (v < 0.1242) {
    db = 0.504 + ((1. - 0.504) / 0.1242) * v;
    dg = dr = 0.;
  } else if (v < 0.3747) {
    db = 1.;
    dr = 0.;
    dg = (v - 0.1242) * (1. / (0.3747 - 0.1242));
  } else if (v < 0.6253) {
    db = (0.6253 - v) * (1. / (0.6253 - 0.3747));
    dg = 1.;
    dr = (v - 0.3747) * (1. / (0.6253 - 0.3747));
  } else if (v < 0.8758) {
    db = 0.;
    dr = 1.;
    dg = (0.8758 - v) * (1. / (0.8758 - 0.6253));
  } else {
    db = 0.;
    dg = 0.;
    dr = 1. - (v - 0.8758) * ((1. - 0.504) / (1. - 0.8758));
  }

  r = (uint8_t)(255 * dr);
  g = (uint8_t)(255 * dg);
  b = (uint8_t)(255 * db);
}

void buildVoxelMap(const std::vector<pointWithCov> &input_points,
                   const float voxel_size, const int max_layer,
                   const std::vector<int> &layer_point_size,
                   const int max_points_size, const int max_cov_points_size,
                   const float planer_threshold,
                   std::unordered_map<VOXEL_LOC, OctoTree *> &feat_map) {
  uint plsize = input_points.size();
  for (uint i = 0; i < plsize; i++) {
    const pointWithCov p_v = input_points[i];
    float loc_xyz[3];
    for (int j = 0; j < 3; j++) {
      loc_xyz[j] = p_v.point[j] / voxel_size;
      if (loc_xyz[j] < 0) {
        loc_xyz[j] -= 1.0;
      }
    }
    VOXEL_LOC position((int64_t)loc_xyz[0], (int64_t)loc_xyz[1],
                       (int64_t)loc_xyz[2]);
    auto iter = feat_map.find(position);
    if (iter != feat_map.end()) {
      feat_map[position]->temp_points_.push_back(p_v);
      feat_map[position]->new_points_num_++;
    } else {
      OctoTree *octo_tree =
          new OctoTree(max_layer, 0, layer_point_size, max_points_size,
                       max_cov_points_size, planer_threshold);
      feat_map[position] = octo_tree;
      feat_map[position]->quater_length_ = voxel_size / 4;
      feat_map[position]->voxel_center_[0] = (0.5 + position.x) * voxel_size;
      feat_map[position]->voxel_center_[1] = (0.5 + position.y) * voxel_size;
      feat_map[position]->voxel_center_[2] = (0.5 + position.z) * voxel_size;
      feat_map[position]->temp_points_.push_back(p_v);
      feat_map[position]->new_points_num_++;
      feat_map[position]->layer_point_size_ = layer_point_size;
    }
  }
  for (auto iter = feat_map.begin(); iter != feat_map.end(); ++iter) {
    iter->second->init_octo_tree();
  }
}

void updateVoxelMap(const std::vector<pointWithCov> &input_points,
                    const float voxel_size, const int max_layer,
                    const std::vector<int> &layer_point_size,
                    const int max_points_size, const int max_cov_points_size,
                    const float planer_threshold,
                    std::unordered_map<VOXEL_LOC, OctoTree *> &feat_map) {
  uint plsize = input_points.size();
  for (uint i = 0; i < plsize; i++) {
    const pointWithCov p_v = input_points[i];
    float loc_xyz[3];
    for (int j = 0; j < 3; j++) {
      loc_xyz[j] = p_v.point[j] / voxel_size;
      if (loc_xyz[j] < 0) {
        loc_xyz[j] -= 1.0;
      }
    }
    VOXEL_LOC position((int64_t)loc_xyz[0], (int64_t)loc_xyz[1],
                       (int64_t)loc_xyz[2]);
    auto iter = feat_map.find(position);
    if (iter != feat_map.end()) {
      feat_map[position]->UpdateOctoTree(p_v);
    } else {
      OctoTree *octo_tree =
          new OctoTree(max_layer, 0, layer_point_size, max_points_size,
                       max_cov_points_size, planer_threshold);
      feat_map[position] = octo_tree;
      feat_map[position]->quater_length_ = voxel_size / 4;
      feat_map[position]->voxel_center_[0] = (0.5 + position.x) * voxel_size;
      feat_map[position]->voxel_center_[1] = (0.5 + position.y) * voxel_size;
      feat_map[position]->voxel_center_[2] = (0.5 + position.z) * voxel_size;
      feat_map[position]->UpdateOctoTree(p_v);
    }
  }
}
// void transformLidar(const StatesGroup &state,
//                     const shared_ptr<ImuProcess> &p_imu,
//                     const PointCloudXYZI::Ptr &input_cloud,
//                     pcl::PointCloud<pcl::PointXYZI>::Ptr &trans_cloud) {
void transformLidar(const StatesGroup &state,
                    const PointCloudXYZI::Ptr &input_cloud,
                    pcl::PointCloud<pcl::PointXYZI>::Ptr &trans_cloud) {
  trans_cloud->clear();
  for (size_t i = 0; i < input_cloud->size(); i++) {
    pcl::PointXYZINormal p_c = input_cloud->points[i];
    Eigen::Vector3d p(p_c.x, p_c.y, p_c.z);
    // p = p_imu->Lid_rot_to_IMU * p + p_imu->Lid_offset_to_IMU;
    p = state.rot_end * p + state.pos_end;
    pcl::PointXYZI pi;
    pi.x = p(0);
    pi.y = p(1);
    pi.z = p(2);
    pi.intensity = p_c.intensity;
    trans_cloud->points.push_back(pi);
  }
}

void build_single_residual(const pointWithCov &pv, const Eigen::Vector3d& point_world, const OctoTree *current_octo,
                           const int current_layer, const int max_layer,
                           const double sigma_num, bool &is_sucess,
                           double &prob, ptpl &single_ptpl) {
  double radius_k = 3;
  // Eigen::Vector3d p_w = pv.point_world;
  Eigen::Vector3d p_w = point_world;
  if (current_octo->plane_ptr_->is_plane) {
    Plane &plane = *current_octo->plane_ptr_;
    Eigen::Vector3d p_world_to_center = p_w - plane.center;
    double proj_x = p_world_to_center.dot(plane.x_normal);
    double proj_y = p_world_to_center.dot(plane.y_normal);
    float dis_to_plane =
        fabs(plane.normal(0) * p_w(0) + plane.normal(1) * p_w(1) +
             plane.normal(2) * p_w(2) + plane.d);
    float dis_to_center =
        (plane.center(0) - p_w(0)) * (plane.center(0) - p_w(0)) +
        (plane.center(1) - p_w(1)) * (plane.center(1) - p_w(1)) +
        (plane.center(2) - p_w(2)) * (plane.center(2) - p_w(2));
    float range_dis = sqrt(dis_to_center - dis_to_plane * dis_to_plane);

    if (range_dis <= radius_k * plane.radius) {
      Eigen::Matrix<double, 1, 6> J_nq;
      J_nq.block<1, 3>(0, 0) = p_w - plane.center;
      J_nq.block<1, 3>(0, 3) = -plane.normal;
      double sigma_l = J_nq * plane.plane_cov * J_nq.transpose();
      sigma_l += plane.normal.transpose() * pv.cov * plane.normal;
      if (dis_to_plane < sigma_num * sqrt(sigma_l)) {
        is_sucess = true;
        double this_prob = 1.0 / (sqrt(sigma_l)) *
                           exp(-0.5 * dis_to_plane * dis_to_plane / sigma_l);
        if (this_prob > prob) {
          prob = this_prob;
          single_ptpl.point = pv.point;
          single_ptpl.plane_cov = plane.plane_cov;
          single_ptpl.normal = plane.normal;
          single_ptpl.center = plane.center;
          single_ptpl.d = plane.d;
          single_ptpl.layer = current_layer;
          single_ptpl.point_cov = pv.cov;
        }
        return;
      } else {
        // is_sucess = false;
        return;
      }
    } else {
      // is_sucess = false;
      return;
    }
  } else {
    if (current_layer < max_layer) {
      for (size_t leafnum = 0; leafnum < 8; leafnum++) {
        if (current_octo->leaves_[leafnum] != nullptr) {

          OctoTree *leaf_octo = current_octo->leaves_[leafnum];
          build_single_residual(pv, point_world, leaf_octo, current_layer + 1, max_layer,
                                sigma_num, is_sucess, prob, single_ptpl);
        }
      }
      return;
    } else {
      // is_sucess = false;
      return;
    }
  }
}

void GetUpdatePlane(const OctoTree *current_octo, const int pub_max_voxel_layer,
                    std::vector<Plane> &plane_list) {
  if (current_octo->layer_ > pub_max_voxel_layer) {
    return;
  }
  if (current_octo->plane_ptr_->is_update) {
    plane_list.push_back(*current_octo->plane_ptr_);
  }
  if (current_octo->layer_ < current_octo->max_layer_) {
    if (!current_octo->plane_ptr_->is_plane) {
      for (size_t i = 0; i < 8; i++) {
        if (current_octo->leaves_[i] != nullptr) {
          GetUpdatePlane(current_octo->leaves_[i], pub_max_voxel_layer,
                         plane_list);
        }
      }
    }
  }
  return;
}

// void BuildResidualListTBB(const unordered_map<VOXEL_LOC, OctoTree *>
// &voxel_map,
//                           const double voxel_size, const double sigma_num,
//                           const int max_layer,
//                           const std::vector<pointWithCov> &pv_list,
//                           std::vector<ptpl> &ptpl_list,
//                           std::vector<Eigen::Vector3d> &non_match) {
//   std::mutex mylock;
//   ptpl_list.clear();
//   std::vector<ptpl> all_ptpl_list(pv_list.size());
//   std::vector<bool> useful_ptpl(pv_list.size());
//   std::vector<size_t> index(pv_list.size());
//   for (size_t i = 0; i < index.size(); ++i) {
//     index[i] = i;
//     useful_ptpl[i] = false;
//   }
//   std::for_each(
//       std::execution::par_unseq, index.begin(), index.end(),
//       [&](const size_t &i) {
//         pointWithCov pv = pv_list[i];
//         float loc_xyz[3];
//         for (int j = 0; j < 3; j++) {
//           loc_xyz[j] = pv.point_world[j] / voxel_size;
//           if (loc_xyz[j] < 0) {
//             loc_xyz[j] -= 1.0;
//           }
//         }
//         VOXEL_LOC position((int64_t)loc_xyz[0], (int64_t)loc_xyz[1],
//                            (int64_t)loc_xyz[2]);
//         auto iter = voxel_map.find(position);
//         if (iter != voxel_map.end()) {
//           OctoTree *current_octo = iter->second;
//           ptpl single_ptpl;
//           bool is_sucess = false;
//           double prob = 0;
//           build_single_residual(pv, current_octo, 0, max_layer, sigma_num,
//                                 is_sucess, prob, single_ptpl);
//           if (!is_sucess) {
//             VOXEL_LOC near_position = position;
//             if (loc_xyz[0] > (current_octo->voxel_center_[0] +
//                               current_octo->quater_length_)) {
//               near_position.x = near_position.x + 1;
//             } else if (loc_xyz[0] < (current_octo->voxel_center_[0] -
//                                      current_octo->quater_length_)) {
//               near_position.x = near_position.x - 1;
//             }
//             if (loc_xyz[1] > (current_octo->voxel_center_[1] +
//                               current_octo->quater_length_)) {
//               near_position.y = near_position.y + 1;
//             } else if (loc_xyz[1] < (current_octo->voxel_center_[1] -
//                                      current_octo->quater_length_)) {
//               near_position.y = near_position.y - 1;
//             }
//             if (loc_xyz[2] > (current_octo->voxel_center_[2] +
//                               current_octo->quater_length_)) {
//               near_position.z = near_position.z + 1;
//             } else if (loc_xyz[2] < (current_octo->voxel_center_[2] -
//                                      current_octo->quater_length_)) {
//               near_position.z = near_position.z - 1;
//             }
//             auto iter_near = voxel_map.find(near_position);
//             if (iter_near != voxel_map.end()) {
//               build_single_residual(pv, iter_near->second, 0, max_layer,
//                                     sigma_num, is_sucess, prob, single_ptpl);
//             }
//           }
//           if (is_sucess) {

//             mylock.lock();
//             useful_ptpl[i] = true;
//             all_ptpl_list[i] = single_ptpl;
//             mylock.unlock();
//           } else {
//             mylock.lock();
//             useful_ptpl[i] = false;
//             mylock.unlock();
//           }
//         }
//       });
//   for (size_t i = 0; i < useful_ptpl.size(); i++) {
//     if (useful_ptpl[i]) {
//       ptpl_list.push_back(all_ptpl_list[i]);
//     }
//   }
// }

void BuildResidualListOMP(const unordered_map<VOXEL_LOC, OctoTree *> &voxel_map,
                          const double voxel_size, const double sigma_num,
                          const int max_layer,
                          const std::vector<pointWithCov> &pv_list,
                          const std::vector<Eigen::Vector3d> &pv_world_list,
                          std::vector<ptpl> &ptpl_list,
                          std::vector<Eigen::Vector3d> &non_match) {
  std::mutex mylock;
  ptpl_list.clear();
  std::vector<ptpl> all_ptpl_list(pv_list.size());
  std::vector<bool> useful_ptpl(pv_list.size());
  std::vector<size_t> index(pv_list.size());
  for (size_t i = 0; i < index.size(); ++i) {
    index[i] = i;
    useful_ptpl[i] = false;
  }
#ifdef MP_EN
  omp_set_num_threads(MP_PROC_NUM);
#pragma omp parallel for
#endif
  for (int i = 0; i < index.size(); i++) {
    pointWithCov pv = pv_list[i];
    const Eigen::Vector3d& pv_world = pv_world_list[i];
    float loc_xyz[3];
    for (int j = 0; j < 3; j++) {
      loc_xyz[j] = pv_world[j] / voxel_size;
      if (loc_xyz[j] < 0) {
        loc_xyz[j] -= 1.0;
      }
    }
    VOXEL_LOC position((int64_t)loc_xyz[0], (int64_t)loc_xyz[1],
                       (int64_t)loc_xyz[2]);
    auto iter = voxel_map.find(position);
    if (iter != voxel_map.end()) {
      OctoTree *current_octo = iter->second;
      ptpl single_ptpl;
      bool is_sucess = false;
      double prob = 0;
      build_single_residual(pv, pv_world,current_octo, 0, max_layer, sigma_num,
                            is_sucess, prob, single_ptpl);
      if (!is_sucess) {
        VOXEL_LOC near_position = position;
        if (loc_xyz[0] >
            (current_octo->voxel_center_[0] + current_octo->quater_length_)) {
          near_position.x = near_position.x + 1;
        } else if (loc_xyz[0] < (current_octo->voxel_center_[0] -
                                 current_octo->quater_length_)) {
          near_position.x = near_position.x - 1;
        }
        if (loc_xyz[1] >
            (current_octo->voxel_center_[1] + current_octo->quater_length_)) {
          near_position.y = near_position.y + 1;
        } else if (loc_xyz[1] < (current_octo->voxel_center_[1] -
                                 current_octo->quater_length_)) {
          near_position.y = near_position.y - 1;
        }
        if (loc_xyz[2] >
            (current_octo->voxel_center_[2] + current_octo->quater_length_)) {
          near_position.z = near_position.z + 1;
        } else if (loc_xyz[2] < (current_octo->voxel_center_[2] -
                                 current_octo->quater_length_)) {
          near_position.z = near_position.z - 1;
        }
        auto iter_near = voxel_map.find(near_position);
        if (iter_near != voxel_map.end()) {
          build_single_residual(pv, pv_world, iter_near->second, 0, max_layer, sigma_num,
                                is_sucess, prob, single_ptpl);
        }
      }
      if (is_sucess) {

        mylock.lock();
        useful_ptpl[i] = true;
        all_ptpl_list[i] = single_ptpl;
        mylock.unlock();
      } else {
        mylock.lock();
        useful_ptpl[i] = false;
        mylock.unlock();
      }
    }
  }
  for (size_t i = 0; i < useful_ptpl.size(); i++) {
    if (useful_ptpl[i]) {
      ptpl_list.push_back(all_ptpl_list[i]);
    }
  }
}

void BuildResidualListNormal(
    const unordered_map<VOXEL_LOC, OctoTree *> &voxel_map,
    const double voxel_size, const double sigma_num, const int max_layer,
    const std::vector<pointWithCov> &pv_list, const std::vector<Eigen::Vector3d>& pv_world_lst , std::vector<ptpl> &ptpl_list,
    std::vector<Eigen::Vector3d> &non_match) {
  ptpl_list.clear();
  std::vector<size_t> index(pv_list.size());
  for (size_t i = 0; i < pv_list.size(); ++i) {
    pointWithCov pv = pv_list[i];
    const Eigen::Vector3d& pv_world = pv_world_lst[i];
    float loc_xyz[3];
    for (int j = 0; j < 3; j++) {
      loc_xyz[j] = pv_world[j] / voxel_size;
      if (loc_xyz[j] < 0) {
        loc_xyz[j] -= 1.0;
      }
    }
    VOXEL_LOC position((int64_t)loc_xyz[0], (int64_t)loc_xyz[1],
                       (int64_t)loc_xyz[2]);
    auto iter = voxel_map.find(position);
    if (iter != voxel_map.end()) {
      OctoTree *current_octo = iter->second;
      ptpl single_ptpl;
      bool is_sucess = false;
      double prob = 0;
      build_single_residual(pv, pv_world, current_octo, 0, max_layer, sigma_num,
                            is_sucess, prob, single_ptpl);

      if (!is_sucess) {
        VOXEL_LOC near_position = position;
        if (loc_xyz[0] >
            (current_octo->voxel_center_[0] + current_octo->quater_length_)) {
          near_position.x = near_position.x + 1;
        } else if (loc_xyz[0] < (current_octo->voxel_center_[0] -
                                 current_octo->quater_length_)) {
          near_position.x = near_position.x - 1;
        }
        if (loc_xyz[1] >
            (current_octo->voxel_center_[1] + current_octo->quater_length_)) {
          near_position.y = near_position.y + 1;
        } else if (loc_xyz[1] < (current_octo->voxel_center_[1] -
                                 current_octo->quater_length_)) {
          near_position.y = near_position.y - 1;
        }
        if (loc_xyz[2] >
            (current_octo->voxel_center_[2] + current_octo->quater_length_)) {
          near_position.z = near_position.z + 1;
        } else if (loc_xyz[2] < (current_octo->voxel_center_[2] -
                                 current_octo->quater_length_)) {
          near_position.z = near_position.z - 1;
        }
        auto iter_near = voxel_map.find(near_position);
        if (iter_near != voxel_map.end()) {
          build_single_residual(pv, pv_world, iter_near->second, 0, max_layer, sigma_num,
                                is_sucess, prob, single_ptpl);
        }
      }
      if (is_sucess) {
        ptpl_list.push_back(single_ptpl);
      } else {
        non_match.push_back(pv_world);
      }
    }
  }
}

void CalcVectQuation(const Eigen::Vector3d &x_vec, const Eigen::Vector3d &y_vec,
                     const Eigen::Vector3d &z_vec,
                     geometry_msgs::Quaternion &q) {

  Eigen::Matrix3d rot;
  rot << x_vec(0), x_vec(1), x_vec(2), y_vec(0), y_vec(1), y_vec(2), z_vec(0),
      z_vec(1), z_vec(2);
  Eigen::Matrix3d rotation = rot.transpose();
  Eigen::Quaterniond eq(rotation);
  q.w = eq.w();
  q.x = eq.x();
  q.y = eq.y();
  q.z = eq.z();
}

void CalcQuation(const Eigen::Vector3d &vec, const int axis,
                 geometry_msgs::Quaternion &q) {
  Eigen::Vector3d x_body = vec;
  Eigen::Vector3d y_body(1, 1, 0);
  if (x_body(2) != 0) {
    y_body(2) = -(y_body(0) * x_body(0) + y_body(1) * x_body(1)) / x_body(2);
  } else {
    if (x_body(1) != 0) {
      y_body(1) = -(y_body(0) * x_body(0)) / x_body(1);
    } else {
      y_body(0) = 0;
    }
  }
  y_body.normalize();
  Eigen::Vector3d z_body = x_body.cross(y_body);
  Eigen::Matrix3d rot;

  rot << x_body(0), x_body(1), x_body(2), y_body(0), y_body(1), y_body(2),
      z_body(0), z_body(1), z_body(2);
  Eigen::Matrix3d rotation = rot.transpose();
  if (axis == 2) {
    Eigen::Matrix3d rot_inc;
    rot_inc << 0, 0, 1, 0, 1, 0, -1, 0, 0;
    rotation = rotation * rot_inc;
  }
  Eigen::Quaterniond eq(rotation);
  q.w = eq.w();
  q.x = eq.x();
  q.y = eq.y();
  q.z = eq.z();
}

// TODO: pubSinglePlane
// void pubSinglePlane(visualization_msgs::MarkerArray &plane_pub,
//                     const std::string plane_ns, const Plane &single_plane,
//                     const float alpha, const Eigen::Vector3d rgb) {
//   visualization_msgs::Marker plane;
//   plane.header.frame_id = "camera_init";
//   plane.header.stamp = ros::Time();
//   plane.ns = plane_ns;
//   plane.id = single_plane.id;
//   plane.type = visualization_msgs::Marker::CYLINDER;
//   plane.action = visualization_msgs::Marker::ADD;
//   plane.pose.position.x = single_plane.center[0];
//   plane.pose.position.y = single_plane.center[1];
//   plane.pose.position.z = single_plane.center[2];
//   geometry_msgs::Quaternion q;
//   CalcVectQuation(single_plane.x_normal, single_plane.y_normal,
//                   single_plane.normal, q);
//   plane.pose.orientation = q;
//   plane.scale.x = 3 * sqrt(single_plane.max_eigen_value);
//   plane.scale.y = 3 * sqrt(single_plane.mid_eigen_value);
//   plane.scale.z = 2 * sqrt(single_plane.min_eigen_value);
//   plane.color.a = alpha;
//   plane.color.r = rgb(0);
//   plane.color.g = rgb(1);
//   plane.color.b = rgb(2);
//   plane.lifetime = ros::Duration();
//   plane_pub.markers.push_back(plane);
// }

// void pubNoPlaneMap(const std::unordered_map<VOXEL_LOC, OctoTree *> &feat_map,
//                    const ros::Publisher &plane_map_pub) {
//   int id = 0;
//   ros::Rate loop(500);
//   float use_alpha = 0.8;
//   visualization_msgs::MarkerArray voxel_plane;
//   voxel_plane.markers.reserve(1000000);
//   for (auto iter = feat_map.begin(); iter != feat_map.end(); iter++) {
//     if (!iter->second->plane_ptr_->is_plane) {
//       for (uint i = 0; i < 8; i++) {
//         if (iter->second->leaves_[i] != nullptr) {
//           OctoTree *temp_octo_tree = iter->second->leaves_[i];
//           if (!temp_octo_tree->plane_ptr_->is_plane) {
//             for (uint j = 0; j < 8; j++) {
//               if (temp_octo_tree->leaves_[j] != nullptr) {
//                 if (!temp_octo_tree->leaves_[j]->plane_ptr_->is_plane) {
//                   Eigen::Vector3d plane_rgb(1, 1, 1);
//                   pubSinglePlane(voxel_plane, "no_plane",
//                                  *(temp_octo_tree->leaves_[j]->plane_ptr_),
//                                  use_alpha, plane_rgb);
//                 }
//               }
//             }
//           }
//         }
//       }
//     }
//   }
//   plane_map_pub.publish(voxel_plane);
//   loop.sleep();
// }

// void pubVoxelMap(const std::unordered_map<VOXEL_LOC, OctoTree *> &voxel_map,
//                  const int pub_max_voxel_layer,
//                  const ros::Publisher &plane_map_pub) {
//   double max_trace = 0.25;
//   double pow_num = 0.2;
//   ros::Rate loop(500);
//   float use_alpha = 0.8;
//   visualization_msgs::MarkerArray voxel_plane;
//   voxel_plane.markers.reserve(1000000);
//   std::vector<Plane> pub_plane_list;
//   for (auto iter = voxel_map.begin(); iter != voxel_map.end(); iter++) {
//     GetUpdatePlane(iter->second, pub_max_voxel_layer, pub_plane_list);
//   }
//   for (size_t i = 0; i < pub_plane_list.size(); i++) {
//     V3D plane_cov = pub_plane_list[i].plane_cov.block<3, 3>(0, 0).diagonal();
//     double trace = plane_cov.sum();
//     if (trace >= max_trace) {
//       trace = max_trace;
//     }
//     trace = trace * (1.0 / max_trace);
//     trace = pow(trace, pow_num);
//     uint8_t r, g, b;
//     mapJet(trace, 0, 1, r, g, b);
//     Eigen::Vector3d plane_rgb(r / 256.0, g / 256.0, b / 256.0);
//     double alpha;
//     if (pub_plane_list[i].is_plane) {
//       alpha = use_alpha;
//     } else {
//       alpha = 0;
//     }
//     pubSinglePlane(voxel_plane, "plane", pub_plane_list[i], alpha, plane_rgb);
//   }
//   plane_map_pub.publish(voxel_plane);
//   loop.sleep();
// }

// void pubPlaneMap(const std::unordered_map<VOXEL_LOC, OctoTree *> &feat_map,
//                  const ros::Publisher &plane_map_pub) {
//   OctoTree *current_octo = nullptr;

//   double max_trace = 0.25;
//   double pow_num = 0.2;
//   ros::Rate loop(500);
//   float use_alpha = 1.0;
//   visualization_msgs::MarkerArray voxel_plane;
//   voxel_plane.markers.reserve(1000000);

//   for (auto iter = feat_map.begin(); iter != feat_map.end(); iter++) {
//     if (iter->second->plane_ptr_->is_update) {
//       Eigen::Vector3d normal_rgb(0.0, 1.0, 0.0);

//       V3D plane_cov =
//           iter->second->plane_ptr_->plane_cov.block<3, 3>(0, 0).diagonal();
//       double trace = plane_cov.sum();
//       if (trace >= max_trace) {
//         trace = max_trace;
//       }
//       trace = trace * (1.0 / max_trace);
//       trace = pow(trace, pow_num);
//       uint8_t r, g, b;
//       mapJet(trace, 0, 1, r, g, b);
//       Eigen::Vector3d plane_rgb(r / 256.0, g / 256.0, b / 256.0);
//       // Eigen::Vector3d plane_rgb(1, 0, 0);
//       float alpha = 0.0;
//       if (iter->second->plane_ptr_->is_plane) {
//         alpha = use_alpha;
//       } else {
//         // std::cout << "delete plane" << std::endl;
//       }
//       // if (iter->second->update_enable_) {
//       //   plane_rgb << 1, 0, 0;
//       // } else {
//       //   plane_rgb << 0, 0, 1;
//       // }
//       pubSinglePlane(voxel_plane, "plane", *(iter->second->plane_ptr_), alpha,
//                      plane_rgb);

//       iter->second->plane_ptr_->is_update = false;
//     } else {
//       for (uint i = 0; i < 8; i++) {
//         if (iter->second->leaves_[i] != nullptr) {
//           if (iter->second->leaves_[i]->plane_ptr_->is_update) {
//             Eigen::Vector3d normal_rgb(0.0, 1.0, 0.0);

//             V3D plane_cov = iter->second->leaves_[i]
//                                 ->plane_ptr_->plane_cov.block<3, 3>(0, 0)
//                                 .diagonal();
//             double trace = plane_cov.sum();
//             if (trace >= max_trace) {
//               trace = max_trace;
//             }
//             trace = trace * (1.0 / max_trace);
//             // trace = (max_trace - trace) / max_trace;
//             trace = pow(trace, pow_num);
//             uint8_t r, g, b;
//             mapJet(trace, 0, 1, r, g, b);
//             Eigen::Vector3d plane_rgb(r / 256.0, g / 256.0, b / 256.0);
//             plane_rgb << 0, 1, 0;
//             // fabs(iter->second->leaves_[i]->plane_ptr_->normal[0]),
//             //     fabs(iter->second->leaves_[i]->plane_ptr_->normal[1]),
//             //     fabs(iter->second->leaves_[i]->plane_ptr_->normal[2]);
//             float alpha = 0.0;
//             if (iter->second->leaves_[i]->plane_ptr_->is_plane) {
//               alpha = use_alpha;
//             } else {
//               // std::cout << "delete plane" << std::endl;
//             }
//             pubSinglePlane(voxel_plane, "plane",
//                            *(iter->second->leaves_[i]->plane_ptr_), alpha,
//                            plane_rgb);
//             // loop.sleep();
//             iter->second->leaves_[i]->plane_ptr_->is_update = false;
//             // loop.sleep();
//           } else {
//             OctoTree *temp_octo_tree = iter->second->leaves_[i];
//             for (uint j = 0; j < 8; j++) {
//               if (temp_octo_tree->leaves_[j] != nullptr) {
//                 if (temp_octo_tree->leaves_[j]->octo_state_ == 0 &&
//                     temp_octo_tree->leaves_[j]->plane_ptr_->is_update) {
//                   if (temp_octo_tree->leaves_[j]->plane_ptr_->is_plane) {
//                     // std::cout << "subsubplane" << std::endl;
//                     Eigen::Vector3d normal_rgb(0.0, 1.0, 0.0);
//                     V3D plane_cov =
//                         temp_octo_tree->leaves_[j]
//                             ->plane_ptr_->plane_cov.block<3, 3>(0, 0)
//                             .diagonal();
//                     double trace = plane_cov.sum();
//                     if (trace >= max_trace) {
//                       trace = max_trace;
//                     }
//                     trace = trace * (1.0 / max_trace);
//                     // trace = (max_trace - trace) / max_trace;
//                     trace = pow(trace, pow_num);
//                     uint8_t r, g, b;
//                     mapJet(trace, 0, 1, r, g, b);
//                     Eigen::Vector3d plane_rgb(r / 256.0, g / 256.0, b / 256.0);
//                     plane_rgb << 0, 0, 1;
//                     float alpha = 0.0;
//                     if (temp_octo_tree->leaves_[j]->plane_ptr_->is_plane) {
//                       alpha = use_alpha;
//                     }

//                     pubSinglePlane(voxel_plane, "plane",
//                                    *(temp_octo_tree->leaves_[j]->plane_ptr_),
//                                    alpha, plane_rgb);
//                     // loop.sleep();
//                     temp_octo_tree->leaves_[j]->plane_ptr_->is_update = false;
//                   }
//                 }
//               }
//             }
//           }
//         }
//       }
//     }
//   }

//   plane_map_pub.publish(voxel_plane);
//   // plane_map_pub.publish(voxel_norm);
//   loop.sleep();
//   // cout << "[Map Info] Plane counts:" << plane_count
//   //      << " Sub Plane counts:" << sub_plane_count
//   //      << " Sub Sub Plane counts:" << sub_sub_plane_count << endl;
//   // cout << "[Map Info] Update plane counts:" << update_count
//   //      << "total size: " << feat_map.size() << endl;
// }

void calcBodyCov(Eigen::Vector3d &pb, const float range_inc,
                 const float degree_inc, Eigen::Matrix3d &cov) {
  float range = sqrt(pb[0] * pb[0] + pb[1] * pb[1] + pb[2] * pb[2]);
  float range_var = range_inc * range_inc;
  Eigen::Matrix2d direction_var;
  direction_var << pow(sin(DEG2RAD(degree_inc)), 2), 0, 0,
      pow(sin(DEG2RAD(degree_inc)), 2);
  Eigen::Vector3d direction(pb);
  direction.normalize();
  Eigen::Matrix3d direction_hat;
  direction_hat << 0, -direction(2), direction(1), direction(2), 0,
      -direction(0), -direction(1), direction(0), 0;
  Eigen::Vector3d base_vector1(1, 1,
                               -(direction(0) + direction(1)) / direction(2));
  base_vector1.normalize();
  Eigen::Vector3d base_vector2 = base_vector1.cross(direction);
  base_vector2.normalize();
  Eigen::Matrix<double, 3, 2> N;
  N << base_vector1(0), base_vector2(0), base_vector1(1), base_vector2(1),
      base_vector1(2), base_vector2(2);
  Eigen::Matrix<double, 3, 2> A = range * direction_hat * N;
  cov = direction * range_var * direction.transpose() +
        A * direction_var * A.transpose();
};

}


