#include <ros/ros.h>
#include <std_msgs/Header.h>
#include <sensor_msgs/PointCloud2.h>
#include <geometry_msgs/PointStamped.h>
//#include <geometry_msgs/PoseStamped.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/conversions.h>
#include <pcl/PointIndices.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <tf/transform_listener.h>
#include <jsk_recognition_utils/pcl_conversion_util.h>
#include <jsk_recognition_utils/geo_util.h>
#include <safe_footstep_planner/OnlineFootStep.h>
#include <safe_footstep_planner/safe_footstep_util.h>
#include <visualization_msgs/Marker.h>


class TargetHeightPublisher
{
public:
    TargetHeightPublisher();
    ~TargetHeightPublisher(){};

private:
    void pointcloudCallback(const sensor_msgs::PointCloud2::ConstPtr& msg);
    void targetCallback(const safe_footstep_planner::OnlineFootStep::ConstPtr& msg);
    void matrixTFToEigen(const tf::Matrix3x3 &t, Eigen::Matrix3f &e);
    void vectorTFToEigen(const tf::Vector3& t, Eigen::Vector3f& k);
    void calcFootRotFromNormal (Eigen::Matrix3f& foot_rot, const Eigen::Matrix3f& orig_rot, const Eigen::Vector3f& n);
    Eigen::Vector3f rpyFromRot(const Eigen::Matrix3f& m);
    ros::NodeHandle nh_;
    ros::NodeHandle pnh_;
    ros::Publisher height_publisher_;
    ros::Publisher landing_pose_publisher_;
    ros::Subscriber cloud_sub_;
    ros::Subscriber target_sub_;
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_;
    tf::TransformListener listener_;
};

TargetHeightPublisher::TargetHeightPublisher() : nh_(""), pnh_("~")
{
    height_publisher_ = nh_.advertise<safe_footstep_planner::OnlineFootStep>("landing_height", 1);
    //landing_pose_publisher_ = nh_.advertise<geometry_msgs::PoseStamped>("landing_pose", 1);
    landing_pose_publisher_ = nh_.advertise<visualization_msgs::Marker>("landing_pose_marker", 1);
    // cloud_sub_ = nh_.subscribe("rt_accumulated_heightmap_pointcloud/output", 1, &TargetHeightPublisher::pointcloudCallback, this);
    cloud_sub_ = nh_.subscribe("rt_accumulated_heightmap_pointcloud_odomrelative/output", 1, &TargetHeightPublisher::pointcloudCallback, this);
    // cloud_sub_ = nh_.subscribe("rt_accumulated_heightmap_pointcloud_odomrelative_fixed/output", 1, &TargetHeightPublisher::pointcloudCallback, this);
    target_sub_ = nh_.subscribe("landing_target", 1, &TargetHeightPublisher::targetCallback, this);
    // std::cerr << "TargetHeihgtPublisher Initialized!!!!!! " << std::endl;
}

void TargetHeightPublisher::pointcloudCallback(const sensor_msgs::PointCloud2::ConstPtr& msg)
{
    cloud_.reset(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::fromROSMsg(*msg, *cloud_);
    // std::cout << "cloud size : " << cloud_->size() << std::endl;
}

void TargetHeightPublisher::targetCallback(const safe_footstep_planner::OnlineFootStep::ConstPtr& msg)
{
    double px = msg->x;
    double py = msg->y;
    double pz = msg->z;
    // std::cerr << "hoge: " << std::endl;
    // transform
    std::string target_frame;
    if (msg->l_r) {
        target_frame = "/lleg_end_coords";
    }
    else {
        target_frame = "/rleg_end_coords";
    }

    tf::StampedTransform transform;
    // listener_.lookupTransform("/body_on_odom", target_frame, ros::Time(0), transform); // map relative to target_frame
    // listener_.lookupTransform("/odom_ground", target_frame, ros::Time(0), transform); // map relative to target_frame
    listener_.lookupTransform(cloud_->header.frame_id, target_frame, ros::Time(0), transform); // map relative to target_frame
    Eigen::Vector3f cur_foot_pos, ez(Eigen::Vector3f::UnitZ());
    safe_footstep_util::vectorTFToEigen(transform.getOrigin(), cur_foot_pos);
    Eigen::Matrix3f tmp_cur_foot_rot, cur_foot_rot;
    safe_footstep_util::matrixTFToEigen(transform.getBasis(), tmp_cur_foot_rot);
    safe_footstep_util::calcFootRotFromNormal(cur_foot_rot, tmp_cur_foot_rot, ez);
    Eigen::Vector3f next_foot_pos;
    next_foot_pos = cur_foot_pos + cur_foot_rot * Eigen::Vector3f(px, py, pz);

    // double threshold = 0.04;
    double threshold = 0.03;
    // double cur_az = 0.0, next_az = 0.0;
    double cur_az_front = 0.0, next_az_front = 0.0;
    double cur_az_rear = 0.0, next_az_rear = 0.0;
    // int count_cur = 0, count_next = 0;
    int count_cur_front = 0, count_next_front = 0;
    int count_cur_rear = 0, count_next_rear = 0;
    //std::vector<double>  next_az_vec, cur_az_vec;
    std::vector<pcl::PointXYZ>  next_az_vec_front, cur_az_vec_front;
    std::vector<pcl::PointXYZ>  next_az_vec_rear, cur_az_vec_rear;
    std::vector<pcl::PointXYZ>  next_az_vec_left, next_az_vec_right;
    pcl::PointXYZ pp;
    // Eigen::Vector3f pos_margin (0.05, 0, 0);
    Eigen::Vector3f pos_margin_front (0.07, 0, 0);
    Eigen::Vector3f pos_margin_rear (-0.07, 0, 0);
    Eigen::Vector3f pos_margin_left (0.0, 0.04, 0);
    Eigen::Vector3f pos_margin_right (0.0, -0.04, 0);
    pcl::PointIndices::Ptr indices (new pcl::PointIndices);
    pcl::PointIndices::Ptr front_indices (new pcl::PointIndices);
    pcl::PointIndices::Ptr rear_indices (new pcl::PointIndices);

    // TODO rotation should be considered!

    ros::Time a_time = ros::Time::now();

    if (cloud_) {
        //organized点群の座標系とワールド座標系間の変換行列を求める
        double x_x_diff = 0;
        double x_y_diff = 0;
        for (int y = 0; x_x_diff == 0 && x_y_diff == 0 && y < cloud_->height; y++) {
          for (int x = 1; x_x_diff == 0 && x_y_diff == 0 && x < cloud_->width; x++) {
            if (pcl::isFinite(cloud_->points[y*cloud_->width+x-1]) && pcl::isFinite(cloud_->points[y*cloud_->width+x])) {
              x_x_diff = cloud_->points[y*cloud_->width+x].x - cloud_->points[y*cloud_->width+x-1].x;
              x_y_diff = cloud_->points[y*cloud_->width+x].y - cloud_->points[y*cloud_->width+x-1].y;
              if (!pcl::isFinite(cloud_->points[0])) {
                cloud_->points[0].x = cloud_->points[y*cloud_->width+x].x - x_x_diff * x - (-x_y_diff * y);
                cloud_->points[0].y = cloud_->points[y*cloud_->width+x].y - x_y_diff * x - (x_x_diff * y);
              }
            }
          }
        }
        int next_front_x, next_front_y, next_rear_x, next_rear_y;
        int cur_front_x, cur_front_y, cur_rear_x, cur_rear_y;
        Eigen::Matrix2d tmpmat;
        tmpmat << x_x_diff, -x_y_diff,
                  x_y_diff,  x_x_diff;
        Eigen::Vector2d tmpvec;
        Eigen::Vector2d next_front, next_rear, next_left, next_right, cur_front, cur_rear;
        tmpvec << next_foot_pos(0) + pos_margin_front(0) - cloud_->points[0].x, next_foot_pos(1) + pos_margin_front(1) - cloud_->points[0].y;
        next_front = tmpmat.colPivHouseholderQr().solve(tmpvec);
        tmpvec << next_foot_pos(0) + pos_margin_rear(0) - cloud_->points[0].x, next_foot_pos(1) + pos_margin_rear(1) - cloud_->points[0].y;
        next_rear = tmpmat.colPivHouseholderQr().solve(tmpvec);
        tmpvec << cur_foot_pos(0) + pos_margin_front(0) - cloud_->points[0].x, cur_foot_pos(1) + pos_margin_front(1) - cloud_->points[0].y;
        cur_front = tmpmat.colPivHouseholderQr().solve(tmpvec);
        tmpvec << cur_foot_pos(0) + pos_margin_rear(0) - cloud_->points[0].x, cur_foot_pos(1) + pos_margin_rear(1) - cloud_->points[0].y;
        cur_rear = tmpmat.colPivHouseholderQr().solve(tmpvec);
        tmpvec << next_foot_pos(0) + pos_margin_left(0) - cloud_->points[0].x, next_foot_pos(1) + pos_margin_left(1) - cloud_->points[0].y;
        next_left = tmpmat.colPivHouseholderQr().solve(tmpvec);
        tmpvec << next_foot_pos(0) + pos_margin_right(0) - cloud_->points[0].x, next_foot_pos(1) + pos_margin_right(1) - cloud_->points[0].y;
        next_right = tmpmat.colPivHouseholderQr().solve(tmpvec);

        //int tmp = threshold/0.01;//heighmapの１ピクセルは1cm
        int tmp = 4;
        for (int x = (int)(next_front(0)) + 1 - tmp; x < (int)(next_front(0)) + 1 + tmp; x++) {
          for (int y = (int)(next_front(1)) + 1 - tmp; y < (int)(next_front(1)) + 1 + tmp; y++) {
            next_az_front += cloud_->points[x+y*cloud_->width].z;
            count_next_front++;
            front_indices->indices.push_back(x+y*cloud_->width);
            next_az_vec_front.push_back(cloud_->points[x+y*cloud_->width]);
          }
        }
        for (int x = (int)(next_rear(0)) + 1 - tmp; x < (int)(next_rear(0)) + 1 + tmp; x++) {
          for (int y = (int)(next_rear(1)) + 1 - tmp; y < (int)(next_rear(1)) + 1 + tmp; y++) {
            next_az_rear += cloud_->points[x+y*cloud_->width].z;
            count_next_rear++;
            rear_indices->indices.push_back(x+y*cloud_->width);
            next_az_vec_rear.push_back(cloud_->points[x+y*cloud_->width]);
          }
        }
        for (int x = (int)(next_left(0)) + 1 - tmp; x < (int)(next_left(0)) + 1 + tmp; x++) {
          for (int y = (int)(next_left(1)) + 1 - tmp; y < (int)(next_left(1)) + 1 + tmp; y++) {
            next_az_vec_left.push_back(cloud_->points[x+y*cloud_->width]);
          }
        }
        for (int x = (int)(next_right(0)) + 1 - tmp; x < (int)(next_right(0)) + 1 + tmp; x++) {
          for (int y = (int)(next_right(1)) + 1 - tmp; y < (int)(next_right(1)) + 1 + tmp; y++) {
            next_az_vec_right.push_back(cloud_->points[x+y*cloud_->width]);
          }
        }
        for (int x = (int)(cur_front(0)) + 1 - tmp; x < (int)(cur_front(0)) + 1 + tmp; x++) {
          for (int y = (int)(cur_front(1)) + 1 - tmp; y < (int)(cur_front(1)) + 1 + tmp; y++) {
            cur_az_front += cloud_->points[x+y*cloud_->width].z;
            count_cur_front++;
            cur_az_vec_front.push_back(cloud_->points[x+y*cloud_->width]);
          }
        }
        for (int x = (int)(cur_rear(0)) + 1 - tmp; x < (int)(cur_rear(0)) + 1 + tmp; x++) {
          for (int y = (int)(cur_rear(1)) + 1 - tmp; y < (int)(cur_rear(1)) + 1 + tmp; y++) {
            cur_az_rear += cloud_->points[x+y*cloud_->width].z;
            count_cur_rear++;
            cur_az_vec_rear.push_back(cloud_->points[x+y*cloud_->width]);
          }
        }



        //for (int i = 0; i < cloud_->size(); i++) {
        //    pp = cloud_->points[i];
        //    // calc front mean height
        //    if (std::fabs(pp.x - (next_foot_pos(0) + pos_margin_front(0))) < threshold &&
        //        std::fabs(pp.y - next_foot_pos(1)) < threshold) {
        //        next_az_front += pp.z;
        //        count_next_front++;
        //        // TODO indices should be considered!
        //        front_indices->indices.push_back(i);
        //        next_az_vec_front.push_back(pp.z);
        //    }
        //    if (std::fabs(pp.x - (cur_foot_pos(0) + pos_margin_front(0))) < threshold &&
        //        std::fabs(pp.y - cur_foot_pos(1)) < threshold) {
        //        cur_az_front += pp.z;
        //        count_cur_front++;
        //        cur_az_vec_front.push_back(pp.z);
        //    }
        //    // calc rear mean height
        //    if (std::fabs(pp.x - (next_foot_pos(0) - pos_margin_rear(0))) < threshold &&
        //        std::fabs(pp.y - next_foot_pos(1)) < threshold) {
        //        next_az_rear += pp.z;
        //        count_next_rear++;
        //        // TODO indices should be considered!
        //        rear_indices->indices.push_back(i);
        //        next_az_vec_rear.push_back(pp.z);
        //    }
        //    if (std::fabs(pp.x - (cur_foot_pos(0) - pos_margin_rear(0))) < threshold &&
        //        std::fabs(pp.y - cur_foot_pos(1)) < threshold) {
        //        cur_az_rear += pp.z;
        //        count_cur_rear++;
        //        cur_az_vec_rear.push_back(pp.z);
        //    }
        //}
        // ROS_INFO("x: %f  y: %f  z: %f,  rel_pos x: %f  rel_pos y: %f, az/count: %f", landing_pos.getX(), landing_pos.getY(), landing_pos.getZ(), rel_landing_pos.getX(), rel_landing_pos.getY(), az / count);

        ros::Time b_time = ros::Time::now();
        
        // publish point
        if (count_next_front + count_next_rear > 0 && count_cur_front + count_cur_rear > 0) {
            safe_footstep_planner::OnlineFootStep ps;
            std_msgs::Header header;
            header.frame_id = target_frame.substr(1, target_frame.length() - 1);
            // header.stamp = ros::Time::now();
            header.stamp = ros::Time(0);
            ps.header = header;

            // mean
            // cur_foot_pos(2) = cur_az / static_cast<double>(count_cur);
            // next_foot_pos(2) = next_az / static_cast<double>(count_next);
            // median
            std::sort(next_az_vec_front.begin(), next_az_vec_front.end(), [](pcl::PointXYZ a, pcl::PointXYZ b) { return (a.z > b.z); });
            std::sort(cur_az_vec_front.begin(), cur_az_vec_front.end(), [](pcl::PointXYZ a, pcl::PointXYZ b) { return (a.z > b.z); });
            std::sort(next_az_vec_rear.begin(), next_az_vec_rear.end(), [](pcl::PointXYZ a, pcl::PointXYZ b) { return (a.z > b.z); });
            std::sort(cur_az_vec_rear.begin(), cur_az_vec_rear.end(), [](pcl::PointXYZ a, pcl::PointXYZ b) { return (a.z > b.z); });
            std::sort(next_az_vec_left.begin(), next_az_vec_left.end(), [](pcl::PointXYZ a, pcl::PointXYZ b) { return (a.z > b.z); });
            std::sort(next_az_vec_right.begin(), next_az_vec_right.end(), [](pcl::PointXYZ a, pcl::PointXYZ b) { return (a.z > b.z); });

            cur_foot_pos(2) = std::max(cur_az_vec_front[cur_az_vec_front.size()/4].z,
                                       cur_az_vec_rear[cur_az_vec_rear.size()/4].z);
            // omori comment out on 2020/12/24
            // cur_foot_pos(2) = (std::max(cur_az_vec_front[cur_az_vec_front.size()/2],
            //                             cur_az_vec_rear[cur_az_vec_rear.size()/2]) * 2
            //                    + cur_foot_pos(2)) / 3;
            //std::cout << "next_az_vec: " << next_az_vec_front[next_az_vec_front.size()/4].z << " " << next_az_vec_rear[next_az_vec_rear.size()/4].z << std::endl;
            if (std::abs(next_az_vec_front[next_az_vec_front.size()/4].z - next_az_vec_rear[next_az_vec_rear.size()/4].z) < 0.1) {
              front_indices->indices.insert(front_indices->indices.end(), rear_indices->indices.begin(), rear_indices->indices.end());
              indices = front_indices;
            } else if (next_az_vec_front[next_az_vec_front.size()/4].z > next_az_vec_rear[next_az_vec_rear.size()/4].z) {
              indices = front_indices;
            } else {
              indices = rear_indices;
            }

            ps.l_r = msg->l_r;
            // std::cerr << "height: " << limited_h << std::endl;

            // estimage plane by RANSAC
            int minimun_indices = 10;
            pcl::PointIndices::Ptr inliers (new pcl::PointIndices);
            pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients);
            pcl::SACSegmentation<pcl::PointXYZ> seg;
            seg.setOptimizeCoefficients (true);
            seg.setRadiusLimits(0.01, std::numeric_limits<double>::max ());
            seg.setMethodType(pcl::SAC_RANSAC);
            seg.setDistanceThreshold(0.1);
            seg.setModelType(pcl::SACMODEL_PLANE);
            seg.setInputCloud(cloud_);
            //
            seg.setIndices(indices);
            seg.setMaxIterations(100);
            seg.segment(*inliers, *coefficients);

            if (inliers->indices.size() == 0) {
                std::cerr <<  " no plane" << std::endl;
            }
            else if (inliers->indices.size() < minimun_indices) {
                std::cerr <<  " no enough inliners " << inliers->indices.size() <<  std::endl;
            }
            else {
                jsk_recognition_utils::Plane plane(coefficients->values);
                if (!plane.isSameDirection(ez)) {
                    plane = plane.flip();
                }
                Eigen::Vector3f next_n = plane.getNormal();
                next_n = cur_foot_rot.transpose() * next_n; // cur_foot relative

                if (next_n(2) < 0.8) { //平面おかしい
                  ps.nx =  0;
                  ps.ny =  0;
                  ps.nz =  1;
                  std::cout << "too steep slope " << next_n(2) << std::endl;
                } else {
                  ps.nx =  next_n(0);
                  ps.ny =  next_n(1);
                  ps.nz =  next_n(2);
                }
                //ps.nx =  0;
                //ps.ny =  0;
                //ps.nz =  1;
            }

            //Eigen::Vector3f next_n = (Eigen::Vector3f(
            //next_az_vec_front[next_az_vec_front.size()/4].x - next_az_vec_rear[next_az_vec_rear.size()/4].x,
            //next_az_vec_front[next_az_vec_front.size()/4].y - next_az_vec_rear[next_az_vec_rear.size()/4].y,
            //next_az_vec_front[next_az_vec_front.size()/4].z - next_az_vec_rear[next_az_vec_rear.size()/4].z
            //)).cross(Eigen::Vector3f(
            //next_az_vec_left[next_az_vec_left.size()/4].x - next_az_vec_right[next_az_vec_right.size()/4].x,
            //next_az_vec_left[next_az_vec_left.size()/4].y - next_az_vec_right[next_az_vec_right.size()/4].y,
            //next_az_vec_left[next_az_vec_left.size()/4].z - next_az_vec_right[next_az_vec_right.size()/4].z
            //));
            //next_n.normalize();
            //ps.nx =  next_n(0);
            //ps.ny =  next_n(1);
            //ps.nz =  next_n(2);
            //std::cout << ps.nx << " " << ps.ny << " " << ps.nz << std::endl;

            if (ps.nz != 0) {
              if (next_az_vec_front[next_az_vec_front.size()/4].z > next_az_vec_rear[next_az_vec_rear.size()/4].z) {
                next_foot_pos(2) = next_az_vec_front[next_az_vec_front.size()/4].z + (next_az_vec_front[next_az_vec_front.size()/4].x - next_foot_pos(0)) * ps.nx / ps.nz + (next_az_vec_front[next_az_vec_front.size()/4].y - next_foot_pos(1)) * ps.ny / ps.nz;
              } else {
                next_foot_pos(2) = next_az_vec_rear[next_az_vec_rear.size()/4].z + (next_az_vec_rear[next_az_vec_rear.size()/4].x - next_foot_pos(0)) * ps.nx / ps.nz + (next_az_vec_rear[next_az_vec_rear.size()/4].y - next_foot_pos(1)) * ps.ny / ps.nz;
              }
            } else {
              if (std::abs(next_az_vec_front[next_az_vec_front.size()/4].z - next_az_vec_rear[next_az_vec_rear.size()/4].z) < 0.1) {
                next_foot_pos(2) = (next_az_vec_front[next_az_vec_front.size()/2].z + next_az_vec_rear[next_az_vec_rear.size()/2].z) / 2.0;
              } else {
                next_foot_pos(2) = std::max(next_az_vec_front[next_az_vec_front.size()/4].z, next_az_vec_rear[next_az_vec_rear.size()/4].z);
              }
            }

            Eigen::Vector3f tmp_pos;
            tmp_pos = cur_foot_rot.transpose() * (next_foot_pos - cur_foot_pos);
            ps.x = tmp_pos(0);
            ps.y = tmp_pos(1);
            // target height range is -0.2 <= target_height <= 0.2


            if (std::abs(tmp_pos(2)) >= 0.2 || !std::isfinite(tmp_pos(2))) return;
            // double limited_h = std::min(0.2, std::max(-0.2,static_cast<double>(tmp_pos(2))));
            // ps.z = limited_h;

            ps.z = tmp_pos(2);
            //ps.z = 0;
            // ======= omori add 2020/02/16 ===========
            //ps.nx =  0;
            //ps.ny =  0;
            //ps.nz =  1;
            // ========================================
            height_publisher_.publish(ps);

            Eigen::Vector3f start_pos;
            start_pos = tmp_cur_foot_rot.transpose() * cur_foot_rot * Eigen::Vector3f(ps.x, ps.y, ps.z);
            Eigen::Vector3f end_pos;
            end_pos = tmp_cur_foot_rot.transpose() * cur_foot_rot * Eigen::Vector3f(ps.x+0.3*ps.nx, ps.y+0.3*ps.ny, ps.z+0.3*ps.nz);


            // publish pose msg for visualize
            visualization_msgs::Marker pose_msg;
            pose_msg.header = ps.header;
            pose_msg.ns = "landing_pose";
            pose_msg.id = 0;
            pose_msg.lifetime = ros::Duration();
            pose_msg.type = visualization_msgs::Marker::ARROW;
            pose_msg.action = visualization_msgs::Marker::ADD;
            geometry_msgs::Point start;
            start.x = start_pos(0);
            start.y = start_pos(1);
            start.z = start_pos(2);
            geometry_msgs::Point end;
            end.x = end_pos(0);
            end.y = end_pos(1);
            end.z = end_pos(2);
            pose_msg.points.push_back(start);
            pose_msg.points.push_back(end);
            pose_msg.color.r = 0.0;
            pose_msg.color.g = 0.8;
            pose_msg.color.b = 1.0;
            pose_msg.color.a = 1.0;
            pose_msg.scale.x = 0.03;
            pose_msg.scale.y = 0.05;
            pose_msg.scale.z = 0.07;
            pose_msg.pose.orientation.w = 1.0;

            landing_pose_publisher_.publish(pose_msg);
        } else {
          std::cout << "zerodayo" << std::endl;
        }
        ros::Time c_time = ros::Time::now();
        std::cout << "landing_height_publisher" << std::endl;
        std::cout << "a_b  " << (b_time - a_time).sec << "s " << (int)((b_time - a_time).nsec / 1000000) << "ms" << std::endl;
        std::cout << "b_c  " << (c_time - b_time).sec << "s " << (int)((c_time - b_time).nsec / 1000000) << "ms" << std::endl;
    }
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "target_height_publisher");
    TargetHeightPublisher target_height_publisher;
    ros::Duration(5);
    ros::spin();

    return 0;
}
