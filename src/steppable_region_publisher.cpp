#include <ros/ros.h>
#include <std_msgs/Header.h>
#include <sensor_msgs/PointCloud2.h>
#include <jsk_recognition_msgs/PolygonArray.h>
#include <geometry_msgs/PointStamped.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/conversions.h>
#include <pcl/PointIndices.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <tf/transform_listener.h>
#include <jsk_recognition_utils/pcl_conversion_util.h>
#include <jsk_recognition_utils/geo_util.h>
#include <safe_footstep_planner/OnlineFootStep.h>
#include <safe_footstep_planner/SteppableRegion.h>
#include <safe_footstep_planner/PolygonArray.h>
#include <safe_footstep_planner/safe_footstep_util.h>
#include "polypartition.h"
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>


class SteppableRegionPublisher
{
public:
  SteppableRegionPublisher();
  ~SteppableRegionPublisher(){};

private:
  ros::NodeHandle nh_;
  ros::NodeHandle pnh_;
  safe_footstep_planner::PolygonArray combined_meshes_;
  ros::Subscriber target_sub_;
  ros::Subscriber pointcloud_sub_;
  ros::Publisher region_publisher_;
  ros::Publisher combined_mesh_publisher_;
  ros::Publisher image_publisher_;
  ros::Publisher polygon_publisher_;
  tf::TransformListener listener_;
  void polygonarrayCallback(const jsk_recognition_msgs::PolygonArray::ConstPtr& msg);
  void targetCallback(const safe_footstep_planner::OnlineFootStep::ConstPtr& msg);
  void pointcloudCallback(const sensor_msgs::PointCloud2ConstPtr& input);
  cv::Mat median_image_;
  cv::Mat median_image2_;
  float x_x_diff;
  float x_y_diff;
  float y_x_diff;
  float y_y_diff;
};

SteppableRegionPublisher::SteppableRegionPublisher() : nh_(""), pnh_("~")
{
  region_publisher_ = nh_.advertise<safe_footstep_planner::SteppableRegion>("steppable_region", 1);
  combined_mesh_publisher_ = nh_.advertise<safe_footstep_planner::PolygonArray>("combined_meshed_polygons", 1);
  image_publisher_ = nh_.advertise<sensor_msgs::Image> ("output", 1);
  polygon_publisher_ = nh_.advertise<jsk_recognition_msgs::PolygonArray> ("output_polygon", 1);
  target_sub_ = nh_.subscribe("landing_target", 1, &SteppableRegionPublisher::targetCallback, this);
  pointcloud_sub_ = nh_.subscribe("rt_accumulated_heightmap_pointcloud_odomrelative/output", 1, &SteppableRegionPublisher::pointcloudCallback, this);
  median_image_ = cv::Mat::zeros(250, 250, CV_32FC3);
  median_image2_ = cv::Mat::zeros(250, 250, CV_32FC3);
  x_x_diff = 0;
  x_y_diff = 0;
  y_x_diff = 0;
  y_y_diff = 0;
}

void SteppableRegionPublisher::targetCallback(const safe_footstep_planner::OnlineFootStep::ConstPtr& msg)
{
  safe_footstep_planner::SteppableRegion sr;
  std::string target_frame;
  if (msg->l_r) {
    target_frame = "/lleg_end_coords";
  }
  else {
    target_frame = "/rleg_end_coords";
  }

  tf::StampedTransform transform;
  // listener_.lookupTransform("/body_on_odom", target_frame, ros::Time(0), transform); // map relative to target_frame
  listener_.lookupTransform("/odom_ground", target_frame, ros::Time(0), transform); // map relative to target_frame
  // listener_.lookupTransform(combined_meshes_.header.frame_id, target_frame, ros::Time(0), transform); // map relative to target_frame
  Eigen::Vector3f cur_foot_pos, ez(Eigen::Vector3f::UnitZ());
  safe_footstep_util::vectorTFToEigen(transform.getOrigin(), cur_foot_pos);
  Eigen::Matrix3f tmp_cur_foot_rot, cur_foot_rot;
  safe_footstep_util::matrixTFToEigen(transform.getBasis(), tmp_cur_foot_rot);
  safe_footstep_util::calcFootRotFromNormal(cur_foot_rot, tmp_cur_foot_rot, ez);

  if (x_x_diff != 0 || x_y_diff != 0) {
    double norm = std::sqrt(x_x_diff*x_x_diff + x_y_diff*x_y_diff);
    Eigen::Matrix2d tmpmat;
    tmpmat << x_x_diff, -x_y_diff,
              x_y_diff,  x_x_diff;
    Eigen::Vector2d tmpvec;
    tmpvec << cur_foot_pos[0] - median_image_.at<cv::Vec3f>(0,0)[0], cur_foot_pos[1] - median_image_.at<cv::Vec3f>(0,0)[1];
    Eigen::Vector2d tmp;
    tmp = tmpmat.colPivHouseholderQr().solve(tmpvec);
    cur_foot_pos[2] = median_image_.at<cv::Vec3f>((int)(tmp[1]), (int)(tmp[0]))[2];
    //std::cout << x_x_diff << " " << x_y_diff << "  " << cur_foot_pos[0] << " " << cur_foot_pos[1] << "  " << median_image_.at<cv::Vec3f>(0, 0)[0] << " " << median_image_.at<cv::Vec3f>(0, 0)[1] << " " << median_image_.at<cv::Vec3f>(0, 0)[2] << "  " << tmp[0] << " " << tmp[1] << "  " << cur_foot_pos[2] << std::endl;
  }

  // convert to polygon relative to leg_end_coords
  size_t convex_num(combined_meshes_.polygons.size());
  sr.polygons.resize(convex_num);
  for (size_t i = 0; i < convex_num; i++) {
    size_t vs_num(combined_meshes_.polygons[i].points.size());
    sr.polygons[i].polygon.points.resize(vs_num);
    for (size_t j = 0; j < vs_num; j++) {
      safe_footstep_util::transformPoint(combined_meshes_.polygons[i].points[j], cur_foot_rot, cur_foot_pos, sr.polygons[i].polygon.points[j]);
    }
  }

  std_msgs::Header header;
  header.frame_id = target_frame.substr(1, target_frame.length() - 1);
  // header.stamp = ros::Time::now();
  header.stamp = ros::Time(0);
  sr.header = header;
  sr.l_r = msg->l_r;

  region_publisher_.publish(sr);
}

void SteppableRegionPublisher::pointcloudCallback(const sensor_msgs::PointCloud2ConstPtr& input)
{

  ros::Time begin_time = ros::Time::now();
  // Convert the sensor_msgs/PointCloud2 data to pcl/PointCloud
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());
  //pcl::PointCloud<pcl::Normal> cloud;
  pcl::fromROSMsg (*input, *cloud);


  //fill infinite point
  x_x_diff = 0;
  x_y_diff = 0;
  for (int y = 0; y < 500; y+=2) {
    for (int x = 2; x < 500; x+=2) {
      if (pcl::isFinite(cloud->points[y*500+x])) {
        if (x_x_diff == 0 && pcl::isFinite(cloud->points[y*500+x-2])) { //連続してFiniteな部分を探し、diffを計算
          x_x_diff = cloud->points[y*500+x].x - cloud->points[y*500+x-2].x;
          x_y_diff = cloud->points[y*500+x].y - cloud->points[y*500+x-2].y;
        }
      } else {
        //std::cout << "nan value: " << j << " " << i << " " << cloud->points[i*500+j].x << " " << cloud->points[i*500+j].y << " " << cloud->points[i*500+j].z << std::endl;
        if (x_x_diff != 0 && pcl::isFinite(cloud->points[y*500+x-2])) { //inFiniteな部分には隣のzをコピー
          cloud->points[y*500+x].x = cloud->points[y*500+x-2].x + x_x_diff;
          cloud->points[y*500+x].y = cloud->points[y*500+x-2].y + x_y_diff;
          cloud->points[y*500+x].z = cloud->points[y*500+x-2].z;
        }
      }
    }
  }
  for (int i = 498; i >= 0; i-=2) {//逆順
    for (int j = 496; j >= 0; j-=2) {
      if ((!pcl::isFinite(cloud->points[i*500+j])) && pcl::isFinite(cloud->points[i*500+j+2])) { //inFiniteな部分には隣のzをコピー
        cloud->points[i*500+j].x = cloud->points[i*500+j+2].x - x_x_diff;
        cloud->points[i*500+j].y = cloud->points[i*500+j+2].y - x_y_diff;
        cloud->points[i*500+j].z = cloud->points[i*500+j+2].z;
      }
    }
  }
  y_x_diff = 0;
  y_y_diff = 0;
  for (int i = 2; i < 500; i+=2) {
    for (int j = 0; j < 500; j+=2) {
      if (pcl::isFinite(cloud->points[i*500+j])) {
        if (y_x_diff == 0 && pcl::isFinite(cloud->points[(i-2)*500+j])) { //連続してFiniteな部分を探し、diffを計算
          y_x_diff = cloud->points[i*500+j].x - cloud->points[(i-2)*500+j].x;
          y_y_diff = cloud->points[i*500+j].y - cloud->points[(i-2)*500+j].y;
        }
      } else {
        if (y_x_diff != 0 && pcl::isFinite(cloud->points[(i-2)*500+j])) { //inFiniteな部分には隣のzをコピー
          cloud->points[i*500+j].x = cloud->points[(i-2)*500+j].x + y_x_diff;
          cloud->points[i*500+j].y = cloud->points[(i-2)*500+j].y + y_y_diff;
          cloud->points[i*500+j].z = cloud->points[(i-2)*500+j].z;
        }
      }
    }
  }
  for (int i = 496; i >= 0; i-=2) {//逆順
    for (int j = 498; j >= 0; j-=2) {
      if ((!pcl::isFinite(cloud->points[i*500+j])) && pcl::isFinite(cloud->points[(i+2)*500+j])) { //inFiniteな部分には隣のzをコピー
        cloud->points[i*500+j].x = cloud->points[(i+2)*500+j].x - y_x_diff;
        cloud->points[i*500+j].y = cloud->points[(i+2)*500+j].y - y_y_diff;
        cloud->points[i*500+j].z = cloud->points[(i+2)*500+j].z;
      }
    }
  }

  ros::Time a_time = ros::Time::now();

  median_image_ = cv::Mat::zeros(250, 250, CV_32FC3);
  median_image2_ = cv::Mat::zeros(250, 250, CV_32FC3);

  //compress image size by a facter of four
  for (int y = 0; y < 250; y++) {
    for (int x = 0; x < 250; x++) {
      if (!pcl::isFinite(cloud->points[(y*2)*500+x*2])) {
        std::cout << "infinite aruyo " << x << " " << y << std::endl;
      } else {
        median_image_.at<cv::Vec3f>(y, x)[0] = cloud->points[y*2*500+x*2].x;
        median_image_.at<cv::Vec3f>(y, x)[1] = cloud->points[y*2*500+x*2].y;
        median_image_.at<cv::Vec3f>(y, x)[2] = cloud->points[y*2*500+x*2].z;
      }
    }
  }

  ros::Time b_time = ros::Time::now();

  cv::medianBlur(median_image_, median_image_, 3); //中央値を取る(x,y座標は3x3の中心になってしまう)
  cv::medianBlur(median_image_, median_image2_, 5); //中央値を取る(x,y座標は3x3の中心になってしまう)
  //cv::blur(median_image_, median_image_, cv::Size(3, 3));
  //cv::dilate(median_image_, median_image_, cv::Mat(), cv::Point(-1, -1), 1);

  ros::Time c_time = ros::Time::now();

  cv::Mat binarized_image = cv::Mat::zeros(250, 250, CV_8UC1);
  cv::Mat image = cv::Mat::zeros(250, 250, CV_8UC3);
  int steppable_range = 3;
  float steppable_edge_height = steppable_range*0.02*std::tan(0.33);
  float steppable_corner_height = steppable_range*0.02*std::sqrt(2)*std::tan(0.33);
  float steppable_around_edge_range = 18.0/2;//[cm]/[cm]
  float steppable_around_corner_range = (int)(18.0/std::sqrt(8));//[cm]/[cm]
  float steppable_around_height_diff = 0.05;//[m]

  for (int x = (int)(steppable_around_edge_range); x < (250-(int)(steppable_around_edge_range)); x++) {
    for (int y = (int)(steppable_around_edge_range); y < (250-(int)(steppable_around_edge_range)); y++) {
      cv::Vec3f center = median_image_.at<cv::Vec3f>(y, x);

      //if (
      //  std::abs(center[2] - median_image_.at<cv::Vec3f>(y+steppable_range, x+steppable_range)[2]) > steppable_corner_height ||
      //  std::abs(center[2] - median_image_.at<cv::Vec3f>(y+steppable_range, x-steppable_range)[2]) > steppable_corner_height ||
      //  std::abs(center[2] - median_image_.at<cv::Vec3f>(y-steppable_range, x+steppable_range)[2]) > steppable_corner_height ||
      //  std::abs(center[2] - median_image_.at<cv::Vec3f>(y-steppable_range, x-steppable_range)[2]) > steppable_corner_height ||
      //  std::abs(center[2] - median_image_.at<cv::Vec3f>(y+steppable_range, x+0)[2]) > steppable_edge_height ||
      //  std::abs(center[2] - median_image_.at<cv::Vec3f>(y-steppable_range, x+0)[2]) > steppable_edge_height ||
      //  std::abs(center[2] - median_image_.at<cv::Vec3f>(y+0, x+steppable_range)[2]) > steppable_edge_height ||
      //  std::abs(center[2] - median_image_.at<cv::Vec3f>(y+0, x-steppable_range)[2]) > steppable_edge_height) {
      //  continue;
      //}

      if (
        2*(center[2] - median_image_.at<cv::Vec3f>(y, x-steppable_range)[2]) - (median_image_.at<cv::Vec3f>(y, x+steppable_range)[2] - median_image_.at<cv::Vec3f>(y, x-steppable_range)[2]) > steppable_edge_height ||
        2*(center[2] - median_image_.at<cv::Vec3f>(y-steppable_range, x)[2]) - (median_image_.at<cv::Vec3f>(y+steppable_range, x)[2] - median_image_.at<cv::Vec3f>(y-steppable_range, x)[2]) > steppable_edge_height ||
        2*(center[2] - median_image_.at<cv::Vec3f>(y-steppable_range, x-steppable_range)[2]) - (median_image_.at<cv::Vec3f>(y+steppable_range, x+steppable_range)[2] - median_image_.at<cv::Vec3f>(y-steppable_range, x-steppable_range)[2]) > steppable_corner_height ||
        2*(center[2] - median_image_.at<cv::Vec3f>(y+steppable_range, x-steppable_range)[2]) - (median_image_.at<cv::Vec3f>(y-steppable_range, x+steppable_range)[2] - median_image_.at<cv::Vec3f>(y+steppable_range, x-steppable_range)[2]) > steppable_corner_height ||
        std::abs(median_image_.at<cv::Vec3f>(y, x+steppable_range)[2] - median_image_.at<cv::Vec3f>(y, x-steppable_range)[2]) > 2*steppable_edge_height ||
        std::abs(median_image_.at<cv::Vec3f>(y+steppable_range, x)[2] - median_image_.at<cv::Vec3f>(y-steppable_range, x)[2]) > 2*steppable_edge_height ||
        std::abs(median_image_.at<cv::Vec3f>(y+steppable_range, x+steppable_range)[2] - median_image_.at<cv::Vec3f>(y-steppable_range, x-steppable_range)[2]) > 2*steppable_corner_height ||
        std::abs(median_image_.at<cv::Vec3f>(y-steppable_range, x+steppable_range)[2] - median_image_.at<cv::Vec3f>(y+steppable_range, x-steppable_range)[2]) > 2*steppable_corner_height) {
        continue;
      }

      image.at<cv::Vec3b>(y, x)[0] = 100;
      image.at<cv::Vec3b>(y, x)[1] = 100;
      image.at<cv::Vec3b>(y, x)[2] = 100;

      double center_height = std::max({
        median_image_.at<cv::Vec3f>(y-steppable_range, x-steppable_range)[2],
        median_image_.at<cv::Vec3f>(y-steppable_range, x)[2],
        median_image_.at<cv::Vec3f>(y-steppable_range, x+steppable_range)[2],
        median_image_.at<cv::Vec3f>(y, x-steppable_range)[2],
        median_image_.at<cv::Vec3f>(y, x)[2],
        median_image_.at<cv::Vec3f>(y, x+steppable_range)[2],
        median_image_.at<cv::Vec3f>(y+steppable_range, x-steppable_range)[2],
        median_image_.at<cv::Vec3f>(y+steppable_range, x)[2],
        median_image_.at<cv::Vec3f>(y+steppable_range, x+steppable_range)[2]
        });

      //if (
      //  median_image_.at<cv::Vec3f>(y+(int)(steppable_around_edge_range), x)[2] - center[2] > steppable_around_height_diff ||
      //  median_image_.at<cv::Vec3f>(y, x+(int)(steppable_around_edge_range))[2] - center[2] > steppable_around_height_diff ||
      //  median_image_.at<cv::Vec3f>(y-(int)(steppable_around_edge_range), x)[2] - center[2] > steppable_around_height_diff ||
      //  median_image_.at<cv::Vec3f>(y, x-(int)(steppable_around_edge_range))[2] - center[2] > steppable_around_height_diff ||
      //  median_image_.at<cv::Vec3f>(y+(int)(steppable_around_corner_range), x+(int)(steppable_around_corner_range))[2] - center[2] > steppable_around_height_diff ||
      //  median_image_.at<cv::Vec3f>(y+(int)(steppable_around_corner_range), x-(int)(steppable_around_corner_range))[2] - center[2] > steppable_around_height_diff ||
      //  median_image_.at<cv::Vec3f>(y-(int)(steppable_around_corner_range), x+(int)(steppable_around_corner_range))[2] - center[2] > steppable_around_height_diff ||
      //  median_image_.at<cv::Vec3f>(y-(int)(steppable_around_corner_range), x-(int)(steppable_around_corner_range))[2] - center[2] > steppable_around_height_diff) {
      //  continue;
      //}
      if (
        median_image_.at<cv::Vec3f>(y+(int)(steppable_around_edge_range), x)[2] - center_height > steppable_around_height_diff ||
        median_image_.at<cv::Vec3f>(y, x+(int)(steppable_around_edge_range))[2] - center_height > steppable_around_height_diff ||
        median_image_.at<cv::Vec3f>(y-(int)(steppable_around_edge_range), x)[2] - center_height > steppable_around_height_diff ||
        median_image_.at<cv::Vec3f>(y, x-(int)(steppable_around_edge_range))[2] - center_height > steppable_around_height_diff ||
        median_image_.at<cv::Vec3f>(y+(int)(steppable_around_corner_range), x+(int)(steppable_around_corner_range))[2] - center_height > steppable_around_height_diff ||
        median_image_.at<cv::Vec3f>(y+(int)(steppable_around_corner_range), x-(int)(steppable_around_corner_range))[2] - center_height > steppable_around_height_diff ||
        median_image_.at<cv::Vec3f>(y-(int)(steppable_around_corner_range), x+(int)(steppable_around_corner_range))[2] - center_height > steppable_around_height_diff ||
        median_image_.at<cv::Vec3f>(y-(int)(steppable_around_corner_range), x-(int)(steppable_around_corner_range))[2] - center_height > steppable_around_height_diff) {
        continue;
      }
      binarized_image.at<uchar>(y, x) = 255;
      image.at<cv::Vec3b>(y, x)[0] = 200;
      image.at<cv::Vec3b>(y, x)[1] = 200;
      image.at<cv::Vec3b>(y, x)[2] = 200;
    }
  }

  ros::Time d_time = ros::Time::now();

  std::vector<std::vector<cv::Point>> approx_vector;
  std::list<TPPLPoly> polys, result;

  cv::morphologyEx(binarized_image, binarized_image, CV_MOP_CLOSE, cv::noArray(), cv::Point(-1, -1), 2);
  cv::morphologyEx(binarized_image, binarized_image, CV_MOP_OPEN,  cv::noArray(), cv::Point(-1, -1), 2);
  cv::erode(binarized_image, binarized_image, cv::noArray(), cv::Point(-1, -1), 3);
  cv::morphologyEx(binarized_image, binarized_image, CV_MOP_OPEN, cv::noArray(), cv::Point(-1, -1), 2);
  std::vector<std::vector<cv::Point>> contours;
  std::vector<cv::Vec4i> hierarchy;
  cv::findContours(binarized_image, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_NONE);

  ros::Time e_time = ros::Time::now();

  int size_threshold = 5;
  for (int j = 0; j < contours.size(); j++) {
    if (hierarchy[j][3] == -1) { //外側
      if (cv::contourArea(contours[j]) > size_threshold) {
        std::vector<cv::Point> approx;
        cv::approxPolyDP(contours[j], approx, 1.5, true);
        if (approx.size() >= 3) {
          approx_vector.push_back(approx);
          TPPLPoly poly;
          poly.Init(approx.size());
          for (int k = 0; k < approx.size(); k++) {
            poly[k].x = approx[k].x;
            poly[k].y = -approx[k].y;
          }
          polys.push_back(poly);
        }
      }
    } else { //穴
      if (cv::contourArea(contours[j]) > size_threshold) {
        std::vector<cv::Point> approx;
        cv::approxPolyDP(contours[j], approx, 2.0, true);
        if (approx.size() >= 3) {
          approx_vector.push_back(approx);
          TPPLPoly poly;
          poly.Init(approx.size());
          for (int k = 0; k < approx.size(); k++) {
            poly[k].x = approx[k].x;
            poly[k].y = -approx[k].y;
          }
          poly.SetHole(true);
          polys.push_back(poly);
        }
      }
    }
  }

  TPPLPartition pp;
  pp.Triangulate_EC(&polys, &result);

  cv::drawContours(image, approx_vector, -1, cv::Scalar(255, 0, 0));
  ros::Time f_time = ros::Time::now();

  jsk_recognition_msgs::PolygonArray polygon_msg;
  polygon_msg.header = std_msgs::Header();
  polygon_msg.header.frame_id = input->header.frame_id;
  safe_footstep_planner::PolygonArray meshed_polygons_msg;

  int i;
  std::list<TPPLPoly>::iterator iter;
  for (iter = result.begin(), i = 0; iter != result.end(); iter++, i++) {
    geometry_msgs::PolygonStamped ps;
    //for (int j = 0; j < iter->GetNumPoints(); j++) {
    for (int j = iter->GetNumPoints() - 1; j >= 0; j--) {
      image.at<cv::Vec3b>(-iter->GetPoint(j).y, iter->GetPoint(j).x)[2] = 255;
      int p1 = 500 * (-iter->GetPoint(j).y*2) + (iter->GetPoint(j).x*2);
      if (pcl::isFinite(cloud->points[p1])) {
        geometry_msgs::Point32 p;
        p.x = median_image2_.at<cv::Vec3f>(-iter->GetPoint(j).y, iter->GetPoint(j).x)[0];
        p.y = median_image2_.at<cv::Vec3f>(-iter->GetPoint(j).y, iter->GetPoint(j).x)[1];
        p.z = median_image2_.at<cv::Vec3f>(-iter->GetPoint(j).y, iter->GetPoint(j).x)[2];
        //p.x = cloud->points[p1].x;
        //p.y = cloud->points[p1].y;
        //p.z = 0;
        ps.polygon.points.push_back(p);
      } else {
        std::cout << "infinite!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1111i " << std::endl;
      }
    }
    ps.header = std_msgs::Header();
    ps.header.frame_id = input->header.frame_id;
    polygon_msg.polygons.push_back(ps);
    meshed_polygons_msg.polygons.push_back(ps.polygon);
  }

  polygon_publisher_.publish(polygon_msg);
  //meshed_polygons_pub.publish(meshed_polygons_msg);
  image_publisher_.publish(cv_bridge::CvImage(std_msgs::Header(), "bgr8", image).toImageMsg());

  ros::Time end_time = ros::Time::now();
  //std::cout << "all_time " << (end_time - begin_time).sec << "s " << (int)((end_time - begin_time).nsec / 1000000) << "ms" << std::endl;
  //std::cout << "begin_a  " << (a_time - begin_time).sec << "s " << (int)((a_time - begin_time).nsec / 1000000) << "ms" << std::endl;
  //std::cout << "a_b  " << (b_time - a_time).sec << "s " << (int)((b_time - a_time).nsec / 1000000) << "ms" << std::endl;
  //std::cout << "b_c  " << (c_time - b_time).sec << "s " << (int)((c_time - b_time).nsec / 1000000) << "ms" << std::endl;
  //std::cout << "c_d  " << (d_time - c_time).sec << "s " << (int)((d_time - c_time).nsec / 1000000) << "ms" << std::endl;
  //std::cout << "d_e  " << (e_time - d_time).sec << "s " << (int)((e_time - d_time).nsec / 1000000) << "ms" << std::endl;
  //std::cout << "e_f  " << (f_time - e_time).sec << "s " << (int)((f_time - e_time).nsec / 1000000) << "ms" << std::endl;
  //std::cout << "f_end  " << (end_time - f_time).sec << "s " << (int)((end_time - f_time).nsec / 1000000) << "ms" << std::endl;

  //std::cout << x_x_diff << " " << x_y_diff << " " << y_x_diff << " " << y_y_diff << std::endl;




  std::vector<std::vector<Eigen::Vector3f> > meshes;
  std::vector<std::vector<Eigen::Vector3f> > combined_meshes;
  std::vector<std::vector<size_t> > combined_indices;

  // convert to Eigen::Vector3f
  size_t mesh_num(meshed_polygons_msg.polygons.size());
  meshes.resize(mesh_num);
  // std::cerr << "mesh_num : " << mesh_num << std::endl;
  std::vector<bool> is_combined(mesh_num, false);
  for (size_t i = 0; i < mesh_num; i++) {
    size_t vs_num(meshed_polygons_msg.polygons[i].points.size()); // must be 3 (triangle)
    meshes[i].resize(vs_num);
    for (size_t j = 0; j < vs_num; j++) {
      safe_footstep_util::pointsToEigen(meshed_polygons_msg.polygons[i].points[j], meshes[i][j]);
    }
  }

  // debug
  // size_t mesh_num(4);
  // size_t mesh_num(2);
  // std::vector<bool> is_combined(mesh_num, false);
  // meshes.resize(mesh_num);
  // meshes[0].push_back(Eigen::Vector3f(0, 0, 0));
  // meshes[0].push_back(Eigen::Vector3f(500, 0, 0));
  // meshes[0].push_back(Eigen::Vector3f(700, 500, 0));
  // meshes[1].push_back(Eigen::Vector3f(400, 800, 0));
  // meshes[1].push_back(Eigen::Vector3f(700, 500, 0));
  // meshes[1].push_back(Eigen::Vector3f(1000, 700, 0));
  // meshes[2].push_back(Eigen::Vector3f(700, 500, 0));
  // meshes[2].push_back(Eigen::Vector3f(400, 800, 0));
  // meshes[2].push_back(Eigen::Vector3f(0, 0, 0));
  // meshes[3].push_back(Eigen::Vector3f(1000, 700, 0));
  // meshes[3].push_back(Eigen::Vector3f(650, 850, 0));
  // meshes[3].push_back(Eigen::Vector3f(400, 800, 0));
  // meshes[0].push_back(Eigen::Vector3f(0, 0, 0));
  // meshes[0].push_back(Eigen::Vector3f(500, 0, 0));
  // meshes[0].push_back(Eigen::Vector3f(700, 500, 0));
  // meshes[0].push_back(Eigen::Vector3f(1000, 700, 0));
  // meshes[0].push_back(Eigen::Vector3f(650, 850, 0));
  // meshes[0].push_back(Eigen::Vector3f(400, 800, 0));
  // meshes[0].push_back(Eigen::Vector3f(-0.5, -1, 0));
  // meshes[0].push_back(Eigen::Vector3f(-0.5, 1, 0));
  // meshes[0].push_back(Eigen::Vector3f(0.5, 1, 0));
  // meshes[1].push_back(Eigen::Vector3f(-0.5, -1, 0));
  // meshes[1].push_back(Eigen::Vector3f(0.5, 1, 0));
  // meshes[1].push_back(Eigen::Vector3f(0.5, -1, 0));
  // meshes[2].push_back(Eigen::Vector3f(700, -1000, 0));
  // meshes[2].push_back(Eigen::Vector3f(700, 1000, 0));
  // meshes[2].push_back(Eigen::Vector3f(1200, 1000, 0));
  // meshes[3].push_back(Eigen::Vector3f(1200, 1000, 0));
  // meshes[3].push_back(Eigen::Vector3f(1200, -1000, 0));
  // meshes[3].push_back(Eigen::Vector3f(700, -1000, 0));

  // combine meshes
  for (size_t i = 0; i < meshes.size(); i++) {
    std::vector<size_t> is_combined_indices;
    is_combined_indices.push_back(i);
    for (size_t j = i + 1; j < meshes.size(); j++) {
      std::vector<Eigen::Vector3f> inter_v;
      std::vector<Eigen::Vector3f> v1 = meshes[i], v2 = meshes[j];
      std::sort(v1.begin(), v1.end(), safe_footstep_util::compare_eigen3f);
      std::sort(v2.begin(), v2.end(), safe_footstep_util::compare_eigen3f);
      std::set_intersection(v1.begin(), v1.end(), v2.begin(), v2.end(), inserter(inter_v, inter_v.end()), safe_footstep_util::compare_eigen3f);
      if (inter_v.size() == 2) { // adjacent mesh
        std::vector<Eigen::Vector3f> tmp_vs(v1), target_v, tmp_convex;
        std::set_difference(v2.begin(), v2.end(), v1.begin(), v1.end(), inserter(target_v, target_v.end()), safe_footstep_util::compare_eigen3f);
        std::copy(target_v.begin(), target_v.end(), std::back_inserter(tmp_vs));
        safe_footstep_util::calc_convex_hull(tmp_vs, tmp_convex);
        if (tmp_vs.size() == tmp_convex.size()) {
          meshes[i] = tmp_convex;
          meshes[j] = tmp_convex;
          is_combined[j] = true;
          is_combined_indices.push_back(j);
        }
      }
    }
    if (!is_combined[i]) {
      combined_meshes.push_back(meshes[i]);
      combined_indices.push_back(is_combined_indices);
    } else if (is_combined_indices.size() > 1) {
      for (size_t j = 0; j < combined_indices.size(); j++) {
        if (std::find(combined_indices[j].begin(), combined_indices[j].end(), i) != combined_indices[j].end()) {
          combined_meshes[j] = meshes[i];
          combined_indices[j] = is_combined_indices;
        }
      }
    }
    is_combined[i] = true;
  }

  // // add near mesh
  // tf::StampedTransform transform;
  // listener_.lookupTransform("/map", "/rleg_end_coords", ros::Time(0), transform); // map relative to rleg
  // Eigen::Vector3f rfoot_pos;
  // safe_footstep_util::vectorTFToEigen(transform.getOrigin(), rfoot_pos);
  // std::vector<Eigen::Vector3f> near_mesh(4);
  // near_mesh[0] = Eigen::Vector3f(rfoot_pos(0)-0.15, rfoot_pos(1)-0.25, 0);
  // near_mesh[1] = Eigen::Vector3f(rfoot_pos(0)+0.15, rfoot_pos(1)-0.25, 0);
  // near_mesh[2] = Eigen::Vector3f(rfoot_pos(0)+0.15, rfoot_pos(1)+0.45, 0);
  // near_mesh[3] = Eigen::Vector3f(rfoot_pos(0)-0.15, rfoot_pos(1)+0.45, 0);
  // combined_meshes.push_back(near_mesh);

  // convert to safe_footstep_planner::PolygonArray
  size_t combined_mesh_num(combined_meshes.size());
  // std::cerr << "combined_mesh_num : " << combined_mesh_num << std::endl;
  combined_meshes_.polygons.resize(combined_mesh_num);
  for (size_t i = 0; i < combined_mesh_num; i++) {
    size_t vs_num(combined_meshes[i].size());
    combined_meshes_.polygons[i].points.resize(vs_num);
    for (size_t j = 0; j < vs_num; j++) {
      safe_footstep_util::eigenToPoints(combined_meshes[i][j], combined_meshes_.polygons[i].points[j]);
    }
  }

  combined_mesh_publisher_.publish(combined_meshes_);
}

int main(int argc, char **argv)
{
  ros::init(argc, argv, "steppable_region_publisher");
  SteppableRegionPublisher steppable_region_publisher;
  ros::spin();

  return 0;
}
