#include <ros/ros.h>
#include "depth_net.h"
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>

const std::string deploy_prototxt = "/home/ubuntu/catkin_ws/src/ros_caffe/models/depth_net/fast_depth_2s_deploy.prototxt";
const std::string weights_file = "/home/ubuntu/catkin_ws/src/ros_caffe/models/depth_net/fast_depth_2s_iter_5000.caffemodel";
const std::string mean_file = "/home/ubuntu/catkin_ws/src/ros_caffe/models/depth_net/robotDepthRGB_mean.binaryproto";
const std::string CAMERA_TOPIC_NAME = "/usb_cam/image_raw";

DepthNet* depth_net;

void cameraCallback(const sensor_msgs::ImageConstPtr& msg)
{
   cv::Mat img;

   try
   {
      cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(msg, "bgr8");
      img = cv_ptr->image;
   }
   catch (cv_bridge::Exception& e)
   {
      ROS_ERROR("Could not convert from '%s' to 'bgr8'.", msg->encoding.c_str());
      return;
   }

   cv::Mat depth_prediction = depth_net->Predict(img);
   depth_prediction *= 2.5;
   //depth_prediction += 0.5;
   cv::imshow("Depth Prediction", depth_prediction);
   cv::waitKey(1);

   return;
}

int main(int argc, char **argv)
{
   ros::init(argc, argv, "test_depth_net");
   ros::NodeHandle nh;
   image_transport::ImageTransport it(nh);
   image_transport::Subscriber sub = it.subscribe(CAMERA_TOPIC_NAME, 1, cameraCallback);

   depth_net = new DepthNet(deploy_prototxt, weights_file, mean_file);
   cv::namedWindow("Depth Prediction");

   ros::spin();
   delete depth_net;
   cv::destroyWindow("Depth Prediction");
   ros::shutdown();
   return 0;
}
