#include <ros/ros.h>
#include "depth_net.h"

DepthNet* depth_net;
const std::string deploy_prototxt = "/home/ubuntu/catkin_ws/src/ros_caffe/models/depth_net/fast_depth_2s_deploy.prototxt";
const std::string weights_file = "/home/ubuntu/catkin_ws/src/ros_caffe/models/depth_net/fast_depth_2s_iter_5000.caffemodel";
const std::string mean_file = "/home/ubuntu/catkin_ws/src/ros_caffe/models/depth_net/robotDepthRGB_mean.binaryproto";
const std::string test_image = "/home/ubuntu/catkin_ws/src/ros_caffe/00000_rgb.jpg";

int main(int argc, char **argv)
{
   ros::init(argc, argv, "test_depth_net");

   depth_net = new DepthNet(deploy_prototxt, weights_file, mean_file);

   cv::Mat img = cv::imread(test_image);
   cv::Mat depth_prediction = depth_net->Predict(img);

   double min, max;
   cv::minMaxLoc(depth_prediction, &min, &max);
   std::cout << "min: " << min << " max: " << max << std::endl;

   while (ros::ok())
   {
      cv::imshow("Depth Prediction", depth_prediction);
      cv::waitKey(3);
   }

   delete depth_net;
   return 0;
}
