#include <ros/ros.h>
#include "depth_net.h"

DepthNet* depth_net;
//const std::string deploy_prototxt = "/home/ubuntu/catkin_ws/src/ros_caffe/models/depth_net/fast_depth_coarse_deploy.prototxt";
//const std::string weights_file = "/home/ubuntu/catkin_ws/src/ros_caffe/models/depth_net/fast_depth_coarse_iter_25000.caffemodel";
//const std::string mean_file = "/home/ubuntu/catkin_ws/src/ros_caffe/models/depth_net/robotDepthRGB_160x120mean.binaryproto";

const std::string deploy_prototxt = "/home/ubuntu/catkin_ws/src/ros_caffe/models/depth_net/VGG_depth_net_coarse_80x60_deploy.prototxt";
const std::string weights_file = "/home/ubuntu/catkin_ws/src/ros_caffe/models/depth_net/VGG_depth_net_coarse_80x60_rel_iter_150000.caffemodel";
const std::string mean_file = "/home/ubuntu/catkin_ws/src/ros_caffe/models/depth_net/robotDepthRGB_80x60mean.binaryproto";

const std::string test_image = "/home/ubuntu/catkin_ws/src/ros_caffe/00000_rgb.jpg";

int main(int argc, char **argv)
{
   ros::init(argc, argv, "test_depth_net");

   depth_net = new DepthNet(deploy_prototxt, weights_file, mean_file);

   cv::Mat img = cv::imread(test_image);
   cv::Mat img_resized;
   resize(img, img_resized, cv::Size(80,60), 0, 0, cv::INTER_CUBIC);
   cv::Mat depth_prediction = depth_net->Predict(img_resized);

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
