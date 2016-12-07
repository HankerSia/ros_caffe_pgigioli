#include <ros/ros.h>
#include "depth_net.h"
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>

//const std::string deploy_prototxt = "/home/ubuntu/catkin_ws/src/ros_caffe/models/depth_net/VGG_depth_net_coarse_80x60_deploy.prototxt";
//const std::string weights_file = "/home/ubuntu/catkin_ws/src/ros_caffe/models/depth_net/VGG_depth_net_coarse_80x60_rel_iter_150000.caffemodel";
//const std::string mean_file = "/home/ubuntu/catkin_ws/src/ros_caffe/models/depth_net/robotDepthRGB_80x60mean.binaryproto";

const std::string deploy_prototxt = "/home/ubuntu/catkin_ws/src/ros_caffe/models/depth_net/VGG_depth_net_fine_80x60_deploy.prototxt";
const std::string weights_file = "/home/ubuntu/catkin_ws/src/ros_caffe/models/depth_net/VGG_depth_net_fine_80x60_rel_iter_20000.caffemodel";
const std::string mean_file = "/home/ubuntu/catkin_ws/src/ros_caffe/models/depth_net/robotDepthRGB_80x60mean.binaryproto";

//const std::string deploy_prototxt = "/home/ubuntu/catkin_ws/src/ros_caffe/models/depth_net/fast_depth_coarse_80x60_deploy.prototxt";
//const std::string weights_file = "/home/ubuntu/catkin_ws/src/ros_caffe/models/depth_net/fast_depth_coarse_80x60_iter_30000.caffemodel";
//const std::string mean_file = "/home/ubuntu/catkin_ws/src/ros_caffe/models/depth_net/robotDepthRGB_80x60mean.binaryproto";

//const std::string deploy_prototxt = "/home/ubuntu/catkin_ws/src/ros_caffe/models/depth_net/fast_depth_coarse_log_80x60_deploy.prototxt";
//const std::string weights_file = "/home/ubuntu/catkin_ws/src/ros_caffe/models/depth_net/fast_depth_coarse_log_80x60_iter_50000.caffemodel";
//const std::string mean_file = "/home/ubuntu/catkin_ws/src/ros_caffe/models/depth_net/robotDepthRGB_80x60mean.binaryproto";

const std::string CAMERA_TOPIC_NAME = "/usb_cam/image_raw";
static int FRAME_W;
static int FRAME_H;
static int INPUT_W;
static int INPUT_H;

DepthNet* depth_net;

class CameraDepthNet
{
   ros::NodeHandle nh;
   image_transport::ImageTransport it;
   image_transport::Subscriber camera_sub;
   image_transport::Publisher depth_predictions_pub;

public:
   CameraDepthNet() : it(nh)
   {
      camera_sub = it.subscribe(CAMERA_TOPIC_NAME, 1,
                               &CameraDepthNet::cameraCallback,this);
      depth_predictions_pub = it.advertise("depth_prediction", 1);
      cv::namedWindow("Depth Prediction", cv::WINDOW_NORMAL);
   }

   ~CameraDepthNet()
   {
      cv::destroyWindow("Depth Prediction");
   }
private:
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

      cv_bridge::CvImage depth_msg;
      depth_msg.header = msg->header;
      depth_msg.encoding = sensor_msgs::image_encodings::TYPE_32FC1;

      cv::Mat depth_prediction;

      if (INPUT_W == FRAME_W && INPUT_H == FRAME_H)
      {
         depth_prediction = depth_net->Predict(img);
      }
      else
      {
         cv::Mat img_resized;
         cv::resize(img, img_resized, cv::Size(INPUT_W, INPUT_H), 0, 0, cv::INTER_LINEAR);
         cv::Mat depth_prediction_resized = depth_net->Predict(img_resized);

         cv::resize(depth_prediction_resized, depth_prediction, cv::Size(FRAME_W, FRAME_H), 0, 0, cv::INTER_LINEAR);
      }

      depth_msg.image = depth_prediction;
      depth_predictions_pub.publish(depth_msg.toImageMsg());

      //cv::exp(depth_prediction, depth_prediction);
      depth_prediction += 3.0;
      depth_prediction /= 6.0;
double minVal;
double maxVal;
cv::Point minLoc;
cv::Point maxLoc;
minMaxLoc(depth_prediction, &minVal, &maxVal, &minLoc, &maxLoc);
std::cout << "min : " << minVal << " max : " << maxVal << std::endl;

      cv::imshow("Depth Prediction", depth_prediction);
      cv::waitKey(1);

      return;
   }
};

int main(int argc, char **argv)
{
   ros::init(argc, argv, "test_depth_net");

   ros::param::get("/usb_cam/image_width", FRAME_W);
   ros::param::get("/usb_cam/image_height", FRAME_H);

   depth_net = new DepthNet(deploy_prototxt, weights_file, mean_file);
   cv::Size input_size = depth_net->Get_input_geometry();
   INPUT_W = input_size.width;
   INPUT_H = input_size.height;

   CameraDepthNet cdn;

   ros::spin();
   delete depth_net;
   ros::shutdown();
   return 0;
}
