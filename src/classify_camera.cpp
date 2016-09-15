#include <ros/ros.h>
#include <ros/package.h>
#include <std_msgs/String.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include "Classifier.h"

const std::string RECEIVE_IMG_TOPIC_NAME = "/usb_cam/image_raw"; //"camera/rgb/image_raw";
const std::string PUBLISH_RET_TOPIC_NAME = "/caffe_ret";

Classifier* classifier;
std::string model_path;
std::string weights_path;
std::string mean_file;
std::string label_file;

ros::Publisher gPublisher;

void publishRet(const std::vector<Prediction>& predictions);

void imageCallback(const sensor_msgs::ImageConstPtr& msg) {
    try {
        cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(msg, "bgr8");
        //cv::imwrite("rgb.png", cv_ptr->image);
                cv::Mat img = cv_ptr->image;
                std::vector<Prediction> predictions = classifier->Classify(img);
                publishRet(predictions);
    } catch (cv_bridge::Exception& e) {
        ROS_ERROR("Could not convert from '%s' to 'bgr8'.", msg->encoding.c_str());
    }
}

// TODO: Define a msg or create a service
// Try to receive : $rostopic echo /caffe_ret
void publishRet(const std::vector<Prediction>& predictions)  {
    std_msgs::String msg;
    std::stringstream ss;
    std::cout << " " << std::endl;
    std::cout << "--- Top 5 Predictions ---" << std::endl;
    for (size_t i = 0; i < predictions.size(); ++i) {
        Prediction p = predictions[i];
        ss << "[" << p.second << " - " << p.first << "]" << std::endl;
	std::cout << "[" << p.second << " - " << p.first << "]" << std::endl;
    }
    msg.data = ss.str();
    gPublisher.publish(msg);
}

int main(int argc, char **argv) {
    ros::init(argc, argv, "ros_caffe_test");
    ros::NodeHandle nh;
    image_transport::ImageTransport it(nh);
    image_transport::Subscriber sub = it.subscribe(RECEIVE_IMG_TOPIC_NAME, 1, imageCallback);
        gPublisher = nh.advertise<std_msgs::String>(PUBLISH_RET_TOPIC_NAME, 100);

    const std::string ROOT_SAMPLE = ros::package::getPath("ros_caffe");
    model_path = ROOT_SAMPLE + "/models/bvlc_reference_caffenet/deploy.prototxt";
    weights_path = ROOT_SAMPLE + "/models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel";
    mean_file = ROOT_SAMPLE + "/models/bvlc_reference_caffenet/imagenet_mean.binaryproto";
    label_file = ROOT_SAMPLE + "/models/bvlc_reference_caffenet/synset_words.txt";

    classifier = new Classifier(model_path, weights_path, mean_file, label_file);

    ros::spin();
    delete classifier;
    ros::shutdown();
    return 0;
}
