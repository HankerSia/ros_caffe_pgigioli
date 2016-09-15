/*
 * ros_caffe_test.cpp
 *
 *  Created on: Aug 31, 2015
 *      Author: Tzutalin
 */

#include <ros/ros.h>
#include <ros/package.h>
#include <std_msgs/String.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include "Classifier.h"

Classifier* classifier;
std::string model_path;
std::string weights_path;
std::string mean_file;
std::string label_file;
std::string image_path;

int main(int argc, char **argv) {
    // To receive an image from the topic, PUBLISH_RET_TOPIC_NAME
    //const std::string ROOT_SAMPLE = ros::package::getPath("ros_caffe");
    model_path = "/home/ubuntu/catkin_ws/src/ros_caffe/caffe/models/face_identifier/deploy.prototxt";
    weights_path = "/home/ubuntu/catkin_ws/src/ros_caffe/caffe/models/face_identifier/snapshot_iter_168.caffemodel";
    mean_file = "/home/ubuntu/catkin_ws/src/ros_caffe/caffe/models/face_identifier/mean.binaryproto";
    label_file = "/home/ubuntu/catkin_ws/src/ros_caffe/caffe/models/face_identifier/labels.txt";
    image_path = "/home/ubuntu/caffe/models/face_network/cropped_0.jpg";

    classifier = new Classifier(model_path, weights_path, mean_file, label_file);

    // Test data/cat.jpg
    cv::Mat img = cv::imread(image_path, -1);
    std::vector<Prediction> predictions = classifier->Classify(img);
    /* Print the top N predictions. */
    std::cout << "Test facial recognizer" << std::endl;
    for (size_t i = 0; i < predictions.size(); ++i) {
        Prediction p = predictions[i];
        std::cout << std::fixed << std::setprecision(4) << p.second << " - \"" << p.first << "\"" << std::endl;
    }

    //ros::spin()
    delete classifier;
    //ros::shutdown();
    return 0;
}
