#include <ros/ros.h>
#include <ros/package.h>
#include <std_msgs/String.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include "Classifier.h"
#include <ctime>

Classifier* classifier;
std::string model_path;
std::string weights_path;
std::string mean_file;
std::string label_file;
std::string image_path;
std::string images_dir;
std::string image;

int main(int argc, char **argv) {
    // To receive an image from the topic, PUBLISH_RET_TOPIC_NAME
    //const std::string ROOT_SAMPLE = ros::package::getPath("ros_caffe");
    model_path = "/media/ubuntu/face_recognition/vggdeploy.prototxt";
    weights_path = "/media/ubuntu/face_recognition/vgg_ft_iter_100000.caffemodel";
    mean_file = "/media/ubuntu/face_recognition/imagenet_mean.binaryproto";
    label_file = "/media/ubuntu/face_recognition/synset_words.txt";
    //image_path = "/media/ubuntu/face_recognition/sample_pics/face_rec_test.jpg";
    images_dir = "/media/ubuntu/face_recognition/sample_pics/";
    image = argv[1];
    image_path = images_dir + image;

    std::cout << "image path: " << image_path << std::endl;

    classifier = new Classifier(model_path, weights_path, mean_file, label_file);

    // Test data/cat.jpg
    cv::Mat img = cv::imread(image_path, -1);
    std::cout << "start clock" << std::endl;
    std::clock_t start;
    double duration;
    start = clock();

    std::vector<Prediction> predictions = classifier->Classify(img);

    duration = (clock() - start)/(double)CLOCKS_PER_SEC;
    std::cout << "Prediction time: " << duration << std::endl;
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
