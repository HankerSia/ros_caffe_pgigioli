#include <ros/ros.h>
#include <ros/package.h>
#include <std_msgs/String.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include "Classifier.h"
#include <face_tracker/imageArray.h>
#include <face_tracker/stringArray.h>

using namespace std;
using namespace cv;

Classifier* classifier;
string model_path = "/media/ubuntu/face_recognition/vggdeploy.prototxt";
string weights_path = "/media/ubuntu/face_recognition/vgg_ft_iter_100000.caffemodel";
string mean_file = "/media/ubuntu/face_recognition/imagenet_mean.binaryproto";
string label_file = "/media/ubuntu/face_recognition/synset_words.txt";

void printResults(const vector<vector<Prediction> >& predictions);
void publishLabels(const vector<vector<Prediction> >& predictionsArray);

class faceRecognition
{
   ros::NodeHandle nh;
   ros::Subscriber faces_sub;
   ros::Publisher labels_pub;

public:
   faceRecognition()
   {
      faces_sub = nh.subscribe("/cropped_faces", 1, &faceRecognition::imageCallback, this);
      labels_pub = nh.advertise<face_tracker::stringArray>("/face_labels", 1);
   }

private:
   void imageCallback(const face_tracker::imageArray msg)
   {
      int num_faces = msg.images.size();
      vector<vector<Prediction> > predictionsArray;

      // iterate through cropped faces and classify
      for (int i = 0; i < num_faces; i++) {
         try {
            cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(msg.images[i], "bgr8");
	    Mat img = cv_ptr->image;
	    vector<Prediction> predictions = classifier->Classify(img);
	    predictionsArray.push_back(predictions);
         } catch (cv_bridge::Exception& e) {
	    ROS_ERROR("Could not convert from to 'bgr8'.");
         }
      }

      // call function print results on screen
      printResults(predictionsArray);

      // call function to publish labels of faces found
      publishLabels(predictionsArray);
   }

   void printResults(const vector<vector<Prediction> >& predictionsArray)
   {
      cout << " " << endl;
      cout << "--- Top 5 Predictions ---" << endl;

      // iterate through predictions and print on screen
      for (int j = 0; j < predictionsArray.size(); j++) {
         cout << " " << endl;
	 vector<Prediction> predictions = predictionsArray[j];

	 for (size_t i = 0; i < predictions.size(); ++i) {
	    Prediction p = predictions[i];
	    cout << "[" << p.second << " - " << p.first << "]" << endl;
   	 }
      }
   }

   void publishLabels(const vector<vector<Prediction> >& predictionsArray)
   {
      face_tracker::stringArray labels;

      // iterate through predictions and get only top labels
      for (int i = 0; i < predictionsArray.size(); i++) {
	 vector<Prediction> predictions = predictionsArray[i];
	 Prediction p = predictions[0];

	 // only publish the label if the probability is greater than 75%
	 if (p.second > 0.75) {
	    stringstream ss;
	    std_msgs::String name;
	    ss << p.first;
	    name.data = ss.str();

	    labels.labels.push_back(name);
	 } else {
	    std_msgs::String name;
	    name.data = "?";
	    labels.labels.push_back(name);
	 }
       }

       labels_pub.publish(labels);
   }
};

int main(int argc, char **argv)
{
   ros::init(argc, argv, "face_recognition_camera");

   classifier = new Classifier(model_path, weights_path, mean_file, label_file);

   faceRecognition fr;
   ros::spin();
   delete classifier;
   return 0;
}
