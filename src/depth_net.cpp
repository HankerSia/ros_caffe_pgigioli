#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/Image.h>
#include "depth_net.h"

const std::string CAMERA_IMAGE_TOPIC_NAME = "/usb_cam/image_raw";

DepthNet::DepthNet(const std::string& deploy_prototxt, const std::string& weights_file,
                   const std::string& mean_file)
{
   caffe::Caffe::set_mode(caffe::Caffe::GPU);

   LoadNetwork(deploy_prototxt, weights_file);
   SetMean(mean_file);
}

void DepthNet::LoadNetwork(const std::string& deploy_prototxt, const std::string& weights_file)
{
   // initialize new net and load weights
   _net.reset(new caffe::Net<float>(deploy_prototxt, caffe::TEST));
   _net->CopyTrainedLayersFrom(weights_file);

   // get layer dimensions and number of channels from input layer
   caffe::Blob<float>* input_layer = _net->input_blobs()[0];
   _num_channels = input_layer->channels();
   _input_geometry = cv::Size(input_layer->width(), input_layer->height());
}

void DepthNet::SetMean(const std::string& mean_file)
{
   // load mean.binaryproto file
   caffe::BlobProto blob_proto;
   ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);

   // convert BlobProto to Blob<float>
   caffe::Blob<float> mean_blob;
   mean_blob.FromProto(blob_proto);

   // extract data from BGR channels of mean blob
   std::vector<cv::Mat> channels;
   float* mean_blob_data = mean_blob.mutable_cpu_data();
   for (int i = 0; i < _num_channels; ++i)
   {
      // create one channel with data from mean_blob
      cv::Mat channel(mean_blob.height(), mean_blob.width(), CV_32FC1, mean_blob_data);
      channels.push_back(channel);

      // move the memory address to next channel data
      mean_blob_data += mean_blob.height() * mean_blob.width();
   }

   // merge separate channels into a single image
   cv::Mat mean;
   cv::merge(channels, mean);

   // compute global mean pixel value and fill mean image with this value
   cv::Scalar channel_mean = cv::mean(mean);
   _mean = cv::Mat(_input_geometry, mean.type(), channel_mean);
}

cv::Mat DepthNet::Predict(const cv::Mat& img)
{
   caffe::Blob<float>* input_layer = _net->input_blobs()[0];

   // reshape input layer to lead with dimension 1
   input_layer->Reshape(1, _num_channels, _input_geometry.height, _input_geometry.width);

   // apply reshape to full net
   _net->Reshape();

   // create wrapper for input layer channels
   std::vector<cv::Mat> input_channels;
   WrapInputLayer(&input_channels);

   // preprocess image and apply changes directly to input layer
   Preprocess(img, &input_channels);

   _net->ForwardPrefilled();

   caffe::Blob<float>* output_layer = _net->output_blobs()[0];
   float* output_data = output_layer->mutable_cpu_data();
   cv::Mat result(output_layer->height(), output_layer->width(), CV_32FC1, output_data);

   return result;
}

void DepthNet::WrapInputLayer(std::vector<cv::Mat>* input_channels)
{
   caffe::Blob<float>* input_layer = _net->input_blobs()[0];

   // extract dimensions and data of input layer
   int width = input_layer->width();
   int height = input_layer->height();
   float* input_data = input_layer->mutable_cpu_data();

   // create an image with each channel pointing to the channel of the input layer
   for (int i = 0; i < input_layer->channels(); ++i)
   {
      cv::Mat channel(height, width, CV_32FC1, input_data);
      input_channels->push_back(channel);
      input_data += width * height;
   }
}

void DepthNet::Preprocess(const cv::Mat& img, std::vector<cv::Mat>* input_channels)
{
   // convert the input image to the input image format of network
   cv::Mat sample;
   if (img.channels() == 3 && _num_channels == 1)
      cv::cvtColor(img, sample, CV_BGR2GRAY);
   else if (img.channels() == 4 && _num_channels == 1)
      cv::cvtColor(img, sample, CV_BGRA2GRAY);
   else if (img.channels() == 4 && _num_channels == 3)
      cv::cvtColor(img, sample, CV_BGRA2BGR);
   else if (img.channels() == 1 && _num_channels == 3)
      cv::cvtColor(img, sample, CV_GRAY2BGR);
   else
      sample = img;

   // convert input image size to input image size of network
   cv::Mat sample_resized;
   if (sample.size() != _input_geometry)
      cv::resize(sample, sample_resized, _input_geometry);
   else
      sample_resized = sample;

   // convert input image type to float
   cv::Mat sample_float;
   if (_num_channels == 3)
      sample_resized.convertTo(sample_float, CV_32FC3);
   else
      sample_resized.convertTo(sample_float, CV_32FC1);

   cv::Mat sample_subtracted_mean;
   cv::subtract(sample_float, _mean, sample_subtracted_mean);

   float scale = 1.0/255.0;
   cv::Mat sample_scaled = sample_subtracted_mean * scale;

   cv::split(sample_scaled, *input_channels);
}
