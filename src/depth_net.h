#ifndef DEPTHNET_H
#define DEPTHNET_H

#include <iostream>
#include <vector>
#include <sstream>
#include <caffe/caffe.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/gpu/gpu.hpp>

class DepthNet
{
   public:
      DepthNet(const std::string& deploy_prototxt,
               const std::string& weights_file,
               const std::string& mean_file);

      cv::Mat Predict(const cv::Mat& img);

   private:
      void LoadNetwork(const std::string& deploy_prototxt,
                       const std::string& weights_file);

      void SetMean(const std::string& mean_file);

      void Preprocess(const cv::Mat& img,
                      std::vector<cv::Mat>* input_channels);

      void WrapInputLayer(std::vector<cv::Mat>* input_channels);

   private:
      caffe::shared_ptr<caffe::Net<float> > _net;
      int _num_channels;
      cv::Size _input_geometry;
      cv::Mat _mean;
};

#endif
