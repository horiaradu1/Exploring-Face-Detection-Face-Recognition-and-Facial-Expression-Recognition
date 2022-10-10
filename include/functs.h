// Writen by Horia-Gabriel Radu - 2021-2022
// for Third Year Project at University of Manchester

#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/dnn.hpp>

#include <dlib/opencv.h>
#include <dlib/dnn.h>
#include <dlib/image_processing.h>
#include <dlib/image_processing/frontal_face_detector.h>

#ifndef FUNCTS_H
#define FUNCTS_H

std::vector<std::tuple<int, int, int ,int>> encloseFaceDNN(cv::Mat *image, cv::dnn::Net *net, std::string option);

std::vector<std::tuple<int, int, int ,int>> encloseFaceHaar(cv::Mat *image, cv::CascadeClassifier *classifier);

std::vector<std::tuple<int, int, int ,int>> encloseFaceHog(cv::Mat *image, dlib::frontal_face_detector *hogFaceDetector);

// Network definition taken from http://dlib.net/dnn_mmod_ex.cpp.html

template <long num_filters, typename SUBNET> using con5d = dlib::con<num_filters,5,5,2,2,SUBNET>;
template <long num_filters, typename SUBNET> using con5  = dlib::con<num_filters,5,5,1,1,SUBNET>;

template <typename SUBNET> using downsampler  = dlib::relu<dlib::affine<con5d<32, dlib::relu<dlib::affine<con5d<32, dlib::relu<dlib::affine<con5d<16,SUBNET>>>>>>>>>;
template <typename SUBNET> using rcon5  = dlib::relu<dlib::affine<con5<45,SUBNET>>>;

using net_type = dlib::loss_mmod<dlib::con<1,9,9,1,1,rcon5<rcon5<rcon5<downsampler<dlib::input_rgb_image_pyramid<dlib::pyramid_down<6>>>>>>>>;

std::vector<std::tuple<int, int, int ,int>> encloseFaceMMOD(cv::Mat *image, net_type *mmodFaceDetector);

std::vector<std::tuple<int, int, int ,int>> encloseFaceYOLO(cv::Mat *image, cv::dnn::Net *net, std::vector<cv::String> outLayers);

void predictEmotion(cv::Mat *image, std::tuple<int, int, int ,int> *face, std::vector<std::tuple<int, float>> *emotions, cv::dnn::Net *netEmotion);

#endif