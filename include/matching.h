// Writen by Horia-Gabriel Radu - 2021-2022
// for Third Year Project at University of Manchester

#include <string>
#include <iostream>

#include <opencv2/highgui.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/face.hpp>

#include <helpers.h>

#ifndef MATCHING_H
#define MATCHING_H

cv::Mat prepareFace (cv::Rect *opencv_rect, dlib::shape_predictor *shape_model, cv::Mat *image);

void recogniseFacesML(int i, cv::Mat *image, std::tuple<int, int, int ,int> *faces, std::vector<std::tuple<int, std::string, double, float>> *names, std::vector<std::tuple<int, std::string>> *labels, std::vector<dlib::matrix<float, 0, 1>> *face_descriptors, anet_type *anet, dlib::shape_predictor *sp);

void recogniseFaces(int i, cv::Mat *image, std::tuple<int, int, int ,int> *face, std::vector<std::tuple<int, std::string, double, float>> *names, cv::Ptr<cv::face::LBPHFaceRecognizer> *model, std::vector<std::tuple<int, std::string>> *labels, dlib::shape_predictor *shape_model);

void recogniseFacesEigen(int i, cv::Mat *image, std::tuple<int, int, int ,int> *face, std::vector<std::tuple<int, std::string, double, float>> *names, cv::Ptr<cv::face::EigenFaceRecognizer> *model, std::vector<std::tuple<int, std::string>> *labels, dlib::shape_predictor *shape_model);

void recogniseFacesFisher(int i, cv::Mat *image, std::tuple<int, int, int ,int> *face, std::vector<std::tuple<int, std::string, double, float>> *names, cv::Ptr<cv::face::FisherFaceRecognizer> *model, std::vector<std::tuple<int, std::string>> *labels, dlib::shape_predictor *shape_model);


#endif