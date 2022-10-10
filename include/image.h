// Writen by Horia-Gabriel Radu - 2021-2022
// for Third Year Project at University of Manchester

#include <string>
#include <iostream>

#include <opencv2/highgui.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/face.hpp>

#include <dlib/opencv.h>
#include <dlib/image_processing.h>
#include <dlib/dnn.h>
#include <dlib/data_io.h>

#include <helpers.h>

#ifndef IMAGE_H
#define IMAGE_H

std::vector<std::tuple<int, int, int ,int>> processImage(bool display, cv::Mat *image, std::string classifierOption, bool recogniseFaceBool, bool recognizerML, bool doEmotion, cv::Ptr<cv::face::LBPHFaceRecognizer> *model,  cv::Ptr<cv::face::EigenFaceRecognizer> *modelEigen, cv::Ptr<cv::face::FisherFaceRecognizer> *modelFisher, std::vector<std::tuple<int, std::string>> *labels, dlib::shape_predictor *shape_model, std::vector<dlib::matrix<float, 0, 1>> *face_descriptors, anet_type *anet, dlib::shape_predictor *sp, bool doEigen, bool doFisher);

#endif