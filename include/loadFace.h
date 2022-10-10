// Writen by Horia-Gabriel Radu - 2021-2022
// for Third Year Project at University of Manchester

#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/face.hpp>
#include <dlib/opencv.h>
#include <dlib/dnn.h>
#include <dlib/image_processing.h>
#include <dlib/data_io.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_io.h>
#include <dlib/gui_widgets.h>

#include <helpers.h>

#ifndef LOADFACE_H
#define LOADFACE_H

std::vector<std::tuple<int, std::string>> loadFaceLBPH(cv::Mat *image, std::vector<std::tuple<int, int, int ,int>> *faces, cv::Ptr<cv::face::LBPHFaceRecognizer> *model, std::vector<std::tuple<int, std::string>> *labels, dlib::shape_predictor *shape_model, std::string inputName, bool getInputName);

std::vector<dlib::matrix<float, 0, 1>> loadFaceML(cv::Mat *image, std::vector<std::tuple<int, int, int ,int>> *faces, anet_type *net, std::vector<std::tuple<int, std::string>> *labels, dlib::shape_predictor *sp, std::string inputName, bool getInputName);

std::vector<std::tuple<int, std::string>> loadFace(std::string classifierOption, cv::Ptr<cv::face::EigenFaceRecognizer> *modelEigen, cv::Ptr<cv::face::FisherFaceRecognizer> *modelFisher, std::vector<std::tuple<int, std::string>> *labels, dlib::shape_predictor *shape_model, std::vector<std::tuple<std::string, std::string>> *modelFaces, bool doEigen, bool doFisher, bool noPath, bool getInputName);

#endif