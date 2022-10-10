// Writen by Horia-Gabriel Radu - 2021-2022
// for Third Year Project at University of Manchester

#include <iostream>
#include <fstream>
#include <sstream>
#include <opencv2/highgui.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/core/ocl.hpp>
#include <opencv2/face.hpp>

#include <dlib/opencv.h>
#include <dlib/dnn.h>
#include <dlib/image_processing.h>
#include <dlib/data_io.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_io.h>
#include <dlib/gui_widgets.h>
#include <dlib/clustering.h>

#include <functs.h>
#include <matching.h>
#include <helpers.h>

using namespace std;

std::vector<std::tuple<int, std::string>> loadFaceLBPH(cv::Mat *image, std::vector<std::tuple<int, int, int ,int>> *faces, cv::Ptr<cv::face::LBPHFaceRecognizer> *model, std::vector<std::tuple<int, string>> *labels, dlib::shape_predictor *shape_model, std::string inputName, bool getInputName){
    std::vector<int> labels_temp;
    std::vector<cv::Mat> faces_final;
    std::string name;
    cv::Mat face, face_flip;

    for (int i = 0; i < (*faces).size(); i++)
    {
        std::vector<std::tuple<cv::Mat, cv::Mat>> returned_faces; 
        std::vector<cv::Mat> augmented_images;
        augmented_images.clear();
        cv::Mat img_higher_contrast;
        (*image).convertTo(img_higher_contrast, -1, 2, 0);

        cv::Mat img_lower_contrast;
        (*image).convertTo(img_lower_contrast, -1, 0.5, 0);

        cv::Mat img_higher_brightness;
        (*image).convertTo(img_higher_brightness, -1, 1, 50);

        cv::Mat img_lower_brightness;
        (*image).convertTo(img_lower_brightness, -1, 1, -50);

        // cv::Mat img_lower_resolution;
        // cv::resize((*image), img_lower_resolution, cv::Size(image->cols*0.75, image->rows*0.75));
        // cv::resize(img_lower_resolution, img_lower_resolution, cv::Size(image->cols, image->rows));

        augmented_images.push_back((*image));
        augmented_images.push_back(img_higher_contrast);
        augmented_images.push_back(img_lower_contrast);
        augmented_images.push_back(img_higher_brightness);
        augmented_images.push_back(img_lower_brightness);
        //augmented_images.push_back(img_lower_resolution);

        cv::Rect opencvRect(get<2>((*faces)[i]), get<0>((*faces)[i]), get<3>((*faces)[i]), get<1>((*faces)[i]));

        for (int j = 0; j < augmented_images.size(); j++){
            face = augmented_images[j](cv::Range(get<0>((*faces)[i]), get<1>((*faces)[i])), cv::Range(get<2>((*faces)[i]), get<3>((*faces)[i])));
            face = prepareFace(&opencvRect, shape_model, &augmented_images[j]);
            cv::flip(face, face_flip, 1);

            if (!getInputName){
                cv::imshow("Face Detection - Loaded Faces", face);
                cv::waitKey(0);
                cv::destroyAllWindows();
            }
            returned_faces.push_back(make_tuple(face, face_flip));
        }
        
        if (getInputName){
            name = inputName;
            cout << name << " ADD TO RECOGNIZER\n";
        }else{
            cout << "ENTER NAME -> ";
            cin >> name;
            if (name == "NO" || name == "no"){
                cout << "NOT ADDING TO RECOGNIZER\n";
                continue;
            }
        }
        
        auto iter = std::find_if((*labels).begin(), (*labels).end(), [=](std::tuple<int, string> item) {return get<1>(item) == name;});
        
        if (iter != (*labels).end()){
            cout << "FOUND PERSON IN LABELS\n";
            for (int m = 0; m < returned_faces.size(); m++){
                faces_final.push_back(get<0>(returned_faces[m]));
                faces_final.push_back(get<1>(returned_faces[m]));
                labels_temp.push_back(get<0>(*iter));
                labels_temp.push_back(get<0>(*iter));
            }
        }else{
            cout << "NOT FOUND PERSON IN LABELS\n";
            int lSize = (*labels).size();
            (*labels).push_back(make_tuple(lSize, name));
            for (int m = 0; m < returned_faces.size(); m++){
                faces_final.push_back(get<0>(returned_faces[m]));
                faces_final.push_back(get<1>(returned_faces[m]));
                
                labels_temp.push_back(lSize);
                labels_temp.push_back(lSize);
            }    
        }
        cout << "ADDING THIS FACE TO RECOGNIZER\n";
    }
    (*model)->update(faces_final, labels_temp);
    
    labels_temp.clear();
    faces_final.clear();

    return (*labels);
}

// Neural network info taken from http://dlib.net/dnn_face_recognition_ex.cpp.html
std::vector<dlib::matrix<float, 0, 1>> loadFaceML(cv::Mat *image, std::vector<std::tuple<int, int, int ,int>> *faces, anet_type *net, std::vector<std::tuple<int, std::string>> *labels, dlib::shape_predictor *sp, std::string inputName, bool getInputName){
    std::string name;
    //std::vector<dlib::matrix<float, 0, 1>> face_descriptors;
    std::vector<dlib::matrix<dlib::rgb_pixel>> faces_chip;
    for (int i = 0; i < (*faces).size(); i++){
        cv::Mat face = (*image)(cv::Range(get<0>((*faces)[i]), get<1>((*faces)[i])), cv::Range(get<2>((*faces)[i]), get<3>((*faces)[i])));//.clone();
        cv::Rect face_rect(get<2>((*faces)[i]), get<0>((*faces)[i]), get<3>((*faces)[i]), get<1>((*faces)[i]));

        dlib::cv_image<dlib::rgb_pixel> dlibImg((*image));
        dlib::rectangle dlibRect = openCVtoDlib(&face_rect);

        auto shape = (*sp)(dlibImg, dlibRect);
        dlib::matrix<dlib::rgb_pixel> face_chip;
        dlib::extract_image_chip(dlibImg, dlib::get_face_chip_details(shape, 150, 0.25), face_chip);
        
        if (!getInputName){
            cv::imshow("Face Detection - Loaded Faces", face);
            cv::waitKey(0);
            cv::destroyAllWindows();
        }

        if (getInputName){
            name = inputName;
            cout << name << " ADD TO RECOGNIZER\n";
        }else{
            cout << "ENTER NAME -> ";
            cin >> name;
            if (name == "NO" || name == "no"){
                cout << "NOT ADDING TO RECOGNIZER\n";
                continue;
            }
        }

        faces_chip.push_back(face_chip);
        
        int lSize = (*labels).size();
        (*labels).push_back(make_tuple(lSize, name));
        cout << "ADDING THIS FACE TO RECOGNIZER\n";
    }

    std::vector<dlib::matrix<float, 0, 1>> face_descriptors = (*net)(faces_chip);

    return face_descriptors;
}


std::vector<std::tuple<int, std::string>> loadFace(string classifierOption, cv::Ptr<cv::face::EigenFaceRecognizer> *modelEigen, cv::Ptr<cv::face::FisherFaceRecognizer> *modelFisher, std::vector<std::tuple<int, string>> *labels, dlib::shape_predictor *shape_model, std::vector<std::tuple<std::string, std::string>> *modelFaces, bool doEigen, bool doFisher, bool noPath, bool getInputName){
    std::vector<int> labels_temp;
    std::vector<cv::Mat> faces_final;
    cv::CascadeClassifier classifierHaar;
    cv::dnn::Net net;
    dlib::frontal_face_detector hogFaceDetector;
    net_type mmodFaceDetector;
    std::vector<cv::String> outLayers;
    if (classifierOption == "haar")
    {
        classifierHaar.load("models/haarcascade_frontalface_default.xml");
    }
    else if (classifierOption == "caffe")
    {
        const std::string caffeConfigFile = "models/deploy.prototxt";
        const std::string caffeWeightFile = "models/res10_300x300_ssd_iter_140000_fp16.caffemodel";
        net = cv::dnn::readNetFromCaffe(caffeConfigFile, caffeWeightFile);
        net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
        net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
    }
    else if (classifierOption == "tensor")
    {
        const std::string tensorflowConfigFile = "models/opencv_face_detector.pbtxt";
        const std::string tensorflowWeightFile = "models/opencv_face_detector_uint8.pb";
        net = cv::dnn::readNetFromTensorflow(tensorflowWeightFile, tensorflowConfigFile);
        net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
        net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
    }
    else if (classifierOption == "hog")
    {
        hogFaceDetector = dlib::get_frontal_face_detector();
    }
    else if (classifierOption == "mmod")
    {
        cv::String mmodModel = "models/mmod_human_face_detector.dat";
        dlib::deserialize(mmodModel) >> mmodFaceDetector;
    }else if (classifierOption == "yolo")
    {
        const std::string yoloFaceWeightFile = "models/yolov3-wider_16000.weights";
        const std::string yoloFaceConfigFile = "models/yolov3-face.cfg";
        net = cv::dnn::readNetFromDarknet(yoloFaceConfigFile, yoloFaceWeightFile);
        net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
        net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
        std::vector<cv::String> layerNames = net.getLayerNames();
        cv::dnn::MatShape unconnectedOutLayers = net.getUnconnectedOutLayers();
        for (int i = 0; i < unconnectedOutLayers.size(); i++)
        {
            outLayers.push_back(layerNames[unconnectedOutLayers[0] - 1]);
            outLayers.push_back(layerNames[unconnectedOutLayers[1] - 1]);
            outLayers.push_back(layerNames[unconnectedOutLayers[2] - 1]);
            // yolo_82
            // yolo_94
            // yolo_106
        }
    }

    for (int f = 0; f < (*modelFaces).size(); f++)
    {
        cv::Mat face, face_flip;
        std::string name;
        cv::Mat image;
        if (noPath){
            image = cv::imread(get<0>((*modelFaces)[f]), cv::IMREAD_COLOR);
        }else{
            image = cv::imread("images/" + get<0>((*modelFaces)[f]) + ".jpg", cv::IMREAD_COLOR);
        }
        if (! image.data){
            cout << "Could not open the file: " << get<0>((*modelFaces)[f]) << endl;
            continue;
        }
        cv::Mat imageClone = image.clone();
        if (image.cols < 300 && image.rows < 300){
            resizeKeepAspectRatio(&imageClone, cv::Size(300, 300), cv::Scalar(0));
        }else if(image.cols > image.rows){
            resizeKeepAspectRatio(&imageClone, cv::Size(image.cols, image.cols), cv::Scalar(0));
        }else{
            resizeKeepAspectRatio(&imageClone, cv::Size(image.rows, image.rows), cv::Scalar(0));
        }
        std::vector<std::tuple<int, int, int ,int>> faces;
        if (classifierOption == "haar"){
            faces = encloseFaceHaar(&imageClone, &classifierHaar);
        }else if (classifierOption == "hog"){
            faces = encloseFaceHog(&imageClone, &hogFaceDetector);
        }else if (classifierOption == "mmod"){
            faces = encloseFaceMMOD(&imageClone, &mmodFaceDetector);
        }else if (classifierOption == "yolo"){
            faces = encloseFaceYOLO(&imageClone, &net, outLayers);
        }else if (classifierOption == "caffe"){
            faces = encloseFaceDNN(&imageClone, &net, classifierOption);
        }else if (classifierOption == "tensor"){
            faces = encloseFaceDNN(&imageClone, &net, classifierOption);
        }

        for (int i = 0; i < faces.size(); i++)
        {
            std::vector<std::tuple<cv::Mat, cv::Mat>> returned_faces; 
            std::vector<cv::Mat> augmented_images;
            augmented_images.clear();
            cv::Mat img_higher_contrast;
            imageClone.convertTo(img_higher_contrast, -1, 2, 0);

            cv::Mat img_lower_contrast;
            imageClone.convertTo(img_lower_contrast, -1, 0.5, 0);

            cv::Mat img_higher_brightness;
            imageClone.convertTo(img_higher_brightness, -1, 1, 50);

            cv::Mat img_lower_brightness;
            imageClone.convertTo(img_lower_brightness, -1, 1, -50);

            // cv::Mat img_lower_resolution;
            // cv::resize((*image), img_lower_resolution, cv::Size(image->cols*0.75, image->rows*0.75));
            // cv::resize(img_lower_resolution, img_lower_resolution, cv::Size(image->cols, image->rows));
            augmented_images.push_back(imageClone);
            augmented_images.push_back(img_higher_contrast);
            augmented_images.push_back(img_lower_contrast);
            augmented_images.push_back(img_higher_brightness);
            augmented_images.push_back(img_lower_brightness);
            // augmented_images.push_back(img_lower_resolution);

            cv::Rect opencvRect(get<2>(faces[i]), get<0>(faces[i]), get<3>(faces[i]), get<1>(faces[i]));
            //cout << augmented_images.size() << endl;
            for (int j = 0; j < augmented_images.size(); j++){
                face = augmented_images[j](cv::Range(get<0>(faces[i]), get<1>(faces[i])), cv::Range(get<2>(faces[i]), get<3>(faces[i]))).clone();

                face = prepareFace(&opencvRect, shape_model, &augmented_images[j]);
                cv::flip(face, face_flip, 1);

                if (get<1>((*modelFaces)[f]) == ""){
                    cv::imshow("Face Detection - Loaded Faces", face);
                    cv::waitKey(0);
                    cv::destroyAllWindows();
                }
                returned_faces.push_back(make_tuple(face, face_flip));
            }
            
            if (getInputName){
                name = get<1>((*modelFaces)[f]);
                cout << name << " ADD TO RECOGNIZER\n";
            }else if (get<1>((*modelFaces)[f]) == ""){
                cout << "ENTER NAME -> ";
                cin >> name;
                if (name == "NO" || name == "no"){
                    cout << "NOT ADDING TO RECOGNIZER\n";
                    continue;
                }
                (*modelFaces).at(f) = std::make_tuple(get<0>((*modelFaces)[f]), name);
            }else{
                name = get<1>((*modelFaces)[f]);
            }
            
            auto iter = std::find_if((*labels).begin(), (*labels).end(), [=](std::tuple<int, string> item) {return get<1>(item) == name;});
            
            if (iter != (*labels).end()){
                cout << "FOUND PERSON IN LABELS\n";
                for (int m = 0; m < returned_faces.size(); m++){
                    faces_final.push_back(get<0>(returned_faces[m]));
                    faces_final.push_back(get<1>(returned_faces[m]));
                    labels_temp.push_back(get<0>(*iter));
                    labels_temp.push_back(get<0>(*iter));
                }
            }else{
                cout << "NOT FOUND PERSON IN LABELS\n";
                int lSize = (*labels).size();
                (*labels).push_back(make_tuple(lSize, name));
                for (int m = 0; m < returned_faces.size(); m++){
                    faces_final.push_back(get<0>(returned_faces[m]));
                    faces_final.push_back(get<1>(returned_faces[m]));
                    labels_temp.push_back(lSize);
                    labels_temp.push_back(lSize);
                }    
            }
        }
    }

    if (doEigen){
        (*modelEigen)->train(faces_final, labels_temp);
    }else if ((*labels).size() > 1){
        (*modelFisher)->train(faces_final, labels_temp);
    }

    labels_temp.clear();
    faces_final.clear();

    return (*labels);
}