// Writen by Horia-Gabriel Radu - 2021-2022
// for Third Year Project at University of Manchester

#include <string>
#include <iostream>

#include <opencv2/highgui.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/face.hpp>

#include <dlib/dnn.h>
#include <dlib/image_processing/frontal_face_detector.h>

#include <image.h>
#include <functs.h>
#include <matching.h>
#include <helpers.h>

using namespace std;

std::vector<std::tuple<int, int, int ,int>> processImage(bool display, cv::Mat *image, string classifierOption, bool recogniseFaceBool, bool recognizerML, bool doEmotion, cv::Ptr<cv::face::LBPHFaceRecognizer> *model,  cv::Ptr<cv::face::EigenFaceRecognizer> *modelEigen, cv::Ptr<cv::face::FisherFaceRecognizer> *modelFisher, std::vector<std::tuple<int, string>> *labels, dlib::shape_predictor *shape_model, std::vector<dlib::matrix<float, 0, 1>> *face_descriptors, anet_type *anet, dlib::shape_predictor *sp, bool doEigen, bool doFisher)
{
    cv::CascadeClassifier classifierHaar;
    cv::dnn::Net net;
    dlib::frontal_face_detector hogFaceDetector;
    net_type mmodFaceDetector;
    std::vector<cv::String> outLayers;
    std::vector<std::tuple<int, int, int ,int>> faces;
    cv::dnn::Net netEmotion;
    std::vector<string> emotionLabels{"Surprise", "Fear", "Disgust", "Happiness", "Sadness", "Anger", "Neutral"};
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

    if(doEmotion){
        // net2 = cv::dnn::readNet("/home/horia/Documents/third-year-project/code/emotionDetector/frozen_models/frozen_resnet.pb");
        const std::string resnetWeights = "/home/horia/Documents/third-year-project/code/emotionDetector/frozen_models/frozen_graph_resnet.pb";
        const std::string resnetConfig = "/home/horia/Documents/third-year-project/code/emotionDetector/frozen_models/frozen_graph_resnet.pbtxt";
        netEmotion = cv::dnn::readNetFromTensorflow(resnetWeights);
        netEmotion.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
        netEmotion.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
    }
    
    cv::Mat imageClone = (*image).clone();
    if ((*image).cols < 300 && (*image).rows < 300){
        resizeKeepAspectRatio(&imageClone, cv::Size(300, 300), cv::Scalar(0));
    }else if((*image).cols > (*image).rows){
        resizeKeepAspectRatio(&imageClone, cv::Size(imageClone.cols, imageClone.cols), cv::Scalar(0));
    }else{
        resizeKeepAspectRatio(&imageClone, cv::Size(imageClone.rows, imageClone.rows), cv::Scalar(0));
    }
    if (classifierOption == "haar")
    {
        faces = encloseFaceHaar(&imageClone, &classifierHaar);
    }
    else if (classifierOption == "hog")
    {
        faces = encloseFaceHog(&imageClone, &hogFaceDetector);
    }
    else if (classifierOption == "mmod")
    {
        faces = encloseFaceMMOD(&imageClone, &mmodFaceDetector);
    }
    else if (classifierOption == "yolo")
    {
        faces = encloseFaceYOLO(&imageClone, &net, outLayers);
    }
    else
    {
        faces = encloseFaceDNN(&imageClone, &net, classifierOption);
    }

    // DETECT EMOTION AND RECOGNISE FACE

    std::vector<std::tuple<int, float>> emotions;
    std::vector<std::tuple<int, std::string, double, float>> names;
    if (doEmotion || recogniseFaceBool){
        emotions.clear();
        names.clear();
        for (int i = 0; i < faces.size(); i++){
            if (doEmotion){
                predictEmotion(&imageClone, &faces[i], &emotions, &netEmotion);
            }
            if (recogniseFaceBool){
                if (recognizerML){
                    recogniseFacesML(i, &imageClone, &faces[i], &names, labels, face_descriptors, anet, sp);
                }else if(doEigen){
                    recogniseFacesEigen(i, &imageClone, &faces[i], &names, modelEigen, labels, shape_model);
                }else if(doFisher){
                    recogniseFacesFisher(i, &imageClone, &faces[i], &names, modelFisher, labels, shape_model);
                }else{
                    recogniseFaces(i, &imageClone, &faces[i], &names, model, labels, shape_model);
                }
            }
        }
    }

    // DRAW AROUND THE FACE
    if (display){
        for (size_t i = 0; i < faces.size(); i++)
        {
            cv::rectangle(imageClone, cv::Point(get<2>(faces[i]), get<0>(faces[i])), cv::Point(get<3>(faces[i]), get<1>(faces[i])), cv::Scalar(0,255,0), 2, 4);
        }

        if(doEmotion){
            for (int i = 0; i < faces.size(); i++){
                if ( 0 <= get<2>(faces[i])+10 && get<2>(faces[i])+10 < imageClone.cols && 0 <= get<1>(faces[i])-15 && get<1>(faces[i])-15 < imageClone.rows){
                    cv::putText(imageClone, emotionLabels[get<0>(emotions[i])] + ":" + std::to_string(get<1>(emotions[i])), cv::Point(get<2>(faces[i])+10, get<1>(faces[i])-15), cv::FONT_HERSHEY_COMPLEX, 0.6, cv::Scalar(0,0,255), 1.9);
                }
            }
        }

        if (recogniseFaceBool){
            for (int i = 0; i < names.size(); i++){
                //if ((recogniseFaceBool == true) && (faces.size() == names.size())){
                    int faceNr = get<0>(names[i]);
                    if (faceNr == -1){
                        continue;
                    }
                    int x1 = get<2>(faces[faceNr]);
                    int y1 = get<0>(faces[faceNr])-10;
                    int x2 = get<2>(faces[faceNr]) + (get<3>(faces[faceNr])-get<2>(faces[faceNr]))/1.4;
                    int y2 = get<0>(faces[faceNr])+18;
                    if (x1 && y1 <= imageClone.cols && 0 <= x1 && y1 <= imageClone.rows){
                        cv::putText(imageClone, get<1>(names[i]), cv::Point(x1, y1), cv::FONT_HERSHEY_COMPLEX, 0.6, cv::Scalar(0,0,255), 1.9);
                    }
                    if (x2 && y2 <= imageClone.cols && 0 <= x2 && y2 <= imageClone.rows){
                        cv::putText(imageClone, to_string(((int)(get<2>(names[i])))), cv::Point(x2, y2), cv::FONT_HERSHEY_COMPLEX, 0.6, cv::Scalar(0,0,255), 1.9);
                    }
                //}
            }
        }
                
        
        cv::imshow("Face Detection - Image", imageClone);
        cv::waitKey(0);
        cv::destroyAllWindows();
    }
    return faces;
}