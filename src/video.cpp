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
#include <dlib/dnn.h>
#include <dlib/image_processing.h>
#include <dlib/data_io.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_io.h>
#include <dlib/gui_widgets.h>
#include <dlib/clustering.h>
#include <dlib/string.h>

#include <video.h>
#include <functs.h>
#include <matching.h>
#include <helpers.h>

using namespace std;

void processVideo(bool display, string path, string classifierOption, bool recogniseFaceBool, cv::Ptr<cv::face::LBPHFaceRecognizer> *model, cv::Ptr<cv::face::EigenFaceRecognizer> *modelEigen, cv::Ptr<cv::face::FisherFaceRecognizer> *modelFisher, std::vector<std::tuple<int, string>> *labels, dlib::shape_predictor *shape_model, std::vector<dlib::matrix<float, 0, 1>> *face_descriptors, anet_type *anet, dlib::shape_predictor *sp, bool recognizerML, bool doEmotion, bool doEigen, bool doFisher)
{
    cv::CascadeClassifier classifierHaar;
    cv::dnn::Net net;
    dlib::frontal_face_detector hogFaceDetector;
    net_type mmodFaceDetector;
    std::vector<cv::String> outLayers;
    cv::dnn::Net netEmotion;
    std::vector<string> emotionLabels{"Surprise", "Fear", "Disgust", "Happiness", "Sadness", "Anger", "Neutral"};
    if (classifierOption == "haar")
    {
        classifierHaar.load("models/haarcascade_frontalface_default.xml");
    }
    else if (classifierOption == "caffe")
    {
        const std::string caffeWeightFile = "models/res10_300x300_ssd_iter_140000_fp16.caffemodel";
        const std::string caffeConfigFile = "models/deploy.prototxt";
        net = cv::dnn::readNetFromCaffe(caffeConfigFile, caffeWeightFile);
        net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
        net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
    }
    else if (classifierOption == "tensor")
    {
        const std::string tensorflowWeightFile = "models/opencv_face_detector_uint8.pb";
        const std::string tensorflowConfigFile = "models/opencv_face_detector.pbtxt";
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
    }
    else if (classifierOption == "yolo")
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
        const std::string resnetWeights = "/home/horia/Documents/third-year-project/code/emotionDetector/frozen_models/frozen_graph_resnet.pb";
        const std::string resnetConfig = "/home/horia/Documents/third-year-project/code/emotionDetector/frozen_models/frozen_graph_resnet.pbtxt";
        netEmotion = cv::dnn::readNetFromTensorflow(resnetWeights);
        netEmotion.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
        netEmotion.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
    }    

    cv::VideoCapture video;
    cv::Mat image;
    
    if (path == string("images/videos/0"))
        video.open(0);
    else
    {
        try
        {
            video.open(string(path));
        }
        catch(const std::exception& e)
        {
            cout << "Wrong path to video\n";
        }
    }
    
    cv::Mat imageClone;
    std::vector<std::tuple<int, int, int ,int>> faces;
    std::vector<std::tuple<int, std::string, double, float>> names;
    std::vector<std::tuple<int, float>> emotions;
    std::chrono::duration<double> fps;
    std::chrono::time_point<std::chrono::_V2::system_clock,std::chrono::duration<long int,std::ratio<1,1000000000>>> start;
    std::chrono::time_point<std::chrono::_V2::system_clock,std::chrono::duration<long int,std::ratio<1,1000000000>>> end;
    if (display){
        cv::namedWindow("Face Detection - Video", cv::WINDOW_AUTOSIZE);
    }
    if (video.isOpened())
    {
        if (display){
            cout << "Recieved video, starting detection; Press ESC to exit\n";
        }else{
            cout << "Recieved video " << path << endl;
        }
        while (1)
        {
            if (display){
                start = std::chrono::system_clock::now();
            }
            video >> image;
            if (image.empty())
            {
                cout << "Finished " << path << endl;
                break;
            }
            cv::flip(image, image, 1);
            imageClone = image.clone();
            if (image.cols < 300 && image.rows < 300){
                resizeKeepAspectRatio(&imageClone, cv::Size(300, 300), cv::Scalar(0));
            }else if(image.cols > image.rows){
                resizeKeepAspectRatio(&imageClone, cv::Size(imageClone.cols, imageClone.cols), cv::Scalar(0));
            }else{
                resizeKeepAspectRatio(&imageClone, cv::Size(imageClone.rows, imageClone.rows), cv::Scalar(0));
            }
            // cv::resize(imageClone, imageClone, cv::Size(416, 416));

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

                if(doEmotion){
                    for (int i = 0; i < faces.size(); i++){
                        if ( 0 <= get<2>(faces[i])+10 && get<2>(faces[i])+10 < imageClone.cols && 0 <= get<1>(faces[i])-15 && get<1>(faces[i])-15 < imageClone.rows){
                            cv::putText(imageClone, emotionLabels[get<0>(emotions[i])] + ":" + std::to_string(get<1>(emotions[i])), cv::Point(get<2>(faces[i])+10, get<1>(faces[i])-15), cv::FONT_HERSHEY_COMPLEX, 0.6, cv::Scalar(0,0,255), 1.9);
                        }
                    }
                }

                for (int i = 0; i < faces.size(); i++){
                    cv::rectangle(imageClone, cv::Point(get<2>(faces[i]), get<0>(faces[i])), cv::Point(get<3>(faces[i]), get<1>(faces[i])), cv::Scalar(0,255,0), 2, 4);
                }

                cv::putText(imageClone, to_string((int)(1.0/fps.count())), cv::Point(20, 20), cv::FONT_HERSHEY_COMPLEX, 0.8, cv::Scalar(0,0,255), 2.0);

                
                cv::imshow("Face Detection - Video", imageClone);
            

                char stopkey = cv::waitKey(5);
                if (stopkey == 27)
                {
                    break;
                }
            
                end = std::chrono::system_clock::now();
                fps = end-start;
            }
        }
    }
    else
    {
        cout << "Could not open video or camera\n";
    }
    if (display){
        cv::destroyAllWindows();
    }
    video.release();
    return;
}