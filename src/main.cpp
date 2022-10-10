// Writen by Horia-Gabriel Radu - 2021-2022
// for Third Year Project at University of Manchester

#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/imgproc.hpp>
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

#include <image.h>
#include <video.h>
#include <loadFace.h>
#include <helpers.h>

using namespace std;

////////////////////////////////////////////////////////////////////

enum choiceCode {
    imageCode,
    videoCode,
    loadCode,
    emotionCode,
    exitCode,
    defaultCode
};

choiceCode methodHash (string const& inString) {
    if (inString == "image") return imageCode;
    if (inString == "video") return videoCode;
    if (inString == "load") return loadCode;
    if (inString == "emotion") return emotionCode;
    if (inString == "exit") return exitCode;
    return defaultCode;
}

std::vector<std::tuple<int, std::string>> initializeRecognizer(cv::Ptr<cv::face::LBPHFaceRecognizer> model) {
    std::vector<std::tuple<int, std::string>> labels;

    try
    {
        model->read("models/recognizer_model.yml");
        cout << "Opened already made face recognizer\n";
    }
    catch(const std::exception& e)
    {
        std::cerr << e.what() << " | READING MODEL ERROR | \n";
        cout << "No existing recognizer model, will create new\n";
    }

    
    fstream labels_file;
    labels_file.open("models/labels_file.csv", ios::in);

    if (!labels_file.is_open()){
        cout << "No existing labels file, will create new\n";
    }else{
        int number = 0;
        string word, line, temp;
        vector<string> row;
        while(getline(labels_file, line)){
            row.clear();
            stringstream s(line);
            while(std::getline(s, word, ',')){
                row.push_back(word);
            }
            labels.push_back(make_tuple(stoi(row[0]), row[1]));
            number++;
        }
        cout << "Opened already made labels\n";
    }
    cout << "Finished\n";

    labels_file.close();

    return labels;
}

////////////////////////////////////////////////////////////////////

int main( int argc, char** argv )
{
    string option;
    string path;
    bool repeat;
    bool doRecognition;
    bool recognizerML = false;
    bool doEigen = false;
    bool doFisher = false;
    bool doEmotion = false;

    string classifierOption = "caffe";
    vector<string> classifierList{"haar", "caffe", "tensor", "yolo", "hog", "mmod"};

    cv::Ptr<cv::face::LBPHFaceRecognizer> model = cv::face::LBPHFaceRecognizer::create();

    std::vector<std::tuple<int, std::string>> labels;

    dlib::shape_predictor shape_model;
    dlib::deserialize("models/shape_predictor_68_face_landmarks.dat") >> shape_model;

    cv::Ptr<cv::face::EigenFaceRecognizer> modelEigen = cv::face::EigenFaceRecognizer::create();
    cv::Ptr<cv::face::FisherFaceRecognizer> modelFisher = cv::face::FisherFaceRecognizer::create();
    std::vector<std::tuple<std::string, std::string>> modelFaces;

    dlib::shape_predictor sp;
    dlib::deserialize("models/shape_predictor_5_face_landmarks.dat") >> sp;
    anet_type net;
    dlib::deserialize("models/dlib_face_recognition_resnet_model_v1.dat") >> net;

    std::vector<dlib::matrix<float, 0, 1>> face_descriptors_final;

    if (labels.size() > 0)
    {
        doRecognition = true;
    }else
    {
        doRecognition = false;
    }


    do {
        repeat = false;
        cout << "Choose an option (image / load / video / emotion / exit): ";
        cin >> option;
    
        switch (methodHash(option)) {
            {
            case imageCode:
                cout << "--IMAGE--\n";
                cout << "Getting image | CURRENT OPTION: " << classifierOption << endl;
                cout << "AVAILABLE OPTIONS: haar, caffe, tensor, yolo, hog, mmod\n";
                    while (1)
                    {
                        cout << "Enter the image name, 'back' or 'exit': ";
                        cin >> path;
                        if (path == "exit")
                        {
                            break;
                        }
                        else if (path == "back")
                        {
                            repeat = true;
                            break;
                        }
                        else if (find(begin(classifierList), end(classifierList), path) != end(classifierList))
                        {
                            cout << "Changed classifier option from " << classifierOption << " to " << path << endl;
                            classifierOption = path;
                        }
                        else
                        {
                            cv::Mat image;
                            image = cv::imread("images/" + path + ".jpg", cv::IMREAD_COLOR);
                            if (! image.data)
                            {
                                cout << "Could not open the file: " << path << endl;
                                continue;
                            }
                            processImage(true, &image, classifierOption, doRecognition, recognizerML, doEmotion, &model, &modelEigen, &modelFisher, &labels, &shape_model, &face_descriptors_final, &net, &sp, doEigen, doFisher);
                        }
                    }
                break;
            }
            {
            case videoCode:
                cout << "--VIDEO--\n";
                cout << "Getting a video | CURRENT OPTION: " << classifierOption << endl;
                cout << "AVAILABLE OPTIONS: haar, caffe, tensor, yolo, hog, mmod\n";
                while (1)
                {
                    cout << "Enter the video name, '0' for webcam, 'back' or 'exit': ";
                    cin >> path;
                    if (path == "exit")
                    {
                        break;
                    }
                    else if (path == "back")
                    {
                        repeat = true;
                        break;
                    }
                    else if (find(begin(classifierList), end(classifierList), path) != end(classifierList))
                    {
                        cout << "Changed classifier option from " << classifierOption << " to " << path << endl;
                        classifierOption = path;
                    }
                    else
                    {
                        std::string pathVideo = "images/videos/" + string(path);
                        processVideo(true, pathVideo, classifierOption, doRecognition, &model, &modelEigen, &modelFisher, &labels, &shape_model, &face_descriptors_final, &net, &sp, recognizerML, doEmotion, doEigen, doFisher);
                    }
                }
                break;
            }
            {
            case loadCode:
                cout << "--LOAD--\n";
                cout << "Load a face | CURRENT OPTION: " << classifierOption << endl;
                cout << "AVAILABLE OPTIONS: haar, caffe, tensor, yolo, hog, mmod\n";
                if (recognizerML){
                    cout << "Current Mode is - ML\n";
                }else if (doEigen){
                    cout << "Current Mode is - Classifier Eigen\n";
                }else if (doFisher){
                    cout << "Current Mode is - Classifier Fisher\n";
                }else{
                    cout << "Current Mode is - Classifier\n";
                }
                std::vector<dlib::matrix<float, 0, 1>> face_descriptors_vector;
                while (1)
                {
                    cout << "Enter the image name to load, 'load', 'save', 'ml/lbph/eigen/fisher', 'clear', 'back' or 'exit': ";
                    cin >> path;
                    if (path == "exit")
                    {
                        break;
                    }
                    else if (path == "back")
                    {
                        repeat = true;
                        break;
                    }
                    else if (find(begin(classifierList), end(classifierList), path) != end(classifierList))
                    {
                        cout << "Changed classifier option from " << classifierOption << " to " << path << endl;
                        classifierOption = path;
                    }
                    else if (path == "load")
                    {
                        cout << "Loaded saved model\n";
                        labels = initializeRecognizer(model);
                    }
                    else if (path == "save")
                    {
                        cout << "Saved current model\n";
                        model->write("models/recognizer_model.yml");
                        fstream labels_file;
                        labels_file.open("models/labels_file.csv", ios::out);
                        for (int i = 0; i < labels.size(); i++){
                            labels_file << get<0>(labels[i]) << "," << get<1>(labels[i]) << "\n";
                        }
                        labels_file.close();
                    }else if (path == "ml" || path == "lbph" || path == "eigen" || path == "fisher"){
                        cout << "-- CHANGED RECOGNIZER --\n";
                        if (path == "ml"){
                            recognizerML = true;
                            doEigen = false;
                            doFisher = false;
                            cout << "Will do ML\n";
                        }else if (path == "lbph"){
                            recognizerML = false;
                            doEigen = false;
                            doFisher = false;
                            cout << "Will do LBPH Classifier\n";
                        }else if (path == "eigen"){
                            recognizerML = false;
                            doEigen = true;
                            doFisher = false;
                            cout << "Will do Eigen Classifier\n";
                        }else if (path == "fisher"){
                            recognizerML = false;
                            doEigen = false;
                            doFisher = true;
                            cout << "Will do Fisher Classifier\n";
                        }
                        face_descriptors_final.clear();
                        model=cv::face::LBPHFaceRecognizer::create();
                        modelEigen=cv::face::EigenFaceRecognizer::create();
                        modelFisher=cv::face::FisherFaceRecognizer::create();
                        model->clear();
                        model->empty();
                        modelEigen->clear();
                        modelEigen->empty();
                        modelFisher->clear();
                        modelFisher->empty();
                        labels.clear();
                        modelFaces.clear();
                    }else if (path == "clear"){
                        face_descriptors_final.clear();
                        model=cv::face::LBPHFaceRecognizer::create();
                        modelEigen=cv::face::EigenFaceRecognizer::create();
                        modelFisher=cv::face::FisherFaceRecognizer::create();
                        model->clear();
                        model->empty();
                        modelEigen->clear();
                        modelEigen->empty();
                        modelFisher->clear();
                        modelFisher->empty();
                        labels.clear();
                        modelFaces.clear();
                        cout << "Cleared current recognizer model and labels\n";
                        int status = remove("models/recognizer_model.yml");
                        if (status==0)
                            cout<<"Saved model deleted successfully\n";
                        else
                            cout<<"Error when deleating saved model\n";
                        status = remove("models/labels_file.csv");
                        if (status==0)
                            cout<<"Saved labels deleted successfully\n";
                        else
                            cout<<"Error when deleating saved labels\n";
                    }else{
                        std::vector<std::tuple<int, int, int ,int>> faces;
                        if (recognizerML){
                            cv::Mat image;
                            image = cv::imread("images/" + path + ".jpg", cv::IMREAD_COLOR);
                            if (! image.data){
                                cout << "Could not open the file: " << path << endl;
                                continue;
                            }
                            faces = processImage(true, &image, classifierOption, false, false, false, &model, &modelEigen, &modelFisher, &labels, &shape_model, &face_descriptors_final, &net, &sp, false, false);
                            cv::Mat imageClone = image.clone();
                            if (image.cols < 300 && image.rows < 300){
                                resizeKeepAspectRatio(&imageClone, cv::Size(300, 300), cv::Scalar(0));
                            }else if(image.cols > image.rows){
                                resizeKeepAspectRatio(&imageClone, cv::Size(imageClone.cols, imageClone.cols), cv::Scalar(0));
                            }else{
                                resizeKeepAspectRatio(&imageClone, cv::Size(imageClone.rows, imageClone.rows), cv::Scalar(0));
                            }
                            std::vector<dlib::matrix<float, 0, 1>> face_descriptors;
                            face_descriptors = loadFaceML(&imageClone, &faces, &net, &labels, &sp, "", false);
                            for (size_t i = 0; i < face_descriptors.size(); i++){
                                face_descriptors_final.push_back(face_descriptors[i]);
                            }
                        }else if (doEigen || doFisher){
                            modelFaces.push_back(std::make_tuple(path, ""));
                            modelEigen=cv::face::EigenFaceRecognizer::create();
                            modelFisher=cv::face::FisherFaceRecognizer::create();
                            labels = loadFace(classifierOption, &modelEigen, &modelFisher, &labels, &shape_model, &modelFaces, doEigen, doFisher, false, false);
                        }else{
                            cv::Mat image;
                            image = cv::imread("images/" + path + ".jpg", cv::IMREAD_COLOR);
                            if (! image.data){
                                cout << "Could not open the file: " << path << endl;
                                continue;
                            }
                            faces = processImage(true, &image, classifierOption, false, false, false, &model, &modelEigen, &modelFisher, &labels, &shape_model, &face_descriptors_final, &net, &sp, false, false);
                            cv::Mat imageClone = image.clone();
                            if (image.cols < 300 && image.rows < 300){
                                resizeKeepAspectRatio(&imageClone, cv::Size(300, 300), cv::Scalar(0));
                            }else if(image.cols > image.rows){
                                resizeKeepAspectRatio(&imageClone, cv::Size(imageClone.cols, imageClone.cols), cv::Scalar(0));
                            }else{
                                resizeKeepAspectRatio(&imageClone, cv::Size(imageClone.rows, imageClone.rows), cv::Scalar(0));
                            }
                            labels = loadFaceLBPH(&imageClone, &faces, &model, &labels, &shape_model, "", false);
                        }
                    }
                    if (labels.size() > 0 || face_descriptors_final.size() > 0)
                    {
                        doRecognition = true;
                    }else
                    {
                        doRecognition = false;
                    }
                }
                break;
            }
            {
            case emotionCode:
            
                cout << "--EMOTION--\n";
                std::string input;
                cout << "Enter method (ON/on or OFF/off):";
                cin >> input;
                if (input == "ON" || input == "on")
                {
                    doEmotion = true;
                }else
                {
                    doEmotion = false;
                }
                repeat = true;
                break;
            }
            {
            case exitCode:
                cout << "--EXITING--\n";
                return 0;
            }
            {
            default:
                cout << "--WRONG INPUT--\n";
                cout << "Try again\n";
                repeat = true;
                break;
            }
        }
    } while(repeat);
    cout << "Finished\n";
    return 0;
}