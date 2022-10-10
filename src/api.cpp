// Writen by Horia-Gabriel Radu - 2021-2022
// for Third Year Project at University of Manchester

#include <iostream>
#include <sys/stat.h>
#include <filesystem>
#include <fstream>

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

#include <image.h>
#include <video.h>
#include <loadFace.h>
#include <functs.h>
#include <helpers.h>
#include <matching.h>


using namespace std;

////////////////////////////////////////////////////////////////////

void print_vector(vector<std::string> vector, string separator = " ")
{
    for (auto elem : vector) {
        cout << elem << separator;
    }
 
    cout << endl;
}

////////////////////////////////////////////////////////////////////

int main(int argc, char *argv[])
{
///////////////////////////////
    bool doRecognition = false;
    std::vector<std::tuple<int, std::string>> labels;
    
    bool doEigen = false;
    bool doFisher = false;
    dlib::shape_predictor shape_model_68;
    dlib::deserialize("models/shape_predictor_68_face_landmarks.dat") >> shape_model_68;
    cv::Ptr<cv::face::LBPHFaceRecognizer> modelLBPH = cv::face::LBPHFaceRecognizer::create();
    cv::Ptr<cv::face::EigenFaceRecognizer> modelEigen = cv::face::EigenFaceRecognizer::create();
    cv::Ptr<cv::face::FisherFaceRecognizer> modelFisher = cv::face::FisherFaceRecognizer::create();

    bool recognizerML = false;
    dlib::shape_predictor shape_model_5;
    dlib::deserialize("models/shape_predictor_5_face_landmarks.dat") >> shape_model_5;
    anet_type netRecognizer;
    dlib::deserialize("models/dlib_face_recognition_resnet_model_v1.dat") >> netRecognizer;
    std::vector<dlib::matrix<float, 0, 1>> face_descriptors_final;

    bool doEmotion = false;
    cv::dnn::Net netEmotion;
    std::vector<string> emotionLabels{"Surprise", "Fear", "Disgust", "Happiness", "Sadness", "Anger", "Neutral"};

////////////////////////////////////////////////////////////////

    if (argc < 4){
        cout << "Wrong input of arguments" << endl;
        return 0;
    }

    std::string argOption = argv[1];
    std::string argName = argv[2];
    std::string argWriteVideo = argv[3];
    std::string argClassifier = argv[4];

    if (!(argOption == "images" || argOption == "videos")){
        cout << argOption << " unavailable option, choose either  images  or  videos"<< endl;
        return 0;
    }

	struct stat buffer;
    std::string path = "input/" + argOption + "/" + argName;
    if(stat(path.c_str(), &buffer) != 0){
        cout << "Did not find directory " << argName << " in " << path << endl;
        return 0;
    }
    else{
        cout << "Found directory " << argName << " in " << path << endl;
    }

    bool writeVideo = false;
    if (argWriteVideo == "True" || argWriteVideo == "true" || argWriteVideo == "1" || argWriteVideo == "yes"){
        writeVideo = true;
    }

    std::string classifierOption;
    std::vector<std::string> classifierList{"haar", "caffe", "tensor", "yolo", "hog", "mmod"};
    if (!(find(begin(classifierList), end(classifierList), argClassifier) != end(classifierList))){
        cout << argClassifier << " method unavailable" << endl;
        cout << "Please select one of the following: ";
        print_vector(classifierList, ", ");
        return 0;
    }

    bool doDisplayRecognition = false;
    if (argc >= 6){
        std::string argRecognizer = argv[5];
        if (argRecognizer == "ml"){
            doRecognition = true;
            recognizerML = true;
            cout << "Will do ML face recognition" << endl;
        }else if (argRecognizer == "lbph"){
            doRecognition = true;
            cout << "Will do LBPH face recognition" << endl;
        }else if (argRecognizer == "eigen"){
            doRecognition = true;
            doEigen = true;
            cout << "Will do EIGEN face recognition" << endl;
        }else if (argRecognizer == "fisher"){
            doRecognition = true;
            doFisher = true;
            cout << "Will do FISHER face recognition" << endl;
        }if (argRecognizer == "ml-display"){
            doRecognition = true;
            recognizerML = true;
            doDisplayRecognition = true;
            cout << "Will do ML face recognition with display" << endl;
        }else if (argRecognizer == "lbph-display"){
            doRecognition = true;
            doDisplayRecognition = true;
            cout << "Will do LBPH face recognition with display" << endl;
        }else if (argRecognizer == "eigen-display"){
            doRecognition = true;
            doEigen = true;
            doDisplayRecognition = true;
            cout << "Will do EIGEN face recognition with display" << endl;
        }else if (argRecognizer == "fisher-display"){
            doRecognition = true;
            doFisher = true;
            doDisplayRecognition = true;
            cout << "Will do FISHER face recognition with display" << endl;
        }
    }

    bool doDisplayEmotion = false;
    if (argc >= 7){
        std::string argEmotions = argv[6];
        if (argEmotions == "True" || argEmotions == "true" || argEmotions == "1" || argEmotions == "yes" || argEmotions == "emotion"){
            doEmotion = true;
            cout << "Will do emotion detection" << endl;
        }else if (argEmotions == "True-display" || argEmotions == "true-display" || argEmotions == "1-display" || argEmotions == "yes-display" || argEmotions == "emotion-display"){
            doEmotion = true;
            doDisplayEmotion = true;
            cout << "Will do emotion detection and display" << endl;
        }
    }

    bool outputRecognizedFaces = false;
    if (argc >= 8){
        std::string argRecognizedFaces = argv[7];
        if (argRecognizedFaces == "True" || argRecognizedFaces == "true" || argRecognizedFaces == "1" || argRecognizedFaces == "yes"){
            outputRecognizedFaces = true;
            cout << "Will output recognized faces separately" << endl;
        }
    }

//////////////////////////////////////////////////////////////////////////////////

    cv::CascadeClassifier classifierHaar;
    cv::dnn::Net netYOLO;
    cv::dnn::Net netCaffe;
    cv::dnn::Net netTensor;
    dlib::frontal_face_detector hogFaceDetector;
    net_type mmodFaceDetector;
    std::vector<cv::String> outLayers;
    if (argClassifier == "haar")
    {
        classifierHaar.load("models/haarcascade_frontalface_default.xml");
    }
    else if (argClassifier == "caffe")
    {
        const std::string caffeConfigFile = "models/deploy.prototxt";
        const std::string caffeWeightFile = "models/res10_300x300_ssd_iter_140000_fp16.caffemodel";
        netCaffe = cv::dnn::readNetFromCaffe(caffeConfigFile, caffeWeightFile);
        netCaffe.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
        netCaffe.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
    }
    else if (argClassifier == "tensor")
    {
        const std::string tensorflowConfigFile = "models/opencv_face_detector.pbtxt";
        const std::string tensorflowWeightFile = "models/opencv_face_detector_uint8.pb";
        netTensor = cv::dnn::readNetFromTensorflow(tensorflowWeightFile, tensorflowConfigFile);
        netTensor.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
        netTensor.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
    }
    else if (argClassifier == "hog")
    {
        hogFaceDetector = dlib::get_frontal_face_detector();
    }
    else if (argClassifier == "mmod")
    {
        cv::String mmodModel = "models/mmod_human_face_detector.dat";
        dlib::deserialize(mmodModel) >> mmodFaceDetector;
    }else if (argClassifier == "yolo")
    {
        const std::string yoloFaceWeightFile = "models/yolov3-wider_16000.weights";
        const std::string yoloFaceConfigFile = "models/yolov3-face.cfg";
        netYOLO = cv::dnn::readNetFromDarknet(yoloFaceConfigFile, yoloFaceWeightFile);
        netYOLO.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
        netYOLO.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
        std::vector<cv::String> layerNames = netYOLO.getLayerNames();
        cv::dnn::MatShape unconnectedOutLayers = netYOLO.getUnconnectedOutLayers();
        for (int i = 0; i < unconnectedOutLayers.size(); i++)
        {
            outLayers.push_back(layerNames[unconnectedOutLayers[0] - 1]);
            outLayers.push_back(layerNames[unconnectedOutLayers[1] - 1]);
            outLayers.push_back(layerNames[unconnectedOutLayers[2] - 1]);
        }
    }else{
        cout << "Could not assing a model, exiting..." << endl;
        return 0;
    }

/////////////////////////////////////////////////////////////////////////////////
    classifierOption = argClassifier;
    std::string classifierName;

    std::vector<std::string> fileNames;
    for (const auto & entry : std::filesystem::directory_iterator(path)){
        fileNames.push_back(entry.path());
    }

    if (doRecognition){
        std::vector<std::tuple<std::string, std::string>> fileRecognize;
        for (const auto & entry : std::filesystem::directory_iterator("input/recognising")){
            std::filesystem::path entryPath(entry.path());
            for (const auto & entrySub : std::filesystem::directory_iterator(entry.path())){
                fileRecognize.push_back(std::make_tuple(entrySub.path(), entryPath.stem()));
            }
        }
        if (recognizerML){
            cout << "Enter names for ML recognizer" << endl;
            classifierName = "ml";
            std::vector<dlib::matrix<float, 0, 1>> face_descriptors;
            cv::Mat image;
            std::vector<std::tuple<int, int, int, int>> recFaces;
            for (int i = 0; i < fileRecognize.size(); i++){
                image = cv::imread(get<0>(fileRecognize[i]), cv::IMREAD_COLOR);
                if (image.data){
                    float imageWidth = image.cols;
                    float imageHeight = image.rows;
                    if (imageWidth < 300 && imageHeight < 300){
                        resizeKeepAspectRatio(&image, cv::Size(300, 300), cv::Scalar(0));
                    }else if(imageWidth > imageHeight){
                        resizeKeepAspectRatio(&image, cv::Size(int(imageWidth), int(imageWidth)), cv::Scalar(0));
                    }else{
                        resizeKeepAspectRatio(&image, cv::Size(int(imageHeight), int(imageHeight)), cv::Scalar(0));
                    }

                    if (classifierOption == "haar"){
                        recFaces = encloseFaceHaar(&image, &classifierHaar);
                    }else if (classifierOption == "hog"){
                        recFaces = encloseFaceHog(&image, &hogFaceDetector);
                    }else if (classifierOption == "mmod"){
                        recFaces = encloseFaceMMOD(&image, &mmodFaceDetector);
                    }else if (classifierOption == "yolo"){
                        recFaces = encloseFaceYOLO(&image, &netYOLO, outLayers);
                    }else if (classifierOption == "caffe"){
                        recFaces = encloseFaceDNN(&image, &netCaffe, classifierOption);
                    }else if (classifierOption == "tensor"){
                        recFaces = encloseFaceDNN(&image, &netTensor, classifierOption);
                    }

                    face_descriptors = loadFaceML(&image, &recFaces, &netRecognizer, &labels, &shape_model_5, get<1>(fileRecognize[i]), true);
                    for (int j = 0; j < face_descriptors.size(); j++){
                        face_descriptors_final.push_back(face_descriptors[j]);
                    }
                }
            }
        }else if (doEigen || doFisher){
            if (doEigen){
                cout << "Enter names for EIGEN recognizer";
                classifierName = "eigen";
            }else{
                cout << "Enter names for FISHER recognizer";
                classifierName = "fisher";
            }
            std::vector<std::tuple<std::string, std::string>> recognizerFaces;
            cv::Mat image;
            for (int i = 0; i < fileRecognize.size(); i++){
                image = cv::imread(get<0>(fileRecognize[i]), cv::IMREAD_COLOR);
                if (image.data){
                    recognizerFaces.push_back(std::make_tuple(get<0>(fileRecognize[i]), get<1>(fileRecognize[i])));
                }
            }
            labels = loadFace(classifierOption, &modelEigen, &modelFisher, &labels, &shape_model_68, &recognizerFaces, doEigen, doFisher, true, true);
        }else{
            cout << "Enter names for LBPH recognizer";
            classifierName = "lbph";
            cv::Mat image;
            std::vector<std::tuple<int, int, int, int>> recFaces;
            for (int i = 0; i < fileRecognize.size(); i++){
                image = cv::imread(get<0>(fileRecognize[i]), cv::IMREAD_COLOR);
                if (image.data){
                    float imageWidth = image.cols;
                    float imageHeight = image.rows;
                    if (imageWidth < 300 && imageHeight < 300){
                        resizeKeepAspectRatio(&image, cv::Size(300, 300), cv::Scalar(0));
                    }else if(imageWidth > imageHeight){
                        resizeKeepAspectRatio(&image, cv::Size(int(imageWidth), int(imageWidth)), cv::Scalar(0));
                    }else{
                        resizeKeepAspectRatio(&image, cv::Size(int(imageHeight), int(imageHeight)), cv::Scalar(0));
                    }

                    if (classifierOption == "haar"){
                        recFaces = encloseFaceHaar(&image, &classifierHaar);
                    }else if (classifierOption == "hog"){
                        recFaces = encloseFaceHog(&image, &hogFaceDetector);
                    }else if (classifierOption == "mmod"){
                        recFaces = encloseFaceMMOD(&image, &mmodFaceDetector);
                    }else if (classifierOption == "yolo"){
                        recFaces = encloseFaceYOLO(&image, &netYOLO, outLayers);
                    }else if (classifierOption == "caffe"){
                        recFaces = encloseFaceDNN(&image, &netCaffe, classifierOption);
                    }else if (classifierOption == "tensor"){
                        recFaces = encloseFaceDNN(&image, &netTensor, classifierOption);
                    }
                                       
                    labels = loadFaceLBPH(&image, &recFaces, &modelLBPH, &labels, &shape_model_68, get<1>(fileRecognize[i]), true);
                }
            }
        }
        if (doFisher){
            if (!(labels.size() > 1)){
                cout << "Will not do FISHER recognition on one recognised face" << endl;
                doRecognition = false;
            }
        }
        else if (!(labels.size() > 0 || face_descriptors_final.size() > 0)){
            doRecognition = false;
        }
    }

    if (doEmotion){
        const std::string resnetWeights = "/home/horia/Documents/third-year-project/code/emotionDetector/frozen_models/frozen_graph_resnet.pb";
        const std::string resnetConfig = "/home/horia/Documents/third-year-project/code/emotionDetector/frozen_models/frozen_graph_resnet.pbtxt";
        netEmotion = cv::dnn::readNetFromTensorflow(resnetWeights);
        netEmotion.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
        netEmotion.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
    }
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    std::vector<std::tuple<std::string, int, int, int, int, int, int, std::string, std::string, float, std::string, float>> outputList;
    cv::Mat image;
    std::vector<std::tuple<int, int, int ,int>> faces;
    std::vector<std::tuple<int, float>> emotions;
    std::vector<std::tuple<int, std::string, double, float>> names;
    ////////////////////////////////////// IMAGES /////////////////////////////////////
    if (argOption == "images"){
        cout << "Entering files..." << endl;
        for (int f = 0; f < fileNames.size(); f++)
        {
            faces.clear();
            names.clear();
            emotions.clear();
            image = cv::imread(fileNames[f], cv::IMREAD_COLOR);
            if (image.data)
            {
                float imageWidth = image.cols;
                float imageHeight = image.rows;
                int paddingWidth;
                int paddingHeight;
                cv::Mat imageClone = image.clone();
                if (imageWidth < 300 && imageHeight < 300){
                    paddingWidth = (300 - imageWidth)/2;
                    paddingHeight = (300 - imageHeight)/2;
                    resizeKeepAspectRatio(&imageClone, cv::Size(300, 300), cv::Scalar(0));
                }else if(imageWidth > imageHeight){
                    paddingWidth = 0;
                    paddingHeight = (imageWidth - imageHeight)/2;
                    resizeKeepAspectRatio(&imageClone, cv::Size(int(imageWidth), int(imageWidth)), cv::Scalar(0));
                }else{
                    paddingWidth = (imageHeight - imageWidth)/2;
                    paddingHeight = 0;
                    resizeKeepAspectRatio(&imageClone, cv::Size(int(imageHeight), int(imageHeight)), cv::Scalar(0));
                }

                if (classifierOption == "haar"){
                    faces = encloseFaceHaar(&imageClone, &classifierHaar);
                }else if (classifierOption == "hog"){
                    faces = encloseFaceHog(&imageClone, &hogFaceDetector);
                }else if (classifierOption == "mmod"){
                    faces = encloseFaceMMOD(&imageClone, &mmodFaceDetector);
                }else if (classifierOption == "yolo"){
                    faces = encloseFaceYOLO(&imageClone, &netYOLO, outLayers);
                }else if (classifierOption == "caffe"){
                    faces = encloseFaceDNN(&imageClone, &netCaffe, classifierOption);
                }else if (classifierOption == "tensor"){
                    faces = encloseFaceDNN(&imageClone, &netTensor, classifierOption);
                }
                for (int i = 0; i < faces.size(); i++)
                {
                    if (doRecognition){
                        if (recognizerML){
                            recogniseFacesML(i, &imageClone, &faces[i], &names, &labels, &face_descriptors_final, &netRecognizer, &shape_model_5);
                        }else if (doEigen){
                            recogniseFacesEigen(i, &imageClone, &faces[i], &names, &modelEigen, &labels, &shape_model_68);
                        }else if (doFisher){
                            recogniseFacesFisher(i, &imageClone, &faces[i], &names, &modelFisher, &labels, &shape_model_68);
                        }else{
                            recogniseFaces(i, &imageClone, &faces[i], &names, &modelLBPH, &labels, &shape_model_68);
                        }
                    }
                    if (doEmotion){
                        predictEmotion(&imageClone, &faces[i], &emotions, &netEmotion);
                    }

                    int x1 = get<2>(faces[i]) - paddingWidth;
                    int y1 = get<0>(faces[i]) - paddingHeight;
                    int x2 = get<3>(faces[i]) - paddingWidth;
                    int y2 = get<1>(faces[i]) - paddingHeight;
                    
                    if (x1<0){x1=0;}
                    else if (x1>=imageWidth){x1=imageWidth;}
                    if (y1<0){y1=0;}
                    else if (y1>=imageHeight){y1=imageHeight;}
                    if (x2<0){x2=0;}
                    else if (x2>=imageWidth){x2=imageWidth;}
                    if (y2<0){y2=0;}
                    else if (y2>=imageHeight){y2=imageHeight;}
                    
                    if (doEmotion && doRecognition){
                        outputList.push_back(std::make_tuple(fileNames[f], 0, i, x1, y1, x2, y2, classifierOption, get<1>(names[i]), get<3>(names[i]), emotionLabels[get<0>(emotions[i])], get<1>(emotions[i]) ));
                    }else if (doRecognition){
                        outputList.push_back(std::make_tuple(fileNames[f], 0, i, x1, y1, x2, y2, classifierOption, get<1>(names[i]), get<3>(names[i]), "n/a", -1 ));
                    }else if (doEmotion){
                        outputList.push_back(std::make_tuple(fileNames[f], 0, i, x1, y1, x2, y2, classifierOption, "n/a", -1, emotionLabels[get<0>(emotions[i])], get<1>(emotions[i]) ));
                    }else{
                        outputList.push_back(std::make_tuple(fileNames[f], 0, i, x1, y1, x2, y2, classifierOption, "n/a", -1, "n/a", -1));
                    }
                }
            }
            cout << f+1 << "/" << fileNames.size() << endl;
        }
    }
    ////////////////////////////////////// VIDEOS /////////////////////////////////////
    else if (argOption == "videos"){
        cv::VideoCapture video;
        cout << "Entering files..." << endl;
        for (int f = 0; f < fileNames.size(); f++)
        {
            video.open(string(fileNames[f]));
            if (video.isOpened())
            {
                int frameCounter = 0;
                while (1)
                {
                    faces.clear();
                    names.clear();
                    emotions.clear();
                    video >> image;
                    if (image.empty())
                    {
                        cout << "Finished " << fileNames[f] << endl;
                        break;
                    }
                    float imageWidth = image.cols;
                    float imageHeight = image.rows;
                    int paddingWidth;
                    int paddingHeight;
                    cv::Mat imageClone = image.clone();
                    if (imageWidth < 300 && imageHeight < 300){
                        paddingWidth = (300 - imageWidth)/2;
                        paddingHeight = (300 - imageHeight)/2;
                        resizeKeepAspectRatio(&imageClone, cv::Size(300, 300), cv::Scalar(0));
                    }else if(imageWidth > imageHeight){
                        paddingWidth = 0;
                        paddingHeight = (imageWidth - imageHeight)/2;
                        resizeKeepAspectRatio(&imageClone, cv::Size(int(imageWidth), int(imageWidth)), cv::Scalar(0));
                    }else{
                        paddingWidth = (imageHeight - imageWidth)/2;
                        paddingHeight = 0;
                        resizeKeepAspectRatio(&imageClone, cv::Size(int(imageHeight), int(imageHeight)), cv::Scalar(0));
                    }

                    if (classifierOption == "haar"){
                        faces = encloseFaceHaar(&imageClone, &classifierHaar);
                    }else if (classifierOption == "hog"){
                        faces = encloseFaceHog(&imageClone, &hogFaceDetector);
                    }else if (classifierOption == "mmod"){
                        faces = encloseFaceMMOD(&imageClone, &mmodFaceDetector);
                    }else if (classifierOption == "yolo"){
                        faces = encloseFaceYOLO(&imageClone, &netYOLO, outLayers);
                    }else if (classifierOption == "caffe"){
                        faces = encloseFaceDNN(&imageClone, &netCaffe, classifierOption);
                    }else if (classifierOption == "tensor"){
                        faces = encloseFaceDNN(&imageClone, &netTensor, classifierOption);
                    }

                    for (int i = 0; i < faces.size(); i++)
                    {
                        if (doRecognition){
                            if (recognizerML){
                                recogniseFacesML(i, &imageClone, &faces[i], &names, &labels, &face_descriptors_final, &netRecognizer, &shape_model_5);
                            }else if (doEigen){
                                recogniseFacesEigen(i, &imageClone, &faces[i], &names, &modelEigen, &labels, &shape_model_68);
                            }else if (doFisher){
                                recogniseFacesFisher(i, &imageClone, &faces[i], &names, &modelFisher, &labels, &shape_model_68);
                            }else{
                                recogniseFaces(i, &imageClone, &faces[i], &names, &modelLBPH, &labels, &shape_model_68);
                            }
                        }
                        if (doEmotion){
                            predictEmotion(&imageClone, &faces[i], &emotions, &netEmotion);
                        }

                        int x1 = get<2>(faces[i]) - paddingWidth;
                        int y1 = get<0>(faces[i]) - paddingHeight;
                        int x2 = get<3>(faces[i]) - paddingWidth;
                        int y2 = get<1>(faces[i]) - paddingHeight;
                        
                        if (x1<0){x1=0;}
                        else if (x1>=imageWidth){x1=imageWidth;}
                        if (y1<0){y1=0;}
                        else if (y1>=imageHeight){y1=imageHeight;}
                        if (x2<0){x2=0;}
                        else if (x2>=imageWidth){x2=imageWidth;}
                        if (y2<0){y2=0;}
                        else if (y2>=imageHeight){y2=imageHeight;}
                        
                        if (doEmotion && doRecognition){
                            outputList.push_back(std::make_tuple(fileNames[f], frameCounter, i, x1, y1, x2, y2, classifierOption, get<1>(names[i]), get<3>(names[i]), emotionLabels[get<0>(emotions[i])], get<1>(emotions[i]) ));
                        }else if (doRecognition){
                            outputList.push_back(std::make_tuple(fileNames[f], frameCounter, i, x1, y1, x2, y2, classifierOption, get<1>(names[i]), get<3>(names[i]), "n/a", -1));
                        }else if (doEmotion){
                            outputList.push_back(std::make_tuple(fileNames[f], frameCounter, i, x1, y1, x2, y2, classifierOption, "n/a", -1, emotionLabels[get<0>(emotions[i])], get<1>(emotions[i]) ));
                        }else{
                            outputList.push_back(std::make_tuple(fileNames[f], frameCounter, i, x1, y1, x2, y2, classifierOption, "n/a", -1, "n/a", -1));
                        }
                    }
                    frameCounter++;
                }
            }
            cout << f+1 << "/" << fileNames.size() << endl;
        }
        cv::destroyAllWindows();
        video.release();
    }
    std::string createDirect = "output/" + argOption + "/" + argName + "/" + classifierOption;
    std::filesystem::create_directories(createDirect);
    std::fstream csvFile;
    std::string csvPath;
    if (doRecognition){
        csvPath = createDirect + "/results_" + classifierName + ".csv";
    }else{
        csvPath = createDirect + "/results.csv";
    }
    csvFile.open(csvPath, fstream::in | fstream::out | fstream::trunc);
    cout << "Wrote results to: " << csvPath << endl;
    for (int i = 0; i < outputList.size(); i++){
        csvFile << get<0>(outputList[i]) << "," << get<1>(outputList[i]) << "," << get<2>(outputList[i]) << "," << get<3>(outputList[i]) << "," << get<4>(outputList[i]) << "," << get<5>(outputList[i]) << "," << get<6>(outputList[i]) << "," << get<7>(outputList[i]) << "," << get<8>(outputList[i]) << "," << get<9>(outputList[i]) << "," << get<10>(outputList[i]) << "," << get<11>(outputList[i]) << "\n";
    }
    csvFile.close();

    if (writeVideo){
        cout << "Creating a video with the results..." << endl;
        std::string creatDirectWrite = "output/created_videos/" + argOption + "/" + argName + "/" + classifierOption;
        std::filesystem::create_directories(creatDirectWrite);
        std::string writePath;
        if (doRecognition){
            writePath = creatDirectWrite + "/output_" + classifierName + ".mp4";
        }else{
            writePath = creatDirectWrite + "/output.mp4";
        }
        cv::Mat imageWrite;
        std::vector<std::string> imagesToWrite;
        std::vector<std::vector<std::tuple<int,int,int,int,std::string,std::string>>*> facesWrite;
        std::unordered_map<std::string, std::vector<std::tuple<int,int,int,int,std::string,std::string>>*> hashmap;
        if (argOption == "images"){
            cv::Size videoSize(720,720);
            cv::VideoWriter videoWrite(writePath, cv::VideoWriter::fourcc('m','p','4','v'), 1, videoSize);
            for (int i = 0; i < outputList.size(); i++)
            {
                std::string keyName = get<0>(outputList[i]);
                int x1 = get<3>(outputList[i]);
                int y1 = get<4>(outputList[i]);
                int x2 = get<5>(outputList[i]);
                int y2 = get<6>(outputList[i]);
                std::string name = get<8>(outputList[i]);
                std::string emotion = get<10>(outputList[i]);
                if (!(find(begin(imagesToWrite), end(imagesToWrite), get<0>(outputList[i])) != end(imagesToWrite))){
                    imagesToWrite.push_back(keyName);
                    std::vector<std::tuple<int,int,int,int,std::string,std::string>>* temp = new std::vector<std::tuple<int,int,int,int,std::string,std::string>>();
                    temp->push_back(std::make_tuple(x1,y1,x2,y2,name,emotion));
                    facesWrite.push_back(temp);
                    hashmap[keyName] = facesWrite.back();
                }else{
                    hashmap.at(keyName)->push_back(std::make_tuple(x1,y1,x2,y2,name,emotion));
                }
            }
            for (const auto& [k, v] : hashmap){
                imageWrite = cv::imread(k, cv::IMREAD_COLOR);
                
                float imageWidth = imageWrite.cols;
                float imageHeight = imageWrite.rows;

                resizeKeepAspectRatio(&imageWrite, videoSize, cv::Scalar(0));
                cv::resize(imageWrite, imageWrite, videoSize);

                float ratio = float(videoSize.width / std::max(imageWidth, imageHeight));
                int paddingWidth;
                int paddingHeight;
                if (imageWidth > imageHeight){
                    paddingWidth = 0;
                    paddingHeight = (videoSize.height - imageHeight*ratio)/2;
                }else{
                    paddingWidth = (videoSize.width - imageWidth*ratio)/2;
                    paddingHeight = 0;
                }
                
                for(int i = 0; i < hashmap.at(k)->size(); i++){
                    int x1 = get<0>((*(hashmap.at(k)))[i]);
                    int y1 = get<1>((*(hashmap.at(k)))[i]);
                    int x2 = get<2>((*(hashmap.at(k)))[i]);
                    int y2 = get<3>((*(hashmap.at(k)))[i]);
                    std::string name = get<4>((*(hashmap.at(k)))[i]);
                    std::string emotion = get<5>((*(hashmap.at(k)))[i]);

                    x1 = x1 * ratio;
                    y1 = y1 * ratio;
                    x2 = x2 * ratio;
                    y2 = y2 * ratio;

                    x1 = x1 + paddingWidth;
                    y1 = y1 + paddingHeight;
                    x2 = x2 + paddingWidth;
                    y2 = y2 + paddingHeight;

                    cv::rectangle(imageWrite, cv::Rect(cv::Point(x1,y1),cv::Point(x2,y2)), cv::Scalar(0,255,0), 2);

                    if (doDisplayRecognition){
                        if (name != "n/a"){
                            cv::putText(imageWrite, name, cv::Point(x1+5, y1+15), cv::FONT_HERSHEY_COMPLEX, 0.6, cv::Scalar(0,0,255), 1.9);
                        }
                    }
                    if (doDisplayEmotion){
                        cv::putText(imageWrite, emotion, cv::Point(x1+10, y2-15), cv::FONT_HERSHEY_COMPLEX, 0.6, cv::Scalar(0,0,255), 1.9);
                    }
                }
                videoWrite.write(imageWrite);
            }
            videoWrite.release();
        }else if (argOption == "videos"){
            cv::Size videoSize(720,720);
            cv::VideoWriter videoWrite(writePath, cv::VideoWriter::fourcc('m','p','4','v'), 30, videoSize);
            int frameCounter;
            for (int i = 0; i < outputList.size(); i++)
            {
                std::string keyName = get<0>(outputList[i]);
                int x1 = get<3>(outputList[i]);
                int y1 = get<4>(outputList[i]);
                int x2 = get<5>(outputList[i]);
                int y2 = get<6>(outputList[i]);
                std::string name = get<8>(outputList[i]);
                std::string emotion = get<10>(outputList[i]);
                if (find(begin(imagesToWrite), end(imagesToWrite), get<0>(outputList[i])) != end(imagesToWrite)){
                    if (get<1>(outputList[i]) == frameCounter){
                        hashmap.at(keyName+","+std::to_string(frameCounter))->push_back(std::make_tuple(x1,y1,x2,y2,name,emotion));
                    }else{
                        std::vector<std::tuple<int,int,int,int,std::string,std::string>>* temp = new std::vector<std::tuple<int,int,int,int,std::string,std::string>>();
                        temp->push_back(std::make_tuple(x1,y1,x2,y2,name,emotion));
                        facesWrite.push_back(temp);
                        frameCounter = get<1>(outputList[i]);
                        hashmap[keyName+","+std::to_string(frameCounter)] = facesWrite.back();
                    }
                }else{
                    imagesToWrite.push_back(keyName);
                    std::vector<std::tuple<int,int,int,int,std::string,std::string>>* temp = new std::vector<std::tuple<int,int,int,int,std::string,std::string>>();
                    temp->push_back(std::make_tuple(x1,y1,x2,y2,name,emotion));
                    facesWrite.push_back(temp);
                    frameCounter = get<1>(outputList[i]);
                    hashmap[keyName+","+std::to_string(frameCounter)] = facesWrite.back();
                }
            }
            cv::VideoCapture video;
            for (int f = 0; f < fileNames.size(); f++){
                frameCounter = 0;
                video.open(fileNames[f]);
                if (video.isOpened())
                {
                    while (1)
                    {
                        video >> imageWrite;
                        if (imageWrite.empty())
                        {
                            cout << "Finished " << fileNames[f] << endl;
                            break;
                        }
                        
                        float imageWidth = imageWrite.cols;
                        float imageHeight = imageWrite.rows;

                        resizeKeepAspectRatio(&imageWrite, videoSize, cv::Scalar(0));
                        cv::resize(imageWrite, imageWrite, videoSize);

                        float ratio = float(videoSize.width / std::max(imageWidth, imageHeight));
                        int paddingWidth;
                        int paddingHeight;
                        if (imageWidth > imageHeight){
                            paddingWidth = 0;
                            paddingHeight = (videoSize.height - imageHeight*ratio)/2;
                        }else{
                            paddingWidth = (videoSize.width - imageWidth*ratio)/2;
                            paddingHeight = 0;
                        }
                        
                        if (hashmap.count(fileNames[f]+","+std::to_string(frameCounter)) > 0){
                            for(int i = 0; i < hashmap.at(fileNames[f]+","+std::to_string(frameCounter))->size(); i++){
                                int x1 = get<0>((*(hashmap.at(fileNames[f]+","+std::to_string(frameCounter))))[i]);
                                int y1 = get<1>((*(hashmap.at(fileNames[f]+","+std::to_string(frameCounter))))[i]);
                                int x2 = get<2>((*(hashmap.at(fileNames[f]+","+std::to_string(frameCounter))))[i]);
                                int y2 = get<3>((*(hashmap.at(fileNames[f]+","+std::to_string(frameCounter))))[i]);
                                std::string name = get<4>((*(hashmap.at(fileNames[f]+","+std::to_string(frameCounter))))[i]);
                                std::string emotion = get<5>((*(hashmap.at(fileNames[f]+","+std::to_string(frameCounter))))[i]);

                                x1 = x1 * ratio;
                                y1 = y1 * ratio;
                                x2 = x2 * ratio;
                                y2 = y2 * ratio;

                                x1 = x1 + paddingWidth;
                                y1 = y1 + paddingHeight;
                                x2 = x2 + paddingWidth;
                                y2 = y2 + paddingHeight;

                                cv::rectangle(imageWrite, cv::Rect(cv::Point(x1,y1),cv::Point(x2,y2)), cv::Scalar(0,255,0), 2);

                                if (doDisplayRecognition){
                                    if (name != "n/a"){
                                        cv::putText(imageWrite, name, cv::Point(x1+5, y1+15), cv::FONT_HERSHEY_COMPLEX, 0.6, cv::Scalar(0,0,255), 1.9);
                                    }
                                }
                                if (doDisplayEmotion){
                                    cv::putText(imageWrite, emotion, cv::Point(x1+10, y2-15), cv::FONT_HERSHEY_COMPLEX, 0.6, cv::Scalar(0,0,255), 1.9);
                                }
                            }
                        }

                        videoWrite.write(imageWrite);
                        frameCounter++;
                    }
                }
            }
            video.release();
            videoWrite.release();
        }
    }

    // if (outputRecognizedFaces){

    // }

    cout << "DONE" << endl;
    return 0;
}