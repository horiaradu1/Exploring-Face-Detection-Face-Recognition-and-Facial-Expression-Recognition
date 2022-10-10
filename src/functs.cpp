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
#include <dlib/data_io.h>
#include <dlib/image_processing/frontal_face_detector.h>

#include <functs.h>
#include <helpers.h>

using namespace std;

// DNN consts
const double inScaleFactor = 1.0;
const cv::Scalar meanVal(104.0, 177.0, 123.0);
const float confidenceThreshold = 0.7;

int depth = 1;
int cropPad = 30;

std::vector<std::tuple<int, int, int ,int>> encloseFaceDNN(cv::Mat *image, cv::dnn::Net *net, string option){
    std::vector<std::tuple<int, int, int ,int>> cropped_faces;    
    // RATIO AND RESOLUTION PROBLEMs FOR IMAGES

    int imageWidth = (*image).cols;
    int imageHeight = (*image).rows;
    float ratioWidth = float(imageWidth / 600.0);
    float ratioHeight = float(imageHeight / 600.0);

    cv::Mat imageClone = (*image).clone();
    resizeKeepAspectRatio(&imageClone, cv::Size(600, 600), cv::Scalar(0));
    
    // Center crop and split crop maybe to get smaller faces (QUAD TREE)
    // Scale invariant input (4^0 + 4^1 + 4^2 + 4^3 + 4^4 + ...)
	
    // std::vector<cv::Mat> inputBlobs;
    cv::Mat inputBlob;
    std::vector<cv::Mat> arrayImgs;
    std::vector<std::tuple<int, int>> cropOffsets;
    std::vector<std::tuple<int, int, int>> cropSizes;

    overlappingCropWithPad(&imageClone, &arrayImgs, &cropOffsets, &cropSizes, &cropPad, &depth);
    // for (int i = 0; i < arrayImgs.size(); i++){
        if (option == "caffe")
        {
            // if (arrayImgs[i].cols == imageClone.cols || arrayImgs[i].rows == imageClone.rows){
            //     inputBlobs.push_back(cv::dnn::blobFromImage(arrayImgs[i], 1.0, cv::Size(300, 300), meanVal, false, false));
            // }else{                
                // inputBlobs.push_back(cv::dnn::blobFromImage(arrayImgs[i], 1.0, cv::Size(arrayImgs[i].cols/2, arrayImgs[i].rows/2), meanVal, false, false));
            // }
            inputBlob = cv::dnn::blobFromImages(arrayImgs, 1.0, cv::Size(300, 300), meanVal, false, false);
        }
        else
        {
            // if (arrayImgs[i].cols < 300 || arrayImgs[i].rows < 300){
            //     inputBlobs.push_back(cv::dnn::blobFromImage(arrayImgs[i], 1.0, cv::Size(300, 300), cv::Scalar(104, 117, 123), true, false));
            // }else{
                // inputBlobs.push_back(cv::dnn::blobFromImage(arrayImgs[i], 1.0, cv::Size(arrayImgs[i].cols/2, arrayImgs[i].rows/2), cv::Scalar(104, 117, 123), true, false));
            // }
            inputBlob = cv::dnn::blobFromImages(arrayImgs, 1.0, cv::Size(300, 300), cv::Scalar(104, 117, 123), true, false);
        }
    // }
    cv::Mat detection;
    // cv::Mat detectionMat;

    // for (int i = 0; i < inputBlobs.size(); i++){
        // (*net).setInput(inputBlobs[i], "data");
        (*net).setInput(inputBlob, "data");
        detection = (*net).forward("detection_out");
        cv::Mat detectionMat(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());
        // cv::Mat detectionMatrix(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());
        // for (int x = 0; x < detectionMatrix.rows; x++)
            // detectionMatrix.at<float>(x, 0) = float(i);
        // detectionMat.push_back(detectionMatrix);
    // }

    std::vector<float> scores;
    std::vector<cv::Rect> boxes;
    for (int i = 0; i < detectionMat.rows; i++)
    {
        float confidence = detectionMat.at<float>(i, 2);
        if (confidence < confidenceThreshold){
            continue;
        }
        if (detectionMat.at<float>(i, 3) < 0.0 || detectionMat.at<float>(i, 4) < 0.0 || detectionMat.at<float>(i, 5) < 0.0 || detectionMat.at<float>(i, 6) < 0.0 || detectionMat.at<float>(i, 3) > 1.0 || detectionMat.at<float>(i, 4) > 1.0 || detectionMat.at<float>(i, 5) > 1.0 || detectionMat.at<float>(i, 6) > 1.0){
            continue;
        }
        int bNr = (int)detectionMat.at<float>(i, 0);
        int x1 = static_cast<int>(detectionMat.at<float>(i, 3) * get<1>(cropSizes[bNr]));
        int y1 = static_cast<int>(detectionMat.at<float>(i, 4) * get<0>(cropSizes[bNr]));
        int x2 = static_cast<int>(detectionMat.at<float>(i, 5) * get<1>(cropSizes[bNr]));
        int y2 = static_cast<int>(detectionMat.at<float>(i, 6) * get<0>(cropSizes[bNr]));
        int boxWidth = x2-x1;
        int boxHeight = y2-y1;
        if (!(0 <= x1 && 0 <= y1 && 0 <= x2 && 0 <= y2 && x1 < imageClone.cols && y1 < imageClone.rows && x2 < imageClone.cols && y2 < imageClone.rows)){
            continue;
        }
        if (bNr == 0){
            scores.push_back(confidence+3);
        }else if (bNr < 5){
            scores.push_back(confidence+2);
        }else if (bNr < 21){
            scores.push_back(confidence+1);
        }else{
            scores.push_back(confidence);
        }
        boxes.push_back(cv::Rect(x1 + get<1>(cropOffsets[bNr]), y1 + get<0>(cropOffsets[bNr]), boxWidth, boxHeight));
        //cropped_faces.push_back(make_tuple(y1 + get<0>(cropOffsets[bNr]), y2 + get<0>(cropOffsets[bNr]), x1 + get<1>(cropOffsets[bNr]), x2 + get<1>(cropOffsets[bNr])));
    }

    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, scores, 0.8, 0.1, indices);

    for (int i=0; i < indices.size(); i++){
        int y1 = boxes[indices[i]].y * ratioHeight;
        int y2 = (y1 + boxes[indices[i]].height * ratioHeight);
        int x1 = boxes[indices[i]].x * ratioWidth;
        int x2 = (x1 + boxes[indices[i]].width * ratioWidth);
        if (0 <= x1 && 0 <= y1 && 0 <= x2 && 0 <= y2 && x1 < (*image).cols && y1 < (*image).rows && x2 < (*image).cols && y2 < (*image).rows){
            // cropped_faces.push_back(make_tuple(boxes[indices[i]].y, (boxes[indices[i]].y + boxes[indices[i]].height), boxes[indices[i]].x, (boxes[indices[i]].x + boxes[indices[i]].width)));
            cropped_faces.push_back(make_tuple(y1, y2, x1, x2));
        // }else if (0 <= boxes[indices[i]].x && 0 <= boxes[indices[i]].y && 0 <= (boxes[indices[i]].x + boxes[indices[i]].width) && 0 <= (boxes[indices[i]].y + boxes[indices[i]].height) && boxes[indices[i]].x < (*image).cols && boxes[indices[i]].y < (*image).rows && (boxes[indices[i]].x + boxes[indices[i]].width) < (*image).cols && (boxes[indices[i]].y + boxes[indices[i]].height) < (*image).rows){
        }
        //else{
        //     cropped_faces.push_back(make_tuple(boxes[indices[i]].y, (boxes[indices[i]].y + boxes[indices[i]].height), boxes[indices[i]].x, (boxes[indices[i]].x + boxes[indices[i]].width)));
        // }
    }
    return cropped_faces;
}


// Haar consts
const float scaleFactor = 1.1;
const float minNeighbor = 3;

std::vector<std::tuple<int, int, int ,int>> encloseFaceHaar(cv::Mat *image, cv::CascadeClassifier *classifier){
    std::vector<std::tuple<int, int, int ,int>> cropped_faces;
    std::vector<cv::Rect> faces;

    int imageHeight = (*image).rows;
    int imageWidth = (*image).cols;
    float ratioHeight = float(imageHeight / 400.0);
    float ratioWidth = float(imageWidth / 400.0);

    cv::Mat imageClone = (*image).clone();
    cv::resize(imageClone, imageClone, cv::Size(400, 400));
    // resizeKeepAspectRatio(&imageClone, cv::Size(600, 600), cv::Scalar(0));

    cv::Mat grey;
    cv::cvtColor(imageClone, grey, cv::COLOR_BGR2GRAY);
    // cv::equalizeHist(grey, grey);
        
    (*classifier).detectMultiScale(grey, faces, scaleFactor, minNeighbor, 0);

    for (size_t i = 0; i < faces.size(); i++)
    {
        int w = faces[i].width;
        int h = faces[i].height;
        int x1 = (faces[i].x) * ratioWidth;
        int y1 = (faces[i].y) * ratioHeight;
        int x2 = (faces[i].x + w) * ratioWidth;
        int y2 = (faces[i].y + h) * ratioHeight;

        if (!(0 <= x1 && 0 <= y1 && 0 <= x2 && 0 <= y2 && x1 < (*image).cols && y1 < (*image).rows && x2 < (*image).cols && y2 < (*image).rows)){
            continue;
        }

        cropped_faces.push_back(make_tuple(y1, y2, x1, x2));

    }
    return cropped_faces;
}


std::vector<std::tuple<int, int, int ,int>> encloseFaceHog(cv::Mat *image, dlib::frontal_face_detector *faceDetector){
    std::vector<std::tuple<int, int, int ,int>> cropped_faces;

    int imageHeight = (*image).rows;
    int imageWidth = (*image).cols;
    float ratioHeight = float(imageHeight / 400.0);
    float ratioWidth = float(imageWidth / 400.0);

    cv::Mat imageClone = (*image).clone();
    cv::resize(imageClone, imageClone, cv::Size(400, 400));
    // resizeKeepAspectRatio(&imageClone, cv::Size(600, 600), cv::Scalar(0));

    dlib::cv_image<dlib::bgr_pixel> dlibImg(imageClone);

    vector<dlib::rectangle> faceRects = (*faceDetector)(dlibImg);

    for (size_t i = 0; i < faceRects.size(); i++)
    {
        int x1 = faceRects[i].left() * ratioWidth;
        int y1 = faceRects[i].top() * ratioHeight;
        int x2 = faceRects[i].right() * ratioWidth;
        int y2 = faceRects[i].bottom() * ratioHeight;

        if (!(0 <= x1 && 0 <= y1 && 0 <= x2 && 0 <= y2 && x1 < (*image).cols && y1 < (*image).rows && x2 < (*image).cols && y2 < (*image).rows)){
            continue;
        }

        cropped_faces.push_back(make_tuple(y1, y2, x1, x2));
    }
    return cropped_faces;
}


std::vector<std::tuple<int, int, int ,int>> encloseFaceMMOD(cv::Mat *image, net_type *mmodFaceDetector){
    std::vector<std::tuple<int, int, int ,int>> cropped_faces;

    int imageHeight = (*image).rows;
    int imageWidth = (*image).cols;
    float ratioHeight = float(imageHeight / 400.0);
    float ratioWidth = float(imageWidth / 400.0);

    cv::Mat imageClone = (*image).clone();
    cv::resize(imageClone, imageClone, cv::Size(400, 400));
    // resizeKeepAspectRatio(&imageClone, cv::Size(600, 600), cv::Scalar(0));

    dlib::cv_image<dlib::bgr_pixel> dlibImg(imageClone);
    dlib::matrix<dlib::rgb_pixel> dlibMatrix;
    dlib::assign_image(dlibMatrix, dlibImg);

    std::vector<dlib::mmod_rect> faceRects = (*mmodFaceDetector)(dlibMatrix);

    for (size_t i = 0; i < faceRects.size(); i++)
    {
        int x1 = faceRects[i].rect.left() * ratioWidth;
        int y1 = faceRects[i].rect.top() * ratioHeight;
        int x2 = faceRects[i].rect.right() * ratioWidth;
        int y2 = faceRects[i].rect.bottom() * ratioHeight;

        if (!(0 <= x1 && 0 <= y1 && 0 <= x2 && 0 <= y2 && x1 < (*image).cols && y1 < (*image).rows && x2 < (*image).cols && y2 < (*image).rows)){
            continue;
        }
        cropped_faces.push_back(make_tuple(y1, y2, x1, x2));
    }
    
    return cropped_faces;
}

std::vector<std::tuple<int, int, int ,int>> encloseFaceYOLO(cv::Mat *image, cv::dnn::Net *net, std::vector<cv::String> outLayers){
    std::vector<std::tuple<int, int, int ,int>> cropped_faces;
    std::vector<float> scores;
    std::vector<cv::Rect> boxes;

    // cv::Mat imageClone;
    // cv::resize((*image), imageClone, cv::Size(416, 416));
    //imageClone = resizeKeepAspectRatio((*image), cv::Size(416, 416), cv::Scalar(0));
    // cv::Mat inputBlob = cv::dnn::blobFromImage(resizeKeepAspectRatio((*image), cv::Size(416, 416), cv::Scalar(0)), 1/255.0, cv::Size(416, 416), cv::Scalar(0, 0, 0), false, false);
    cv::Mat inputBlob = cv::dnn::blobFromImage((*image), 1.0/255.0, cv::Size(416, 416), cv::Scalar(0, 0, 0), true, false);

    (*net).setInput(inputBlob);
    std::vector<cv::Mat> detection;
    (*net).forward(detection, outLayers);

    for (int j = 0; j < detection.size(); j++){
        for (int i = 0; i < detection[j].rows; i++){            
            if (detection[j].at<float>(i, 5) > 0.5){
                int centerX = detection[j].at<float>(i, 0) * (*image).cols;
                int centerY = detection[j].at<float>(i, 1) * (*image).rows;
                int detectionWidth = detection[j].at<float>(i, 2) * (*image).cols;
                int detectionHeight = detection[j].at<float>(i, 3) * (*image).rows;

                if (0 <= centerX-detectionWidth/2 && 0 <= centerY-detectionHeight/2 && 0 <= (centerX-detectionWidth/2 + detectionWidth) && 0 <= (centerY-detectionHeight/2 + detectionHeight) && centerX-detectionWidth/2 < (*image).cols && centerY-detectionHeight/2 < (*image).rows && (centerX-detectionWidth/2 + detectionWidth) < (*image).cols && (centerY-detectionHeight/2 + detectionHeight) < (*image).rows){
                    boxes.push_back(cv::Rect(centerX-detectionWidth/2, centerY-detectionHeight/2, detectionWidth, detectionHeight));
                    scores.push_back((float)detection[j].at<float>(i, 5));
                }
                // cropped_faces.push_back(make_tuple(centerY - detectionHeight/2, centerY + detectionHeight/2, centerX-detectionWidth/2, centerX+detectionWidth/2));
            }
        }
    }
    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, scores, 0.5, 0.25, indices);

    for (int i=0; i < indices.size(); i++){
        int y1 = boxes[indices[i]].y - boxes[indices[i]].height * 0.05;
        int y2 = (boxes[indices[i]].y + boxes[indices[i]].height) +  boxes[indices[i]].height * 0.05;
        int x1 = boxes[indices[i]].x - boxes[indices[i]].width * 0.025;
        int x2 = (boxes[indices[i]].x + boxes[indices[i]].width) +  boxes[indices[i]].width * 0.025;
        if (0 <= x1 && 0 <= y1 && 0 <= x2 && 0 <= y2 && x1 < (*image).cols && y1 < (*image).rows && x2 < (*image).cols && y2 < (*image).rows){
            cropped_faces.push_back(make_tuple(y1, y2, x1, x2));
        // }else if (0 <= boxes[indices[i]].x && 0 <= boxes[indices[i]].y && 0 <= (boxes[indices[i]].x + boxes[indices[i]].width) && 0 <= (boxes[indices[i]].y + boxes[indices[i]].height) && boxes[indices[i]].x < (*image).cols && boxes[indices[i]].y < (*image).rows && (boxes[indices[i]].x + boxes[indices[i]].width) < (*image).cols && (boxes[indices[i]].y + boxes[indices[i]].height) < (*image).rows){
        }else{
            cropped_faces.push_back(make_tuple(boxes[indices[i]].y, (boxes[indices[i]].y + boxes[indices[i]].height), boxes[indices[i]].x, (boxes[indices[i]].x + boxes[indices[i]].width)));
        }   
    }
    return cropped_faces;
}

void predictEmotion(cv::Mat *image, std::tuple<int, int, int ,int> *face, std::vector<std::tuple<int, float>> *emotions, cv::dnn::Net *netEmotion){
    cv::Mat emotion;
    if (true){
        // Normal resize losing original aspect ratio
        emotion = (*image)(cv::Range(get<0>(*face), get<1>(*face)), cv::Range(get<2>(*face), get<3>(*face)));
        cv::resize(emotion, emotion, cv::Size(100, 100));
    }else if (false){
        // Resize with padding to keep aspect ration
        emotion = (*image)(cv::Range(get<0>(*face), get<1>(*face)), cv::Range(get<2>(*face), get<3>(*face)));
        resizeKeepAspectRatio(&emotion, cv::Size(100, 100), cv::Scalar(0));
    }else if (false){
        // Resize upwards by the biggest side of the detected face and take extra parts of the image
        int h = get<1>(*face) - get<0>(*face);
        int w = get<3>(*face) - get<2>(*face);
        int centerY = (get<1>(*face) + get<0>(*face))/2;
        int centerX = (get<3>(*face) + get<2>(*face))/2;
        int x1,x2,y1,y2;
        if (h > w){
            x1 = centerX - h/2;
            y1 = centerY - h/2;
            x2 = centerX + h/2;
            y2 = centerY + h/2;
        }
        else{
            x1 = centerX - w/2;
            y1 = centerY - w/2;
            x2 = centerX + w/2;
            y2 = centerY + w/2;
        }
        emotion = (*image)(cv::Range(y1,y2), cv::Range(x1,x2));
    }

    emotion = cv::dnn::blobFromImage(emotion, 1.0, cv::Size(100, 100), cv::Scalar(0, 0, 0), false, false);
    (*netEmotion).setInput(emotion, "inputLayer");

    cv::Mat outEmotion = (*netEmotion).forward("StatefulPartitionedCall/StatefulPartitionedCall/ResNet/activationLayer/Softmax");
    int emotionOut = -1;
    float maxEmotion;
    maxEmotion = 0.0;
    for (int j = 0; j < outEmotion.cols; j++)
    {
        float curEmotion = outEmotion.at<float>(0, j);
        if (maxEmotion <= curEmotion){
            maxEmotion = curEmotion;
            emotionOut = j;
        }
    }
    (*emotions).push_back(std::make_tuple(emotionOut, maxEmotion));
}