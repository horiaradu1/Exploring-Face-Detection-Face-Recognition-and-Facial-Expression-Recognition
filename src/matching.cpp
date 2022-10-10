// Writen by Horia-Gabriel Radu - 2021-2022
// for Third Year Project at University of Manchester

#include <string>
#include <iostream>

#include <opencv2/highgui.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/face.hpp>
#include <opencv2/photo.hpp>

#include <dlib/dnn.h>
#include <dlib/image_processing/frontal_face_detector.h>

#include <matching.h>
#include <helpers.h>

using namespace std;

cv::Mat prepareFace (cv::Rect *opencvRect, dlib::shape_predictor *shape_model, cv::Mat *image)
{
    cv::Mat outputImg;
    cv::Mat temp;

    //cv::convertScaleAbs((*face), (*face));
    //cv::addWeighted((*face), 1.5, (*face), 20, 0, (*face));

    //(*face) = resizeKeepAspectRatio((*face), cv::Size(160, 160), cv::Scalar(255));
    //cv::cvtColor((*image), temp, cv::COLOR_BGR2GRAY);
    //cv::medianBlur((*face), (*face), 7);
    //cv::fastNlMeansDenoising(temp, temp, 10, 7, 21);
    // cv::GaussianBlur((*image), temp, cv::Size(3, 3), 99);
    //cv::blur((*face), (*face), cv::Size(7,7));

    //dlib::cv_image<unsigned char> dlibImg(temp);
    dlib::cv_image<dlib::rgb_pixel> dlibImg((*image));
    std::vector<dlib::full_object_detection> shapes;
    dlib::rectangle dlibRect;
    
	if ((*opencvRect).width != 0 | (*opencvRect).height != 0){
		dlibRect = openCVtoDlib(opencvRect);
		dlib::full_object_detection shape = (*shape_model)(dlibImg, dlibRect);
    	shapes.push_back(shape);
	}
    
	if (shapes.size() == 0) {
		return outputImg;
	}

    // Processing of the shape of the face and alignment inspired from - https://github.com/GeorgeSeif/Face-Recognition

    dlib::point top = shapes[0].part(17);
    dlib::point left = shapes[0].part(4);
    dlib::point bottom = shapes[0].part(11);
    dlib::point right = shapes[0].part(12);

    for (int x = 1; x <= 67; x++){
        if (shapes[0].part(x).y() < top.y()){
            top = shapes[0].part(x);
        }
        if (shapes[0].part(x).x() < left.x()){
            left = shapes[0].part(x);
        }
        if (shapes[0].part(x).x() > right.x()){
            right = shapes[0].part(x);
        }
    }

    dlib::point leftEye = shapes[0].part(36);
    dlib::point rightEye = shapes[0].part(45);
    dlib::point dlibEyeCenter = shapes[0].part(27);

    cv::Point2i eyeCenter(dlibEyeCenter.x(), dlibEyeCenter.y());

    double dy = (rightEye.y() - leftEye.y());
    double dx = (rightEye.x() - leftEye.x());
    double len = sqrt(dx*dx + dy*dy);
    double angle = atan2(dy, dx) * 180.0 / CV_PI;

    cv::Mat rotMat(2, 3, CV_32FC1);

    rotMat = cv::getRotationMatrix2D(eyeCenter, angle, 1);

    dlib::point topAlign(   (long)top.x()   *rotMat.at<double>(0, 0) + (long)top.y()   *rotMat.at<double>(0, 1) + rotMat.at<double>(0, 2),
                            (long)top.x()   *rotMat.at<double>(1, 0) + (long)top.y()   *rotMat.at<double>(1, 1) + rotMat.at<double>(1, 2));
    dlib::point bottomAlign((long)bottom.x()*rotMat.at<double>(0, 0) + (long)bottom.y()*rotMat.at<double>(0, 1) + rotMat.at<double>(0, 2),
                            (long)bottom.x()*rotMat.at<double>(1, 0) + (long)bottom.y()*rotMat.at<double>(1, 1) + rotMat.at<double>(1, 2));
    dlib::point rightAlign( (long)right.x() *rotMat.at<double>(0, 0) + (long)right.y() *rotMat.at<double>(0, 1) + rotMat.at<double>(0, 2),
                            (long)right.x() *rotMat.at<double>(1, 0) + (long)right.y() *rotMat.at<double>(1, 1) + rotMat.at<double>(1, 2));
    dlib::point leftAlign(  (long)left.x()  *rotMat.at<double>(0, 0) + (long)left.y()  *rotMat.at<double>(0, 1) + rotMat.at<double>(0, 2),
                            (long)left.x()  *rotMat.at<double>(1, 0) + (long)left.y()  *rotMat.at<double>(1, 1) + rotMat.at<double>(1, 2));

    dlib::array<dlib::array2d<unsigned char> > box_chips;
    dlib::extract_image_chips(dlibImg, dlib::get_face_chip_details(shapes), box_chips);

    // dlib::rectangle faceRect;

    dlib::rectangle faceRect = dlib::rectangle((long)leftAlign.x(), (long)topAlign.y(), (long)rightAlign.x(), (long)bottomAlign.y());

    dlib::matrix<unsigned char> dlibFace = dlib::tile_images(box_chips);

    cv::Mat cvFace = dlib::toMat(dlibFace);

    cv::Rect OpenCVBox = dlibToOpenCV(&faceRect);

    temp = (*image).clone();

    cv::warpAffine(temp, temp, rotMat, temp.size());

    if (!(0 <= OpenCVBox.x && 0 <= OpenCVBox.width && OpenCVBox.x + OpenCVBox.width <= temp.cols
    	&& 0 <= OpenCVBox.y && 0 <= OpenCVBox.y && OpenCVBox.y + OpenCVBox.height <= temp.rows)){
    	return outputImg;
    }

    outputImg = temp(OpenCVBox);

    // outputImg = tanTriggs(&outputImg);
    // preprocessInputImg(&outputImg);

    cv::resize(outputImg, outputImg, cv::Size(300, 300));
    // outputImg = resizeKeepAspectRatio(outputImg, cv::Size(300, 300), cv::Scalar(128));

    cv::cvtColor(outputImg, outputImg, cv::COLOR_BGR2GRAY);

    return outputImg;
}

// Neural network info taken from http://dlib.net/dnn_face_recognition_ex.cpp.html
void recogniseFacesML(int i, cv::Mat *image, std::tuple<int, int, int ,int> *face, std::vector<std::tuple<int, std::string, double, float>> *names, std::vector<std::tuple<int, std::string>> *labels, std::vector<dlib::matrix<float, 0, 1>> *face_descriptors, anet_type *anet, dlib::shape_predictor *sp){
    cv::Rect face_rect(get<2>((*face)), get<0>((*face)), get<3>((*face)), get<1>((*face)));

    // dlib::cv_image<dlib::bgr_pixel> matrix(imageClone);
    // dlib::matrix<dlib::rgb_pixel> dlibImg;
    // dlib::assign_image(dlibImg, matrix);

    dlib::cv_image<dlib::rgb_pixel> dlibImg((*image));
    dlib::rectangle dlibRect = openCVtoDlib(&face_rect);

    auto shape = (*sp)(dlibImg, dlibRect);
    dlib::matrix<dlib::rgb_pixel> face_chip;
    dlib::extract_image_chip(dlibImg, dlib::get_face_chip_details(shape, 150, 0.25), face_chip);
    
    dlib::matrix<float, 0, 1> descriptor = (*anet)(face_chip);
    int best_match = -1;
    float best_score = 1;
    for (int j = 0; j < (*face_descriptors).size(); ++j){
        if (length((*face_descriptors)[j]-descriptor) < 0.6){
            if (best_score > length((*face_descriptors)[j]-descriptor)){
                best_score = length((*face_descriptors)[j]-descriptor);
                best_match = j;
            }
        }
    }
    if (best_match != -1){
        (*names).push_back(make_tuple(i, get<1>((*labels)[best_match]), 100 - (length((*face_descriptors)[best_match]-descriptor) * 100), length((*face_descriptors)[best_match]-descriptor)));
    }else{
        (*names).push_back(make_tuple(-1, "n/a", -1, -1));
    }
}

void recogniseFaces(int i, cv::Mat *image, std::tuple<int, int, int ,int> *face, std::vector<std::tuple<int, std::string, double, float>> *names, cv::Ptr<cv::face::LBPHFaceRecognizer> *model, std::vector<std::tuple<int, std::string>> *labels, dlib::shape_predictor *shape_model){
    cv::Mat facePrep;
    cv::Rect opencvRect(get<2>((*face)), get<0>((*face)), get<3>((*face)), get<1>((*face)));
    facePrep = prepareFace(&opencvRect, shape_model, image);
    if (!(facePrep.empty())){
        double conf = 0.0;
        double percent = 0;
        int labelID = -1;

        (*model)->predict(facePrep, labelID, conf);
        
        if(conf == 0){
            percent = 100;
        }
        else if((0 <= conf)&(conf <= 100)){
            percent = 100 - (conf);
        }
        else{
            percent = 0;
        }

        (*names).push_back(make_tuple(i, get<1>((*labels)[labelID]), percent, conf));
    }else{
        (*names).push_back(make_tuple(-1, "n/a", -1, -1));
    }
}

void recogniseFacesEigen(int i, cv::Mat *image, std::tuple<int, int, int ,int> *face, std::vector<std::tuple<int, std::string, double, float>> *names, cv::Ptr<cv::face::EigenFaceRecognizer> *model, std::vector<std::tuple<int, string>> *labels, dlib::shape_predictor *shape_model){
    cv::Mat facePrep;
    cv::Rect opencvRect(get<2>((*face)), get<0>((*face)), get<3>((*face)), get<1>((*face)));
    facePrep = prepareFace(&opencvRect, shape_model, image);
    if (!(facePrep.empty())){
        double conf = 0.0;
        double percent = 0;
        int labelID = -1;

        (*model)->predict(facePrep, labelID, conf);

        (*names).push_back(make_tuple(i, get<1>((*labels)[labelID]), conf, conf));
    }else{
        (*names).push_back(make_tuple(-1, "n/a", -1, -1));
    }
}

void recogniseFacesFisher(int i, cv::Mat *image, std::tuple<int, int, int ,int> *face, std::vector<std::tuple<int, std::string, double, float>> *names, cv::Ptr<cv::face::FisherFaceRecognizer> *model, std::vector<std::tuple<int, string>> *labels, dlib::shape_predictor *shape_model){
    cv::Mat facePrep;
    cv::Rect opencvRect(get<2>((*face)), get<0>((*face)), get<3>((*face)), get<1>((*face)));
    facePrep = prepareFace(&opencvRect, shape_model, image);
    if (!(facePrep.empty())){
        double conf = 0.0;
        double percent = 0;
        int labelID = -1;

        (*model)->predict(facePrep, labelID, conf);

        (*names).push_back(make_tuple(i, get<1>((*labels)[labelID]), conf, conf));
    }else{
        (*names).push_back(make_tuple(-1, "n/a", -1, -1));
    }
}