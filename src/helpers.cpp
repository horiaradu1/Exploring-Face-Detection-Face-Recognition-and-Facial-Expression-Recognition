// Writen by Horia-Gabriel Radu - 2021-2022
// for Third Year Project at University of Manchester

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <dlib/opencv.h>
#include <dlib/image_processing.h>
#include <dlib/image_io.h>
#include <dlib/gui_widgets.h>

#include <helpers.h>

using namespace std;

void resizeKeepAspectRatio(cv::Mat *input, cv::Size outputSize, cv::Scalar padColor)
{
	int padHeight = 0;
    int padWidth = 0;
    double inputWidthRatio = outputSize.width * ((*input).rows/(double)(*input).cols);
    double inputHeightRatio = outputSize.height * ((*input).cols/(double)(*input).rows);
    
    if( inputWidthRatio <= outputSize.height){
        padHeight = (outputSize.height - inputWidthRatio) / 2;
        cv::resize((*input), (*input), cv::Size(outputSize.width, inputWidthRatio));
    }else{
        padWidth = (outputSize.width - inputHeightRatio) / 2;
        cv::resize((*input), (*input), cv::Size(inputHeightRatio, outputSize.height));
    }

    cv::copyMakeBorder((*input), (*input), padHeight, padHeight, padWidth, padWidth, cv::BORDER_CONSTANT, padColor);
}

dlib::rectangle openCVtoDlib(cv::Rect *r)
{
	return dlib::rectangle(r->x, r->y, r->width, r->height);
}

cv::Rect dlibToOpenCV(dlib::rectangle *r)
{
	return cv::Rect(cv::Point2i(r->left(), r->top()), cv::Point2i(r->right(), r->bottom()));
}

void overlappingCropWithPad(cv::Mat *imageClone, std::vector<cv::Mat> *outputCrops, std::vector<std::tuple<int, int>> *cropOffsets, std::vector<std::tuple<int, int, int>> *cropSizes, const int *pad, const int *depth){
	// std::vector<cv::Mat> outputCrops;
	std::deque<cv::Mat> croppedImgs;
	int oNr = 0;
	int level = 0;
	(*outputCrops).push_back((*imageClone));
	(*cropOffsets).push_back(std::make_tuple(0, 0));
	(*cropSizes).push_back(std::make_tuple((*imageClone).rows, (*imageClone).cols, 0));

	croppedImgs.push_back((*imageClone));

	int P = (*pad);
	while (level < (*depth))
	{
		int limit = croppedImgs.size();
		for (int i = 0; i < limit; i++)
		{
			cv::Mat img = croppedImgs[0].clone();
			croppedImgs.pop_front();
			int h = img.rows;
			int w = img.cols;
			int centerY = h/2;
			int centerX = w/2;
			cv::Mat topLeft     = img(cv::Range(0		    , centerY+P), cv::Range(0           , centerX+P));
            cv::Mat topRight    = img(cv::Range(0		    , centerY+P), cv::Range(centerX-P, w		     ));
            cv::Mat bottomLeft  = img(cv::Range(centerY-P, h		  	  ), cv::Range(0	       , centerX+P));
            cv::Mat bottomRight = img(cv::Range(centerY-P, h		  	  ), cv::Range(centerX-P, w		     ));
			(*cropOffsets).push_back(make_tuple(0		     + get<0>((*cropOffsets)[oNr]), 0          + get<1>((*cropOffsets)[oNr])));
            (*cropOffsets).push_back(make_tuple(0 		     + get<0>((*cropOffsets)[oNr]), centerX-P + get<1>((*cropOffsets)[oNr])));
            (*cropOffsets).push_back(make_tuple(centerY-P + get<0>((*cropOffsets)[oNr]), 0          + get<1>((*cropOffsets)[oNr])));
            (*cropOffsets).push_back(make_tuple(centerY-P + get<0>((*cropOffsets)[oNr]), centerX-P + get<1>((*cropOffsets)[oNr])));
            (*outputCrops).push_back(topLeft);
            (*outputCrops).push_back(topRight);
            (*outputCrops).push_back(bottomLeft);
            (*outputCrops).push_back(bottomRight);
			for (int c = 0; c < 4; c++){
                (*cropSizes).push_back(make_tuple(h/2+P, w/2+P, P));
			}
			croppedImgs.push_back(topLeft);
			croppedImgs.push_back(topRight);
			croppedImgs.push_back(bottomLeft);
			croppedImgs.push_back(bottomRight);
			oNr++;
		}
		P=P/2;
		level++;
	}
	// return outputCrops;
}

// Tan & Triggs function and calculation originally taken from:
// Tan, X., & Triggs, B. (2010). Enhanced local texture feature sets for face recognition under difficult lighting conditions.
// IEEE transactions on image processing, 19(6), 1635-1650.
// and helped supporting code - https://github.com/GeorgeSeif/Face-Recognition
// Modified by Horia-Gabriel Radu

cv::Mat tanTriggs(cv::Mat *src) {

	cv::Mat img;
	cv::Mat gaussianFilterFirst, gaussianFilterSecond;

	int sigma0 = 1;
	int sigma1 = 2;
	double mean = 0.0;
	float alpha = 0.1;
	float tau = 10.0;
	float gamma = 0.2;

	img = (*src).clone();
	img.convertTo(img, CV_32FC1);
	cv::pow(img, gamma, img);
	
	int kernelSizeFirst = (3 * sigma0);
	kernelSizeFirst += ((kernelSizeFirst % 2) == 0) ? 1 : 0;
	cv::GaussianBlur(img, gaussianFilterFirst, cv::Size(kernelSizeFirst, kernelSizeFirst), sigma0, sigma0, cv::BORDER_REPLICATE);
	int kernelSizeSecond = (3 * sigma1);
	kernelSizeSecond += ((kernelSizeSecond % 2) == 0) ? 1 : 0;
	cv::GaussianBlur(img, gaussianFilterSecond, cv::Size(kernelSizeSecond, kernelSizeSecond), sigma1, sigma1, cv::BORDER_REPLICATE);
	cv::subtract(gaussianFilterFirst, gaussianFilterSecond, img);
	
	mean = 0.0;
	cv::Mat tmp = img.clone();
	cv::pow(cv::abs(img), alpha, tmp);
	mean = cv::mean(tmp).val[0];
	img = img / cv::pow(mean, 1.0 / alpha);
	
	mean = 0.0;
	tmp = img.clone();
	cv::pow(cv::min(cv::abs(img), tau), alpha, tmp);
	mean = cv::mean(tmp).val[0];
	img = img / cv::pow(mean, 1.0 / alpha);
	
	cv::Mat exponent, exponentNegative;
	cv::exp(img / tau, exponent);
	cv::exp(-img / tau, exponentNegative);
	cv::divide(exponent - exponentNegative, exponent + exponentNegative, img);
	img = tau * img;
	
	if (img.channels() == 1){
		cv::normalize(img, img, 0, 255, cv::NORM_MINMAX, CV_8UC1);
	}else if (img.channels() == 3){
		cv::normalize(img, img, 0, 255, cv::NORM_MINMAX, CV_8UC3);
	}
	return img;
}

void convertToGray(cv::Mat *img){
	if ((*img).channels() == 3) {
		cvtColor((*img), (*img), cv::COLOR_BGRA2GRAY);
	}
	else if (
		(*img).channels() == 4) {
		cvtColor((*img), (*img), cv::COLOR_BGRA2GRAY);
	}
}

// Preprocessing steps and calculation inspiration from - https://github.com/GeorgeSeif/Face-Recognition
// Modified by Horia-Gabriel Radu

void histogramEqualization(cv::Mat *img){
	int w = (*img).cols;
	int h = (*img).rows;
	cv::Mat face;
	cv::equalizeHist((*img), face);
	
	int centerW = w / 2;
	cv::Mat right = (*img)(cv::Rect(centerW, 0, w - centerW, h));
	cv::equalizeHist(right, right);
	cv::Mat left = (*img)(cv::Rect(0, 0, centerW, h));
	cv::equalizeHist(left, left);
	
	for (int y = 0; y < h; y++) {
		for (int x = 0; x < w; x++) {
			int v;
			if (x < (w / 4)) {
				v = left.at<uchar>(y, x);
			}
			else if (x < (w * 2 / 4)) {
				int lv = left.at<uchar>(y, x);
				int wv = face.at<uchar>(y, x);
				float f = (x - (w * 1 / 4)) / (float)(w / 4);
				v = cvRound((1.0 - f) * lv + (f) * wv);
			}
			else if (x < (w * 3 / 4)) {
				int rv = right.at<uchar>(y, x - centerW);
				int wv = face.at<uchar>(y, x);
				float f = (x - (w * 2 / 4)) / (float)(w / 4);
				v = cvRound((1.0 - f) * wv + (f) * rv);
			}
			else {
				v = right.at<uchar>(y, x - centerW);
			}
			(*img).at<uchar>(y, x) = v;
		}
	}
}

void elipticalMask(cv::Mat *img, int height, int width){

	cv::Mat mask = cv::Mat((*img).size(), CV_8UC1, cv::Scalar(255));

	cv::Point center = cv::Point(cvRound(width * 0.5), cvRound(height * 0.4));
	cv::Size maskSize = cv::Size(cvRound(width * 0.5), cvRound(height * 0.8));
	ellipse(mask, center, maskSize, 0, 0, 360, cv::Scalar(0), cv::FILLED);
	
	(*img).setTo(cv::Scalar(128), mask);
}

void preprocessInputImg(cv::Mat *image){
	convertToGray(image);
	histogramEqualization(image);
	elipticalMask(image, image->rows, image->cols);
}