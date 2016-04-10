#include <stdlib.h>
#include "opencv2/opencv.hpp"
#include "opencv2/ml/ml.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <stdio.h>
#include <string.h>
#include <fstream>
using namespace std;
using namespace cv;

////////////////////////////////////////
#define CLASSES 7								// Number of distinct labels.
#define ATTRIBUTES 16							// Number of pixels per sample (32x32)
#define ALL_ATTRIBUTES (ATTRIBUTES*ATTRIBUTES)  // All pixels per sample.
#define INPUT_PATH_XML "C:\\Users\\StoffesBok\\Desktop\\AI-projekt\\param.xml"
#define CONTOUR_SIZE 35							// Accept found letters bigger than this size
#define SCENE_SIZE_X 800						// Render in this size
#define SCENE_SIZE_Y 600
////////////////////////////////////////

void scaleDownImage(cv::Mat &originalImg, cv::Mat &scaledDownImage)
{
	for (int x = 0; x<ATTRIBUTES; x++)
	{
		for (int y = 0; y<ATTRIBUTES; y++)
		{
			int yd = ceil((float)(y*originalImg.cols / ATTRIBUTES));
			int xd = ceil((float)(x*originalImg.rows / ATTRIBUTES));
			scaledDownImage.at<uchar>(x, y) = originalImg.at<uchar>(xd, yd);
		}
	}
}

void convertToPixelValueArray(cv::Mat &img, int pixelarray[])
{
	int i = 0;
	for (int x = 0; x<ATTRIBUTES; x++)
	{
		for (int y = 0; y<ATTRIBUTES; y++)
		{
			pixelarray[i] = (img.at<uchar>(x, y) == 255) ? 1 : 0;
			i++;
		}
	}
}

std::vector<cv::Rect> detectLetters(cv::Mat img)
{
	std::vector<cv::Rect> boundRect;
	cv::Mat img_gray, img_sobel, img_threshold, img_denoise, element;
	cvtColor(img, img_gray, CV_BGR2GRAY); // Make the image gray
	//fastNlMeansDenoising(img_gray, img_denoise, 10);
	cv::Sobel(img_gray, img_sobel, CV_8U, 1, 0, 3, 5, 0, cv::BORDER_DEFAULT); // apply sobel-filter
	cv::threshold(img_sobel, img_threshold, 0, 255, CV_THRESH_OTSU + CV_THRESH_BINARY); // threshold it
	element = getStructuringElement(cv::MORPH_RECT, cv::Size(30, 30));
	cv::morphologyEx(img_threshold, img_threshold, CV_MOP_CLOSE, element); //Does the trick
	std::vector< std::vector< cv::Point> > contours;
	cv::findContours(img_threshold, contours, 0, 1);
	std::vector<std::vector<cv::Point> > contours_poly(contours.size());

	for (int i = 0; i < contours.size(); i++)
	if (contours[i].size()>CONTOUR_SIZE)
	{
		cv::approxPolyDP(cv::Mat(contours[i]), contours_poly[i], 3, true);
		cv::Rect appRect(boundingRect(cv::Mat(contours_poly[i])));

		//if (appRect.width>appRect.height)
			boundRect.push_back(appRect);
	}

	return boundRect;
}

string convertInt(int number)
{
	stringstream ss;//create a stringstream
	ss << number;//add number to the stream
	return ss.str();//return a string with the contents of the stream
}

int main(int argc, char** argv)
{
	//read the model from the XML file and create the neural network.
	CvANN_MLP nnetwork;
	CvFileStorage* storage = cvOpenFileStorage(INPUT_PATH_XML, 0, CV_STORAGE_READ);
	CvFileNode *n = cvGetFileNodeByName(storage, 0, "DigitOCR");
	nnetwork.read(storage, n);
	cvReleaseFileStorage(&storage);
	VideoCapture cap(1);

	int counter = 0;

	if (!cap.isOpened())
	{
		cout << "Could not open the camera! " << endl;
		system("pause");
		return -1;
	}

	while (true)
	{
		// Grab a frame from camera
		Mat frame;
		cap >> frame;
		resize(frame, frame, Size(SCENE_SIZE_X, SCENE_SIZE_Y), 0, 0, INTER_AREA);

		//Detect
		std::vector<cv::Rect> letterBBoxes1 = detectLetters(frame);
		
		// Have we found any letters in this frame?
		if (letterBBoxes1.size() > 0)
		{
			// Loop through the found letters
			for (int i = 0; i < letterBBoxes1.size(); i++)
			{
				Mat test_img = frame(letterBBoxes1[i]);
				Mat output;
				Mat img_gray, img_sobel, img_threshold, element;
				GaussianBlur(test_img, output, cv::Size(5, 5), 0);
				
				cvtColor(output, img_gray, CV_BGR2GRAY);
				//Sobel(img_gray, img_sobel, CV_8U, 1, 0, 3, 5, 0, cv::BORDER_DEFAULT);
				threshold(img_gray, img_threshold, 0, 255, CV_THRESH_OTSU + CV_THRESH_BINARY);
				//threshold(img_sobel, img_threshold, 150, 255, 0);
				//element = getStructuringElement(cv::MORPH_RECT, cv::Size(10, 10));
				//morphologyEx(img_threshold, img_threshold, CV_MOP_CLOSE, element); //Does the trick

				//imshow("Test...", img_threshold);
				//cvWaitKey(1);

				Mat scaledDownImage(ATTRIBUTES, ATTRIBUTES, CV_8U, cv::Scalar(0));
				int pixelValueArray[ALL_ATTRIBUTES];
				scaleDownImage(img_threshold, scaledDownImage);
				convertToPixelValueArray(scaledDownImage, pixelValueArray);

				Mat data(1, ALL_ATTRIBUTES, CV_32F);
				for (int i = 0; i < ALL_ATTRIBUTES; i++){
					data.at<float>(0, i) = (float)pixelValueArray[i];
				}

				int maxIndex = 0;
				cv::Mat classOut(1, CLASSES, CV_32F);

				//prediction
				nnetwork.predict(data, classOut);
				float value;
				float maxValue = classOut.at<float>(0, 0);
				for (int index = 1; index<CLASSES; index++)
				{
					value = classOut.at<float>(0, index);
					if (value>maxValue)
					{
						maxValue = value;
						maxIndex = index; //maxIndex is the predicted class.
					}
				}

				// Check which letter we found
				string text;
				if (maxIndex == 0)
					text = "A";
				else if (maxIndex == 1)
					text = "B";
				else if (maxIndex == 2)
					text = "C";
				else if (maxIndex == 3)
					text = "X";
				else if (maxIndex == 4)
					text = "E";
				else if (maxIndex == 5)
					text = "O";
				else if (maxIndex == 6)
					text = "T";

				// Write text (the letter) next to the found letter
				int fontFace = CV_FONT_HERSHEY_SIMPLEX;
				double fontScale = 1;
				int thickness = 2;
				Point textOrg(letterBBoxes1[i].x + letterBBoxes1[i].width / 2, letterBBoxes1[i].y - 12);
				putText(frame, text, textOrg, fontFace, fontScale, Scalar::all(55), thickness, 8);
				rectangle(frame, letterBBoxes1[i], cv::Scalar(0, 255, 0), 3, 8, 0);
			}
		}

		imshow("Letter search...", frame);
		cvWaitKey(1);
	}
}