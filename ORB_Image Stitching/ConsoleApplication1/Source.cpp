#define _CRT_SECURE_NO_DEPRECATE
#include <stdio.h>
#include <iostream>
#include "opencv2/core.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/xfeatures2d/nonfree.hpp"
#include <opencv2/opencv.hpp>
#include "opencv2/stitching.hpp"


using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;


Mat cal_homography_matrix(Mat image1, Mat image2, int features, int layers) {

	//Ptr<SIFT> detector = SIFT::create(features, layers);
	Ptr<ORB> detector = ORB::create(features);

	std::vector<KeyPoint> keypoints_1, keypoints_2;

	detector->detect(image1, keypoints_1);
	detector->detect(image2, keypoints_2);

	//-- Draw keypoints
	Mat img_keypoints_1; Mat img_keypoints_2;

	drawKeypoints(image1, keypoints_1, img_keypoints_1, Scalar::all(-1),
		DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
	drawKeypoints(image2, keypoints_2, img_keypoints_2, Scalar::all(-1),
		DrawMatchesFlags::DRAW_RICH_KEYPOINTS);



	//-- Step 2: Calculate descriptors (feature vectors)
	Mat descriptors_1, descriptors_2;
	detector->compute(image1, keypoints_1, descriptors_1);
	detector->compute(image2, keypoints_2, descriptors_2);

	//-- Show detected (drawn) keypoints

	imshow("Keypoints 1", img_keypoints_1);
	imshow("Keypoints 2", img_keypoints_2);


	//-- Step 3: Matching descriptor vectors using BFMatcher :
	BFMatcher matcher;
	std::vector<std::vector<cv::DMatch>> matches;

	//https://github.com/opencv/opencv/issues/5937 Why ORB and FLANN issues.
	//FlannBasedMatcher matcher;
	//	std::vector< DMatch > matches;
	//	matcher.match(descriptors_1, descriptors_2, matches);
	//	matcher.match(descriptors_1, descriptors_2, matches);



	//	double max_dist = 0; double min_dist = 100;

	//-- Quick calculation of max and min distances between keypoints 
	//	for (int i = 0; i < descriptors_1.rows; i++)
	//	{
	//		double dist = matches[i].distance;
	//		if (dist < min_dist) min_dist = dist;
	//		if (dist > max_dist) max_dist = dist;
	//	}

	//	printf("-- Max dist: %f \n", max_dist);
	//	printf("-- Min dist: %f \n", min_dist);


	//-- Use only "good" matches (i.e. whose distance is less than 3*min_dist )
	//std::vector< DMatch > good_matches;
	matcher.knnMatch(descriptors_1, descriptors_2, matches, 2);  // Find two nearest matches

	vector<cv::DMatch> good_matches;
	for (int i = 0; i < matches.size(); ++i)
	{
		const float ratio = 0.8; // As in Lowe's paper; can be tuned
		if (matches[i][0].distance < ratio * matches[i][1].distance)
		{
			good_matches.push_back(matches[i][0]);
		}
	}

	cv::Mat H;
	//for (int i = 0; i < descriptors_1.rows; i++)
	//{
	//	if (matches[i].distance < 3* min_dist)
	//	{
	//		good_matches.push_back(matches[i]);
	//	}
	//}
	std::vector< Point2f > obj;
	std::vector< Point2f > scene;

	for (int i = 0; i < good_matches.size(); i++)
	{
		//-- Get the keypoints from the good matches
		obj.push_back(keypoints_1[good_matches[i].queryIdx].pt);
		scene.push_back(keypoints_2[good_matches[i].trainIdx].pt);
	}


	// Find the Homography Matrix for img 1 and img2
	H = findHomography(obj, scene, CV_RANSAC);
	//H is the transformation. Using ransac to remove the outliers
	return H;
}
Mat stitch_image(Mat image1, Mat image2, Mat H)
{


	cv::Mat result;
	warpPerspective(image1, result, H, cv::Size(image1.cols + image2.cols, image1.rows));
	cv::Mat half(result, cv::Rect(0, 0, image2.cols, image2.rows));
	image2.copyTo(half);
	return result;

}

Mat crop_image(Mat result) {
	// //Finding the largest contour i.e remove the black region from image 

	Mat img_gray;
	img_gray = result.clone();
	img_gray.convertTo(img_gray, CV_8UC1);
	cvtColor(img_gray, img_gray, COLOR_BGR2GRAY);
	threshold(img_gray, img_gray, 25, 255, THRESH_BINARY); //Threshold the gray 

	vector<vector<Point> > contours; // Vector for storing contour 
	vector<Vec4i> hierarchy;
	findContours(img_gray, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE); // Find the contours in the image 
	int largest_area = 0;
	int largest_contour_index = 0;
	Rect bounding_rect;


	for (int i = 0; i< contours.size(); i++) // iterate through each contour.  
	{
		double a = contourArea(contours[i], false);  //  Find the area of contour 
		if (a>largest_area) {
			largest_area = a;
			largest_contour_index = i;                //Store the index of largest contour 
			bounding_rect = boundingRect(contours[i]); // Find the bounding rectangle for biggest contour 

		}

	}
	result = result(Rect(bounding_rect.x, bounding_rect.y, bounding_rect.width, bounding_rect.height));
	return result;

}
void getMSE(const Mat& I1, const Mat& I2)
{
	Mat s1;
	Mat I2_resized;
	vector<Mat> channels(3);
	// split img:
	split(I2, channels);
	I2_resized = channels[0];
	cv::resize(I2, I2_resized, I1.size());
	absdiff(I1, I2_resized, s1);
	s1.convertTo(s1, CV_32F);  // cannot make a square on 8 bits
	s1 = s1.mul(s1);           // |I1 - I2|^2

	Scalar s = sum(s1);        // sum elements per channel

	double sse = s.val[0]; // sum channels


	double mse = sse / (double)(I1.total());
	cout << sse;
	cout << "\n";
	cout << I1.total();
	printf("mse = %f", mse);

}
void readme();
/** @function main */
int main(int argc, char** argv)
{
	/*References
	https://stackoverflow.com/questions/27533203/how-do-i-use-sift-in-opencv-3-0-with-c
	http://docs.opencv.org/3.3.0/d5/d3c/classcv_1_1xfeatures2d_1_1SIFT.html#details
	http://kushalvyas.github.io/stitching.html
	https://github.com/Manasi94/Image-Stitching
	*/
	int64 t0 = cv::getTickCount();

	int nOctaveLayers = 3; //lowes paper suggests
	int nfeatures = 2000; //number of features

	Mat img_1 = imread("goldengate-02.png");
	Mat img_2 = imread("goldengate-01.png");
	Mat img_3 = imread("goldengate-00.png");
	Mat img_orginal = imread("Original_gray.jpg");

	img_1.convertTo(img_1, CV_8UC1);
	img_2.convertTo(img_2, CV_8UC1);
	img_3.convertTo(img_3, CV_8UC1);

	//cvtColor(img_1, img_1, COLOR_RGB2GRAY);
	//cvtColor(img_2, img_2, COLOR_RGB2GRAY);
	//cvtColor(img_3, img_3, COLOR_RGB2GRAY);

	Mat H23; Mat result23;
	Mat H12; Mat result12;
	Mat H123; Mat result123;
	Mat H_t; Mat result_t;
	H23 = cal_homography_matrix(img_1, img_2, nfeatures, nOctaveLayers);
	result23 = stitch_image(img_1, img_2, H23);


	H_t = cal_homography_matrix(result23, img_3, nfeatures, nOctaveLayers);
	result_t = stitch_image(result23, img_3, H_t);

	/* Using stitcher class : Read on this
	Mat pano;
	bool try_use_gpu = false;
	vector<Mat> imgs;
	imgs.reserve(2);
	//Mat img;
	//for (int i = 1; i <3 ; i++)
	//{
	//	imread("carmel-0" + to_string(i) + ".png");
	//	imgs.push_back(img.clone());
	//}
	imgs.push_back(img_1);
	imgs.push_back(img_2);
	Stitcher::Mode mode = Stitcher::PANORAMA;
	Ptr<Stitcher> stitcher = Stitcher::create(mode, try_use_gpu);
	Stitcher::Status status = stitcher->stitch(imgs,pano);
	imshow("Pano", pano);*/
	//result23 = crop_image(result23);
	//result12 = crop_image(result12);


	//imshow("Stitched Image 23", result23);
	result_t = crop_image(result_t);
	imshow("Stitched Image _t", result_t);
	//imwrite("result_1.jpg", result_t);
	cvtColor(result_t, result_t, COLOR_RGB2GRAY);
	int64 t1 = cv::getTickCount();
	double secs = (t1 - t0) / cv::getTickFrequency();
	cout << "Time elapsed:" << secs << "seconds";
	imwrite("result.jpg", result_t);
	//getMSE(img_orginal,result_t);
	waitKey(0);
	return 0;
}
/** @function readme */
void readme()
{
	std::cout << " " << std::endl;
}