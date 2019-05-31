#include "retinaface.h"
#include <opencv2/opencv.hpp>


RetinaFace *mRetinaFace;
cv::TickMeter tm;


void detect(cv::Mat & img)
{
	
	tm.reset();
	tm.start();
	//ncnn::Mat input = ncnn::Mat::from_pixels_resize(img.data, ncnn::Mat::PIXEL_BGR, img.cols, img.rows, img.cols, img.rows);
	ncnn::Mat input = ncnn::Mat::from_pixels_resize(img.data, ncnn::Mat::PIXEL_BGR2RGB, img.cols, img.rows, img.cols, img.rows);
	//cv::resize(img, img, cv::Size(300, 300));

	// nms
	std::vector<Anchor> result;
	mRetinaFace->detect(input, result);

	//printf("final result %d\n", result.size());
	for (int i = 0; i < result.size(); i++)
	{
		cv::rectangle(img, cv::Point((int)result[i].finalbox.x , (int)result[i].finalbox.y ),
			cv::Point((int)result[i].finalbox.width, (int)result[i].finalbox.height),
			cv::Scalar(0, 255, 255), 2, 8, 0);
		for (int j = 0; j < result[i].pts.size(); ++j) {
			cv::circle(img, cv::Point((int)result[i].pts[j].x , (int)result[i].pts[j].y ), 
				1, cv::Scalar(225, 0, 225), 2, 8);
		}
	}
	//result[0].print();
	tm.stop();
	std::cout << tm.getTimeMilli() << std::endl;
}

int main() {
    
	mRetinaFace = new RetinaFace("./models");

	cv::VideoCapture cap(0);
	if (!cap.isOpened())
		return -1;
	cv::Mat frame;
	for (;;)
	{
		cap >> frame;
		if (frame.empty())
			continue;
		cv::flip(frame, frame, 1);
		detect(frame);

		cv::imshow("img", frame);
		if (cv::waitKey(10) == 'q')
			break;
	}
	cv::destroyAllWindows();
	cap.release();
    return 0;
}


