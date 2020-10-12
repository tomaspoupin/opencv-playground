#include <iostream>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#define INHERIT -1
const double sigma = 2.0;
const int ksize = 3;
const int minThreshold = 50;
const int maxThreshold = 150;

int main(int argc, char **argv)
{
    if (argc != 2)
    {
        std::cerr << "USAGE <Device Path>." << std::endl;
        return 1;
    }

    cv::VideoCapture webCam;
    webCam.open(argv[1]);

    if (!webCam.isOpened())
    {
        std::cerr << "Unable to open video device." << std::endl;
        return 1;
    }

    cv::Mat grayImage, currentFrame, sobelImage, cannyImage;
    while (1)
    {
        webCam >> currentFrame;
        cv::cvtColor(currentFrame, grayImage, cv::COLOR_BGR2GRAY);

        cv::GaussianBlur(grayImage, grayImage, cv::Size(ksize, ksize), sigma);

        cv::Sobel(grayImage, sobelImage, INHERIT, 1, 1, ksize);
        cv::Canny(grayImage, cannyImage, minThreshold, maxThreshold, ksize);

        cv::imshow("Sobel Gradient Without Prefiltering", sobelImage);
        cv::imshow("Canny Image", cannyImage);
        if (cv::waitKey(10) != -1)
            break;
    }

    webCam.release();
}