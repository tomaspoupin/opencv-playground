#include <iostream>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv_playground/algorithms/algorithms.hpp>

const std::string infoText = "Aprete cualquier tecla para capturar el fondo,"
                             "aprete nuevamente cualquier tecla para salir.";
const int threshold = 120;

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

    cv::namedWindow("Main Video");
    cv::displayOverlay("Main Video", infoText, 10000);

    cv::Mat currentFrame, background, fgMask, mogFgMask;

    cvpg::Mog mog; // Objeto que representa el estado de mixture of gaussians

    int keyCount = 0;
    while (1)
    {
        webCam >> currentFrame;

        cv::imshow("Main Video", currentFrame);

        if (!background.empty())
        {
            cvpg::applyBasicFGSegmentation(currentFrame, background, fgMask, threshold);
            cvpg::applyMOGSegmentation(currentFrame, mogFgMask, mog);
            cv::imshow("Basic fg segmentation", fgMask);
            cv::imshow("MOG fg segmentation", mogFgMask);
        }

        int key = cv::waitKey(10);
        if (key != -1 && keyCount == 0)
        {
            currentFrame.copyTo(background);
            keyCount++;
        }
        else if (key != -1 && keyCount == 1)
            break;
    }
}