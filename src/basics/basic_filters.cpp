#include <iostream>
#include <string>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

int main(int argc, char **argv)
{

    if (argc != 2)
    {
        std::cerr << "USAGE <Image Path>." << std::endl;
        return 1;
    }

    cv::Mat src = cv::imread(argv[1], cv::IMREAD_COLOR);

    if (src.empty())
    {
        std::cerr << "Failed to parse image." << std::endl;
        return 1;
    }

    cv::imshow("Original", src);

    cv::Mat boxFilterImage, gaussianImage, medianImage;

    // Primero usamos box filter
    std::vector<int> sizes{5, 11, 15};

    std::stringstream imageText;

    for (auto &&i : sizes)
    {
        imageText << "Box Filter Size: " << i;
        cv::boxFilter(src, boxFilterImage, -1, cv::Size(i, i));
        cv::imshow(imageText.str(), boxFilterImage);
        imageText.str("");
    }

    std::for_each(sizes.begin(), sizes.end(), [&](int i) {
        imageText << "Gaussian Filter Size: " << i;
        cv::GaussianBlur(src, gaussianImage, cv::Size(i, i), 5.0);
        cv::imshow(imageText.str(), gaussianImage);
        imageText.str("");
    });

    for (auto &&i : sizes)
    {
        imageText << "Median Filter Size: " << i;
        cv::medianBlur(src, medianImage, i);
        cv::imshow(imageText.str(), medianImage);
        imageText.str("");
    }

    cv::waitKey(0);
}