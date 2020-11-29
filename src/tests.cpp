#include <iostream>
#include <string>
#include <opencv_playground/algorithms/algorithms.hpp>
#include <opencv2/ximgproc/slic.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

int main(int argc, char **argv)
{
    double threshold = 100.0;

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

    std::vector<cv::Mat> channels;

    cv::split(src, channels);

    // cv::threshold(channels[0], channels[0], 50, 255, 3);
    // cv::threshold(channels[1], channels[1], 50, 255, 3);
    // cv::threshold(channels[2], channels[2], 50, 255, 3);

    cv::Mat shelf, hsvShelf;
    shelf = src;
    // cv::merge(channels, shelf);

    cv::cvtColor(shelf, hsvShelf, cv::COLOR_BGR2HSV);

    int subdivisions = 100;
    // int windowSize = 50;

    cv::Size shelfSize(shelf.size());

    int colSize = (double)shelfSize.width / (double)subdivisions;

    std::vector<cv::Rect> regionRect;
    std::vector<cv::Mat> regionMat;
    std::vector<cv::Mat> shelfHisto;

    // Quantize the hue to 30 levels
    // and the saturation to 32 levels
    int hbins = 16, sbins = 16;
    int histSize[] = {hbins, sbins};
    // hue varies from 0 to 179, see cvtColor
    float hranges[] = {0, 180};
    // saturation varies from 0 (black-gray-white) to
    // 255 (pure spectrum color)
    float sranges[] = {0, 256};
    const float *ranges[] = {hranges, sranges};
    // we compute the histogram from the 0-th and 1-st channels
    int histChannels[] = {0, 1};

    for (int i = 0; i < subdivisions; ++i)
    {
        regionRect.push_back(cv::Rect(
            i * colSize,
            0,
            colSize,
            shelfSize.height));
        cv::Mat regionOfInterest = cv::Mat(hsvShelf, regionRect[i]);
        cv::Mat hist;
        cv::calcHist(&regionOfInterest, 1, histChannels, cv::Mat(), // do not use mask
                     hist, 2, histSize, ranges,
                     true, // the histogram is uniform
                     false);
        std::cout << "Antes: " << hist << std::endl;
        cv::normalize(hist, hist, 1, 0, cv::NORM_L2);
        std::cout << "Despues: " << hist << std::endl;
        shelfHisto.push_back(hist);
    }

    cv::Mat histDistance(
        shelfHisto.size(),
        shelfHisto.size(),
        CV_32FC1);

    for (int i = 0; i < shelfHisto.size(); ++i)
    {
        for (int j = 0; j < shelfHisto.size(); ++j)
        {
            double distance = cv::compareHist(shelfHisto[i], shelfHisto[j], cv::HISTCMP_BHATTACHARYYA);
            histDistance.at<float>(i, j) = distance;
        }
    }

    cv::Mat featureMat(shelfHisto.size(), shelfHisto.size() + 1, CV_32FC1);

    for (int i = 0; i < featureMat.rows; ++i)
    {
        for (int j = 0; j < featureMat.cols; ++j)
        {
            if (j == 0)
                featureMat.at<float>(i, j) = cv::pow(10*(double)(j + 1), 2) / (double)shelfHisto.size();
            else
                featureMat.at<float>(i, j) = histDistance.at<float>(i, j);
        }
    }

    cv::Mat centroids, labels;
    int clusters = 7;

    double compactness = cvpg::basicKmeansSegmentation(featureMat, labels, centroids, clusters, 5, 100);

    // cv::Mat labels(1, shelfHisto.size(), CV_32S);
    cv::Mat mask(shelf.size(), CV_8UC3);

    cv::RNG rng(123456);

    std::vector<cv::Scalar> colors;
    for (int i = 0; i < clusters; ++i)
    {
        colors.push_back(cv::Scalar(rng.uniform(0, 255),
                                    rng.uniform(0, 255),
                                    rng.uniform(0, 255)));
    }

    // std::cout << histDistance << std::endl;

    std::vector<cv::Rect> areasInteres;
    int counter = 0, startPoint = 0;
    for (int i = 0; i < subdivisions; ++i)
    {
        startPoint = i;
        int width = colSize, height = shelfSize.height, exit = false;
        if (i != subdivisions - 1)
        {
            for (int j = i + 1; j < subdivisions; ++j)
            {
                if (labels.at<int>(i) == labels.at<int>(j))
                {
                    if (j == subdivisions - 1)
                        exit = true;
                    width += colSize;
                }
                else
                {
                    i = j - 1;
                    break;
                }
            }
        }
        areasInteres.push_back(cv::Rect(startPoint * colSize, 0, width, height));
        cv::Mat temp(mask, areasInteres[counter]);
        temp = colors[labels.at<int>(i)];
        counter++;
        if (exit == true)
            break;
    }
    cv::imshow("Mask", mask);

    cv::waitKey(0);
}