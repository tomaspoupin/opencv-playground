#include <opencv_playground/algorithms/algorithms.hpp>

cvpg::Mog::Mog(int history,
               int threshold,
               bool shadowDetection,
               double learningRate)
    : history(history), threshold(threshold), shadowDetection(shadowDetection), learningRate(learningRate)
{
    mog = cv::createBackgroundSubtractorMOG2(history, threshold, shadowDetection);
}

void cvpg::Mog::apply(cv::Mat &img, cv::Mat &fg)
{
    mog->apply(img, fg, learningRate);
}