#ifndef CVPG_ALGORITHMS_
#define CVPG_ALGORITHMS_
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/background_segm.hpp>

namespace cvpg
{

    class Mog
    {
    public:
        Mog(int history = 255,
            int threshold = 16,
            bool shadowDetection = true,
            double learningRate = 0.005);
        ~Mog() {}

        void apply(cv::Mat &img, cv::Mat &fg);

    private:
        cv::Ptr<cv::BackgroundSubtractorMOG2> mog;
        int history, threshold;
        bool shadowDetection;
        double learningRate;
    };

    void applyBasicFGSegmentation(const cv::Mat &src, const cv::Mat &bg, cv::Mat &dst, int threshold);
    void applyMOGSegmentation(cv::Mat &src, cv::Mat &fg, Mog &mog);
    double basicKmeansElbowSegmentation(cv::Mat &featureMat, cv::Mat &labels, cv::Mat &centroids, int attemps = 3, int numIter = 10, int range = 10);
    double basicKmeansSegmentation(cv::Mat &featureMat, cv::Mat &labels, cv::Mat &centroids,
        int clusters = 3, int attemps = 3, int numIter = 10);
} // namespace cvpg

#endif