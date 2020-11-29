#include <opencv_playground/utils/commons.hpp>

namespace cvpg
{
    void getBlobsFromLabels(const cv::Mat srcLabels, std::vector<cv::Rect> &dstBlobs)
    {
        dstBlobs.clear();
        if (srcLabels.depth() != CV_32S)
            return;
        if (srcLabels.channels() != 1)
            return;

        for (auto it = srcLabels.begin<int>(); it != srcLabels.end<int>(); ++it)
        {
            
        }
    }
} // namespace cvpg