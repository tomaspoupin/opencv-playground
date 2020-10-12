#include <opencv_playground/algorithms/algorithms.hpp>

namespace cvpg
{
    void applyBasicFGSegmentation(const cv::Mat &src, const cv::Mat &bg, cv::Mat &dst, int threshold)
    {
        int bgWidth = bg.size().width, bgHeight = bg.size().height;
        int srcWidth = src.size().width, srcHeight = src.size().height;

        if (bgWidth != srcWidth || bgHeight != srcHeight)
        {
            return; // retornamos matriz vacia
        }

        dst.create(srcWidth, srcHeight, CV_8UC1);

        cv::Mat diff;

        cv::absdiff(src, bg, diff);

        int dist, t2 = threshold * threshold;
        int c1 = 3;
        for (int i = 0; i < srcHeight; ++i)
        {
            uchar *diffRow = diff.ptr(i), *dstRow = dst.ptr(i);
            for (int j = 0; j < srcWidth; ++j)
            {
                int diffIndex = j * c1;
                dist = diffRow[diffIndex] * diffRow[diffIndex] +
                       diffRow[diffIndex + 1] * diffRow[diffIndex + 1] +
                       diffRow[diffIndex + 2] * diffRow[diffIndex + 2];

                if (dist > t2)
                    dstRow[j] = 255;
                else
                    dstRow[j] = 0;
            }
        }

        return;
    }

    void applyMOGSegmentation(cv::Mat &src, cv::Mat &fg, Mog &mog)
    {
        mog.apply(src, fg);
    }
} // namespace cvpg