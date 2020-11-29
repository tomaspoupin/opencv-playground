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

    double basicKmeansElbowSegmentation(cv::Mat &featureMat, cv::Mat &labels, cv::Mat &centroids, 
        int attemps, int numIter, int range)
    {
        double distortion[range];
        double lastSlope = 0, maxDiff = -1, lastDistortion;
        cv::Mat lastLabels, lastClusters;

        for (int k = 1; k <= range; ++k)
        {
            cv::Mat bestLabels, clusterCenters;
            cv::TermCriteria critera;

            critera.type = critera.COUNT;
            critera.maxCount = numIter;

            double compactness = cv::kmeans(
                featureMat,
                k,
                bestLabels,
                critera, attemps,
                cv::KMEANS_RANDOM_CENTERS,
                clusterCenters);

            distortion[k - 1] = compactness / featureMat.size().height;

            if (k == 2)
            {
                lastSlope = lastDistortion - distortion[k - 1];
            }
            else if (k > 2)
            {
                double slope = lastDistortion - distortion[k - 1];

                if (slope < 0)
                    continue;

                double diff = lastSlope - slope;

                if (diff < maxDiff)
                    break;

                maxDiff = diff;
                lastSlope = slope;
            }

            if (k != range)
            {
                lastLabels = bestLabels;
                lastClusters = clusterCenters;
                lastDistortion = distortion[k - 1];
            }
        }

        centroids = lastClusters;
        labels = lastLabels;
    }

    double basicKmeansSegmentation(cv::Mat &featureMat, cv::Mat &labels, cv::Mat &centroids,
        int clusters, int attemps, int numIter)
    {
        cv::TermCriteria critera;

        critera.type = critera.COUNT;
        critera.maxCount = numIter;

        double compactness = cv::kmeans(
            featureMat,
            clusters,
            labels,
            critera, attemps,
            cv::KMEANS_RANDOM_CENTERS,
            centroids);

        return compactness;
    }
} // namespace cvpg