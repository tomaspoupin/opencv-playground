#include <iostream>
#include <string>
#include <sstream>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

void reduceImage(cv::Mat &source, int reduction);
std::unique_ptr<unsigned char[]> getColorTable(int reduction);

int main(int argc, char **argv)
{
    if (argc != 3)
    {
        std::cout << "Usage: <Image reduction> <Image path>" << std::endl;
        return -1;
    }

    int reduction;
    std::stringstream buffer;
    buffer << argv[1];
    buffer >> reduction;

    if (!buffer || !reduction)
    {
        std::cout << "Invalid image reduction, must be integer." << std::endl;
        return 1;
    }

    cv::Mat image;
    image = cv::imread(argv[2], cv::IMREAD_COLOR);

    if (!image.data)
    {
        printf("No image data \n");
        return 1;
    }

    cv::namedWindow("Original Image", cv::WINDOW_AUTOSIZE);
    cv::imshow("Original Image", image);

    reduceImage(image, reduction); // Reduccion de color usando tabla de busqueda

    cv::namedWindow("Reduced Image", cv::WINDOW_AUTOSIZE);
    cv::imshow("Reduced Image", image);

    cv::waitKey(0);

    return 0;
}

void reduceImage(cv::Mat &source, int reduction)
{
    CV_Assert(source.depth() == CV_8U); // verificar depth de 8 bits
    int channels = source.channels();
    cv::Size size = source.size();

    int numRows = size.height;
    int numCols = size.width * channels;

    static auto colorTable = getColorTable(reduction);

    for (int i = 0; i < numRows; ++i)
    {
        uchar *row = source.ptr<uchar>(i);

        for (int j = 0; j < numCols; ++j)
        {
            row[j] = colorTable[row[j]];
        }
    }
}

std::unique_ptr<unsigned char[]> getColorTable(int reduction)
{
    int colorRange = 256;
    auto colorTable = std::make_unique<unsigned char[]>(colorRange);

    for (int i = 0; i < colorRange; ++i)
    {
        colorTable[i] = static_cast<unsigned char>(reduction * (i / reduction));
    }

    return colorTable;
}