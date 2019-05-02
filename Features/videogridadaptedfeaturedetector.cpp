#include "videogridadaptedfeaturedetector.h"

using namespace std;

VideoGridAdaptedFeatureDetector::VideoGridAdaptedFeatureDetector(const cv::Ptr<StatefulFeatureDetector>& pDetector,
    int maxTotalKeypoints, int gridRows, int gridCols, int edgeThreshold)
    : mMaxTotalKeypoints(maxTotalKeypoints)
    , mGridRows(gridRows)
    , mGridCols(gridCols)
    , mEdgeThreshold(edgeThreshold)
{
    vpDetectors.push_back(pDetector);
    while (vpDetectors.size() < mGridRows * mGridCols)
        vpDetectors.push_back(pDetector->clone());
}

struct ResponseComparator {
    bool operator()(const cv::KeyPoint& a, const cv::KeyPoint& b)
    {
        return abs(a.response) > abs(b.response);
    }
};

void keepStrongest(int N, vector<cv::KeyPoint>& keypoints)
{
    if ((int)keypoints.size() > N) {
        std::vector<cv::KeyPoint>::iterator nth = keypoints.begin() + N;
        std::nth_element(keypoints.begin(), nth, keypoints.end(), ResponseComparator());
        keypoints.erase(nth, keypoints.end());
    }
}

static void aggregateKeypointsPerGridCell(vector<vector<cv::KeyPoint>>& vvSubKeypoints,
    vector<cv::KeyPoint>& vKeypointsOut, cv::Size gridSize, cv::Size imageSize, int edgeThreshold)
{
    for (int i = 0; i < gridSize.height; ++i) {
        int rowstart = std::max((i * imageSize.height) / gridSize.height - edgeThreshold, 0);
        for (int j = 0; j < gridSize.width; ++j) {
            int colstart = std::max((j * imageSize.width) / gridSize.width - edgeThreshold, 0);

            vector<cv::KeyPoint>& cell_keypoints = vvSubKeypoints[j + i * gridSize.width];
            vector<cv::KeyPoint>::iterator it = cell_keypoints.begin(), end = cell_keypoints.end();
            for (; it != end; ++it) {
                it->pt.x += colstart;
                it->pt.y += rowstart;
            }
            vKeypointsOut.insert(vKeypointsOut.end(), cell_keypoints.begin(), cell_keypoints.end());
        }
    }
}

void VideoGridAdaptedFeatureDetector::detect(cv::InputArray _image, vector<cv::KeyPoint>& keypoints, cv::InputArray _mask)
{
    cv::Mat image = _image.getMat();
    cv::Mat mask = _mask.getMat();
    std::vector<std::vector<cv::KeyPoint>> sub_keypoint_vectors(mGridCols * mGridRows);
    keypoints.reserve(mMaxTotalKeypoints);
    int maxPerCell = mMaxTotalKeypoints / (mGridRows * mGridCols);

#pragma omp parallel for
    for (int i = 0; i < mGridRows; ++i) {
        int rowstart = std::max((i * image.rows) / mGridRows - mEdgeThreshold, 0);
        int rowend = std::min(image.rows, ((i + 1) * image.rows) / mGridRows + mEdgeThreshold);
        cv::Range row_range(rowstart, rowend);

#pragma omp parallel for
        for (int j = 0; j < mGridCols; ++j) {
            int colstart = std::max((j * image.cols) / mGridCols - mEdgeThreshold, 0);
            int colend = std::min(image.cols, ((j + 1) * image.cols) / mGridCols + mEdgeThreshold);
            cv::Range col_range(colstart, colend);
            cv::Mat sub_image = image(row_range, col_range);
            cv::Mat sub_mask;
            if (!mask.empty()) {
                sub_mask = mask(row_range, col_range);
            }

            std::vector<cv::KeyPoint>& sub_keypoints = sub_keypoint_vectors[j + i * mGridCols];
            vpDetectors[j + i * mGridCols]->detect(sub_image, sub_keypoints, sub_mask);
            keepStrongest(maxPerCell, sub_keypoints);
        }
    }

    aggregateKeypointsPerGridCell(sub_keypoint_vectors, keypoints, cv::Size(mGridCols, mGridRows), image.size(), mEdgeThreshold);
}

cv::Ptr<StatefulFeatureDetector> VideoGridAdaptedFeatureDetector::clone() const
{
    StatefulFeatureDetector* fd = new VideoGridAdaptedFeatureDetector(vpDetectors[0]->clone(),
        mMaxTotalKeypoints,
        mGridRows, mGridCols,
        mEdgeThreshold);
    cv::Ptr<StatefulFeatureDetector> cloned_obj(fd);
    return cloned_obj;
}
