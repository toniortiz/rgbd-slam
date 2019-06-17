#ifndef FRAME_H
#define FRAME_H

#include "System/macros.h"
#include <DBoW3/BowVector.h>
#include <DBoW3/FeatureVector.h>
#include <DBoW3/Vocabulary.h>
#include <g2o/types/slam3d/vertex_se3.h>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <vector>

#define FRAME_GRID_ROWS 48
#define FRAME_GRID_COLS 64

class Extractor;
class RGBDcamera;
class Map;
class Landmark;

class Frame {
public:
    SMART_POINTER_TYPEDEFS(Frame);

    typedef pcl::PointXYZRGB PointT;
    typedef pcl::PointCloud<PointT> PointCloudT;

    struct Comparator {
        bool operator()(const Frame::Ptr& F1, const Frame::Ptr& F2) const
        {
            if (F1->mnId < F2->mnId)
                return true;
            else
                return false;
        }
    };

public:
    Frame();

    Frame(const cv::Mat& imRGB, const cv::Mat& imDepth, const double& timeStamp,
        std::shared_ptr<Extractor> pExtractor, RGBDcamera* pRGBDcamera);

    ~Frame() {}

    void extractFeatures();

    void computeBoW(std::shared_ptr<DBoW3::Vocabulary> pVoc);

    void setPose(cv::Mat Tcw);
    cv::Mat getPose() const;

    // Computes rotation, translation and camera center matrices from the camera pose.
    void updatePoseMatrices();

    cv::Mat getPoseInverse();
    cv::Mat getCameraCenter();
    cv::Mat getRotationInverse();
    cv::Mat getRotation();
    cv::Mat getTranslation();

    // Compute the cell of a keypoint (return false if outside the grid)
    bool posInGrid(const cv::KeyPoint& kp, int& posX, int& posY);

    std::vector<size_t> getFeaturesInArea(const float& x, const float& y, const float& r,
        const int minLevel = -1, const int maxLevel = -1) const;

    // Backprojects a keypoint (if stereo/depth info available) into 3D world coordinates.
    cv::Mat unprojectWorld(const size_t& i);

    bool isInlier(const size_t& idx) const;
    bool isOutlier(const size_t& idx) const;
    void setInlier(const size_t& idx);
    void setOutlier(const size_t& idx);

    void drawImage(const bool& gray = true, const int& delay = 1, const bool& destroy = false, const std::string& title = "frame") const;
    void drawObservations(const int& delay = 1, const bool& destroy = false, const std::string& title = "observations");
    void drawInliers(const int& delay = 1, const bool& destroy = false, const std::string& title = "inliers");
    cv::Mat drawTackedPoints();

    bool isValidObs(const size_t& idx);

    // Print the pose using streams
    friend std::ostream& operator<<(std::ostream& out, Frame& frame);

    bool operator==(const Frame& other) const;

    void setVertex(g2o::VertexSE3* vertex);
    g2o::VertexSE3* getVertex();
    void correctPose();
    void fixVertex(bool fix);

    void addLandmark(std::shared_ptr<Landmark> pLM, const size_t& i);
    std::shared_ptr<Landmark> getLandmark(const size_t& i);

    int getId();

    // PointCloud functions
    void createCloud();
    bool isValidCloud();
    void downsampleCloud(float leaf);
    void statisticalFilterCloud(int k, double stddev);
    void passThroughFilter(const std::string& field, float ll, float ul, const bool negate = false);
    void updateSensor();
    void setOrientation(Eigen::Quaternionf& q);
    void setOrigin(Eigen::Vector4f& t);
    Eigen::Quaternionf getOrientation();
    Eigen::Vector4f getOrigin();
    PointCloudT::Ptr unprojectWorldCloud();

public:
    cv::Mat mImColor;
    cv::Mat mImGray;
    cv::Mat mImDepth;

    std::shared_ptr<Extractor> mpExtractor;

    RGBDcamera* mpCamera;

    double mTimeStamp;

    // Number of KeyPoints.
    size_t N;

    // Vector of keypoints (original for visualization) and undistorted (actually used by the system).
    // In the RGB-D case, RGB images can be distorted.
    std::vector<cv::KeyPoint> mvKeys;
    std::vector<cv::KeyPoint> mvKeysUn;
    std::vector<cv::Point3f> mvKeys3Dc;
    std::vector<cv::Vec3b> mvKeysColor;

    // Bag of Words Vector structures.
    DBoW3::BowVector mBowVec;
    DBoW3::FeatureVector mFeatVec;
    double mScore;

    // ORB descriptor, each row associated to a keypoint.
    cv::Mat mDescriptors;

    // Flag to identify outlier associations.
    std::vector<bool> mvbOutlier;

    // Keypoints are assigned to cells in a grid to reduce matching complexity when projecting MapPoints.
    static float mfGridElementWidthInv;
    static float mfGridElementHeightInv;
    std::vector<std::size_t> mGrid[FRAME_GRID_COLS][FRAME_GRID_ROWS];

    // Undistorted Image Bounds (computed once).
    static float mnMinX;
    static float mnMaxX;
    static float mnMinY;
    static float mnMaxY;

    static bool mbInitialComputations;

    Frame::Ptr mpReferenceKF;

private:
    // Undistort keypoints given OpenCV distortion parameters.
    void undistortKeyPoints();

    // Computes image bounds for the undistorted image.
    void computeImageBounds();

    // Assign keypoints to the grid for speed up feature matching.
    void assignFeaturesToGrid();

    void uprojectCamera();

    std::vector<std::shared_ptr<Landmark>> mvpLandmarks;

    // Current and Next Frame id.
    static int nNextId;
    int mnId;

    // Camera pose.
    cv::Mat mTcw;
    cv::Mat mTwc;
    cv::Mat mRcw;
    cv::Mat mtcw;
    cv::Mat mRwc;
    cv::Mat mOw; //==mtwc

    g2o::VertexSE3* mpVertex = nullptr;

    std::mutex mMutexVertex;
    std::mutex mMutexId;
    mutable std::mutex mMutexPose;
    std::mutex mMutexFeatures;

    std::mutex mMutexCloud;
    PointCloudT::Ptr mpCloud;
};

#endif
