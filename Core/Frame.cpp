#include "Frame.h"
#include "Features/Extractor.h"
#include "GraphNode.h"
#include "System/Converter.h"
#include "Landmark.h"
#include "RGBDcamera.h"
#include <pcl/common/transforms.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/voxel_grid.h>
#include <thread>

using namespace std;

int Frame::nNextId = 0;
bool Frame::mbInitialComputations = true;
float Frame::mnMinX, Frame::mnMinY, Frame::mnMaxX, Frame::mnMaxY;
float Frame::mfGridElementWidthInv, Frame::mfGridElementHeightInv;

ostream& operator<<(ostream& out, Frame& frame)
{
    cv::Mat R = frame.getRotationInverse();
    vector<float> q = Converter::toQuaternion(R);
    cv::Mat t = frame.getCameraCenter();

    out << setprecision(6) << frame.mTimeStamp << setprecision(7) << " " << t.at<float>(0) << " " << t.at<float>(1) << " " << t.at<float>(2)
        << " " << q[0] << " " << q[1] << " " << q[2] << " " << q[3] << endl;

    return out;
}

Frame::Frame() {}

Frame::Frame(const cv::Mat& imRGB, const cv::Mat& imDepth, const double& timeStamp,
    Extractor::Ptr pExtractor, RGBDcamera* pRGBDcamera)
    : mImColor(imRGB)
    , mpExtractor(pExtractor)
    , mpCamera(pRGBDcamera)
    , mTimeStamp(timeStamp)
    , mpCloud(nullptr)
    , mbIsKF(false)
    , mpNode(nullptr)
{
    // Frame ID
    mnId = nNextId++;

    cvtColor(imRGB, mImGray, CV_BGR2GRAY);
    imDepth.convertTo(mImDepth, CV_32F, static_cast<double>(mpCamera->mDepthMapFactor));

    // Feature extraction
    extractFeatures();

    N = mvKeys.size();
    if (mvKeys.empty())
        return;

    undistortKeyPoints();

    mvpLandmarks = vector<Landmark::Ptr>(N, nullptr);
    mvbOutlier = vector<bool>(N, false);

    if (mbInitialComputations) {
        computeImageBounds();

        mfGridElementWidthInv = static_cast<float>(FRAME_GRID_COLS) / static_cast<float>(mnMaxX - mnMinX);
        mfGridElementHeightInv = static_cast<float>(FRAME_GRID_ROWS) / static_cast<float>(mnMaxY - mnMinY);

        mbInitialComputations = false;
    }

    assignFeaturesToGrid();
    uprojectCamera();
}

void Frame::assignFeaturesToGrid()
{
    int nReserve = 0.5f * N / (FRAME_GRID_COLS * FRAME_GRID_ROWS);
    for (unsigned int i = 0; i < FRAME_GRID_COLS; i++)
        for (unsigned int j = 0; j < FRAME_GRID_ROWS; j++)
            mGrid[i][j].reserve(nReserve);

    for (size_t i = 0; i < N; i++) {
        const cv::KeyPoint& kp = mvKeysUn[i];

        int nGridPosX, nGridPosY;
        if (posInGrid(kp, nGridPosX, nGridPosY))
            mGrid[nGridPosX][nGridPosY].push_back(i);
    }
}

void Frame::uprojectCamera()
{
    mvKeys3Dc = vector<cv::Point3f>(N, cv::Point3f(0, 0, 0));
    mvKeysColor.resize(N);

    for (size_t i = 0; i < N; i++) {
        const cv::KeyPoint& kp = mvKeys[i];
        const cv::KeyPoint& kpU = mvKeysUn[i];

        const float& v = kp.pt.y;
        const float& u = kp.pt.x;

        const float z = mImDepth.at<float>(v, u);

        cv::Vec3b color = mImColor.at<cv::Vec3b>(v, u);
        mvKeysColor[i] = color;

        if (z > 0) {
            // 3D KeyPoint position in Camera coordinates
            const float u = kpU.pt.x;
            const float v = kpU.pt.y;
            const float x = (u - mpCamera->cx()) * z * mpCamera->invfx();
            const float y = (v - mpCamera->cy()) * z * mpCamera->invfy();
            mvKeys3Dc[i] = cv::Point3f(x, y, z);
        }
    }
}

void Frame::extractFeatures()
{
    mpExtractor->detectAndCompute(mImGray, cv::Mat(), mvKeys, mDescriptors);
}

void Frame::setPose(cv::Mat Tcw)
{
    unique_lock<mutex> lock(mMutexPose);
    mTcw = Tcw.clone();
    updatePoseMatrices();
}

cv::Mat Frame::getPose() const
{
    unique_lock<mutex> lock(mMutexPose);
    return mTcw.clone();
}

void Frame::updatePoseMatrices()
{
    mRcw = mTcw.rowRange(0, 3).colRange(0, 3);
    mRwc = mRcw.t();
    mtcw = mTcw.rowRange(0, 3).col(3);
    mOw = -mRcw.t() * mtcw;

    mTwc = cv::Mat::eye(4, 4, CV_32F);
    mRwc.copyTo(mTwc.rowRange(0, 3).colRange(0, 3));
    mOw.copyTo(mTwc.rowRange(0, 3).col(3));
}

cv::Mat Frame::getPoseInverse()
{
    unique_lock<mutex> lock(mMutexPose);
    return mTwc.clone();
}

cv::Mat Frame::getCameraCenter()
{
    unique_lock<mutex> lock(mMutexPose);
    return mOw.clone();
}

cv::Mat Frame::getRotationInverse()
{
    unique_lock<mutex> lock(mMutexPose);
    return mRwc.clone();
}

cv::Mat Frame::getRotation()
{
    unique_lock<mutex> lock(mMutexPose);
    return mRcw.clone();
}

cv::Mat Frame::getTranslation()
{
    unique_lock<mutex> lock(mMutexPose);
    return mtcw.clone();
}

vector<size_t> Frame::getFeaturesInArea(const float& x, const float& y, const float& r,
    const int minLevel, const int maxLevel) const
{
    vector<size_t> vIndices;
    vIndices.reserve(N);

    const int nMinCellX = max(0, (int)floor((x - mnMinX - r) * mfGridElementWidthInv));
    if (nMinCellX >= FRAME_GRID_COLS)
        return vIndices;

    const int nMaxCellX = min((int)FRAME_GRID_COLS - 1, (int)ceil((x - mnMinX + r) * mfGridElementWidthInv));
    if (nMaxCellX < 0)
        return vIndices;

    const int nMinCellY = max(0, (int)floor((y - mnMinY - r) * mfGridElementHeightInv));
    if (nMinCellY >= FRAME_GRID_ROWS)
        return vIndices;

    const int nMaxCellY = min((int)FRAME_GRID_ROWS - 1, (int)ceil((y - mnMinY + r) * mfGridElementHeightInv));
    if (nMaxCellY < 0)
        return vIndices;

    const bool bCheckLevels = (minLevel > 0) || (maxLevel >= 0);

    for (int ix = nMinCellX; ix <= nMaxCellX; ix++) {
        for (int iy = nMinCellY; iy <= nMaxCellY; iy++) {
            const vector<size_t> vCell = mGrid[ix][iy];
            if (vCell.empty())
                continue;

            for (size_t j = 0, jend = vCell.size(); j < jend; j++) {
                const cv::KeyPoint& kpUn = mvKeysUn[vCell[j]];
                if (bCheckLevels) {
                    if (kpUn.octave < minLevel)
                        continue;
                    if (maxLevel >= 0)
                        if (kpUn.octave > maxLevel)
                            continue;
                }

                const float distx = kpUn.pt.x - x;
                const float disty = kpUn.pt.y - y;

                if (fabs(distx) < r && fabs(disty) < r)
                    vIndices.push_back(vCell[j]);
            }
        }
    }

    return vIndices;
}

bool Frame::posInGrid(const cv::KeyPoint& kp, int& posX, int& posY)
{
    posX = round((kp.pt.x - mnMinX) * mfGridElementWidthInv);
    posY = round((kp.pt.y - mnMinY) * mfGridElementHeightInv);

    // Keypoint's coordinates are undistorted, which could cause to go out of the image
    if (posX < 0 || posX >= FRAME_GRID_COLS || posY < 0 || posY >= FRAME_GRID_ROWS)
        return false;

    return true;
}

void Frame::computeBoW(shared_ptr<DBoW3::Vocabulary> pVoc)
{
    if (mBowVec.empty()) {
        vector<cv::Mat> vCurrentDesc = Converter::toDescriptorVector(mDescriptors);
        pVoc->transform(vCurrentDesc, mBowVec, mFeatVec, 4);
    }
}

void Frame::undistortKeyPoints()
{
    cv::Mat distCoef = mpCamera->distCoef();
    cv::Mat mK = mpCamera->k();

    if (distCoef.at<float>(0) == 0.0f) {
        mvKeysUn = mvKeys;
        return;
    }

    // Fill matrix with points
    cv::Mat mat(N, 2, CV_32F);
    for (size_t i = 0; i < N; i++) {
        mat.at<float>(i, 0) = mvKeys[i].pt.x;
        mat.at<float>(i, 1) = mvKeys[i].pt.y;
    }

    // Undistort points
    mat = mat.reshape(2);
    cv::undistortPoints(mat, mat, mK, distCoef, cv::Mat(), mK);
    mat = mat.reshape(1);

    // Fill undistorted keypoint vector
    mvKeysUn.resize(N);
    for (size_t i = 0; i < N; i++) {
        cv::KeyPoint kp = mvKeys[i];
        kp.pt.x = mat.at<float>(i, 0);
        kp.pt.y = mat.at<float>(i, 1);
        mvKeysUn[i] = kp;
    }
}

void Frame::computeImageBounds()
{
    cv::Mat distCoef = mpCamera->distCoef();
    cv::Mat mK = mpCamera->k();

    if (distCoef.at<float>(0) != 0.0f) {
        cv::Mat mat(4, 2, CV_32F);
        mat.at<float>(0, 0) = 0.0;
        mat.at<float>(0, 1) = 0.0;
        mat.at<float>(1, 0) = mImGray.cols;
        mat.at<float>(1, 1) = 0.0;
        mat.at<float>(2, 0) = 0.0;
        mat.at<float>(2, 1) = mImGray.rows;
        mat.at<float>(3, 0) = mImGray.cols;
        mat.at<float>(3, 1) = mImGray.rows;

        // Undistort corners
        mat = mat.reshape(2);
        cv::undistortPoints(mat, mat, mK, distCoef, cv::Mat(), mK);
        mat = mat.reshape(1);

        mnMinX = min(mat.at<float>(0, 0), mat.at<float>(2, 0));
        mnMaxX = max(mat.at<float>(1, 0), mat.at<float>(3, 0));
        mnMinY = min(mat.at<float>(0, 1), mat.at<float>(1, 1));
        mnMaxY = max(mat.at<float>(2, 1), mat.at<float>(3, 1));

    } else {
        mnMinX = 0.0f;
        mnMaxX = mImGray.cols;
        mnMinY = 0.0f;
        mnMaxY = mImGray.rows;
    }
}

cv::Mat Frame::unprojectWorld(const size_t& i)
{
    if (mvKeys3Dc[i].z > 0) {
        const cv::Point3f& p3Dc = mvKeys3Dc[i];
        cv::Mat x3Dc = (cv::Mat_<float>(3, 1) << p3Dc.x, p3Dc.y, p3Dc.z);

        unique_lock<mutex> lock(mMutexPose);
        return mRwc * x3Dc + mOw;
    } else
        return cv::Mat();
}

bool Frame::isInlier(const size_t& idx) const
{
    return mvbOutlier[idx] == false;
}

bool Frame::isOutlier(const size_t& idx) const
{
    return mvbOutlier[idx] == true;
}

void Frame::setInlier(const size_t& idx)
{
    mvbOutlier[idx] = false;
}

void Frame::setOutlier(const size_t& idx)
{
    mvbOutlier[idx] = true;
}

void Frame::drawImage(const bool& gray, const int& delay, const bool& destroy, const string& title) const
{
    cv::imshow(title, gray ? mImGray : mImColor);
    cv::waitKey(delay);

    if (destroy)
        cv::destroyWindow(title);
}

void Frame::drawObservations(const int& delay, const bool& destroy, const string& title)
{
    if (mvKeys.empty())
        return;

    cv::Mat out;
    cv::cvtColor(mImGray, out, cv::COLOR_GRAY2BGR);

    for (size_t i = 0; i < N; ++i) {
        const cv::KeyPoint& kp = mvKeys[i];
        cv::circle(out, kp.pt, 4 * (kp.octave + 1), cv::Scalar(0, 255, 0), 1);
    }

    cv::imshow(title, out);
    cv::waitKey(delay);

    if (destroy)
        cv::destroyWindow(title);
}

void Frame::drawInliers(const int& delay, const bool& destroy, const string& title)
{
    if (mvKeys.empty())
        return;

    cv::Mat out;
    cv::cvtColor(mImGray, out, cv::COLOR_GRAY2BGR);

    for (size_t i = 0; i < N; ++i) {
        if (isInlier(i)) {
            const cv::KeyPoint& kp = mvKeys[i];
            cv::circle(out, kp.pt, 4 * (kp.octave + 1), cv::Scalar(0, 255, 0), 1);
        }
    }

    cv::imshow(title, out);
    cv::waitKey(delay);

    if (destroy)
        cv::destroyWindow(title);
}

cv::Mat Frame::drawTackedPoints()
{
    cv::Mat out;
    cv::cvtColor(mImGray, out, cv::COLOR_GRAY2BGR);

    for (size_t i = 0; i < N; ++i) {
        if (getLandmark(i)) {
            const cv::KeyPoint& kp = mvKeys[i];
            cv::circle(out, kp.pt, 4 * (kp.octave + 1), cv::Scalar(0, 255, 0), 1);
        }
    }

    return out;
}

bool Frame::isValidObs(const size_t& idx)
{
    return mvKeys3Dc[idx].z > 0;
}

bool Frame::operator==(const Frame& other) const
{
    return mnId == other.mnId;
}

void Frame::setVertex(g2o::VertexSE3* vertex)
{
    unique_lock<mutex> lck(mMutexVertex);
    mpVertex = vertex;
}

g2o::VertexSE3* Frame::getVertex()
{
    unique_lock<mutex> lck(mMutexVertex);
    return mpVertex;
}

void Frame::correctPose()
{
    unique_lock<mutex> lck1(mMutexVertex);
    if (mpVertex) {
        // Correct Frame pose
        setPose(Converter::toMat<float, 4, 4>(mpVertex->estimate().inverse().matrix().cast<float>()));

        // Correct Landmark pose
        for (size_t i = 0; i < N; ++i) {
            Landmark::Ptr pLM = getLandmark(i);
            if (!pLM)
                continue;

            cv::Mat Xw = unprojectWorld(i);
            pLM->setWorldPos(Xw);
        }
    }
}

void Frame::fixVertex(bool fix)
{
    unique_lock<mutex> lck1(mMutexVertex);
    if (mpVertex)
        mpVertex->setFixed(fix);
}

std::vector<std::shared_ptr<Landmark>> Frame::getLandmarks()
{
    unique_lock<mutex> lock(mMutexFeatures);
    return mvpLandmarks;
}

int Frame::id()
{
    unique_lock<mutex> lock(mMutexId);
    return mnId;
}

void Frame::createCloud(int res)
{
    unique_lock<mutex> lock(mMutexCloud);

    if (mpCloud)
        return;

    mpCloud = boost::make_shared<PointCloudT>();

    for (int m = 0; m < mImDepth.rows; m += res) {
        for (int n = 0; n < mImDepth.cols; n += res) {
            const float z = mImDepth.at<float>(m, n);

            if (z <= 0)
                continue;

            PointT p;

            p.b = mImColor.data[m * mImColor.step + n * mImColor.channels() + 0]; // blue
            p.g = mImColor.data[m * mImColor.step + n * mImColor.channels() + 1]; // green
            p.r = mImColor.data[m * mImColor.step + n * mImColor.channels() + 2]; // red

            mpCamera->unproject(n, m, z, p.x, p.y, p.z);

            mpCloud->points.push_back(p);
        }
    }

    mpCloud->height = 1;
    mpCloud->width = mpCloud->points.size();
    mpCloud->is_dense = false;
}

bool Frame::isValidCloud()
{
    unique_lock<mutex> lock(mMutexCloud);
    if (mpCloud)
        return true;
    else
        return false;
}

void Frame::downsampleCloud(float leaf)
{
    pcl::VoxelGrid<PointT> voxel;
    voxel.setLeafSize(leaf, leaf, leaf);

    unique_lock<mutex> lock(mMutexCloud);
    voxel.setInputCloud(mpCloud);
    voxel.filter(*mpCloud);
}

void Frame::statisticalFilterCloud(int k, double stddev)
{
    pcl::StatisticalOutlierRemoval<PointT> sor;

    unique_lock<mutex> lock(mMutexCloud);
    sor.setInputCloud(mpCloud);
    sor.setMeanK(k);
    sor.setStddevMulThresh(stddev);
    sor.filter(*mpCloud);
}

void Frame::passThroughFilter(const string& field, float ll, float ul, const bool negate)
{
    unique_lock<mutex> lock(mMutexCloud);
    pcl::PassThrough<PointT> pass;
    pass.setInputCloud(mpCloud);
    pass.setFilterFieldName(field);
    pass.setFilterLimits(ll, ul);
    if (negate)
        pass.setFilterLimitsNegative(true);

    pass.filter(*mpCloud);
}

void Frame::updateSensor()
{
    cv::Mat Ow = getCameraCenter();
    cv::Mat Rwc = getRotationInverse();

    Eigen::Matrix3f R = Converter::toMatrix3d(Rwc).cast<float>();
    Eigen::Quaternionf q(R);
    Eigen::Vector4f t = Eigen::Vector4f::Ones();
    t.head<3>() = Converter::toVector3d(Ow).cast<float>();

    unique_lock<mutex> lock(mMutexCloud);
    mpCloud->sensor_orientation_ = q;
    mpCloud->sensor_origin_ = t;
}

void Frame::setOrientation(Eigen::Quaternionf& q)
{
    unique_lock<mutex> lock(mMutexCloud);
    if (!mpCloud)
        return;

    mpCloud->sensor_orientation_ = q;
}

void Frame::setOrigin(Eigen::Vector4f& t)
{
    unique_lock<mutex> lock(mMutexCloud);
    if (!mpCloud)
        return;

    mpCloud->sensor_origin_ = t;
}

Eigen::Quaternionf Frame::getOrientation()
{
    unique_lock<mutex> lock(mMutexCloud);
    return mpCloud->sensor_orientation_;
}

Eigen::Vector4f Frame::getOrigin()
{
    unique_lock<mutex> lock(mMutexCloud);
    return mpCloud->sensor_origin_;
}

Frame::PointCloudT::Ptr Frame::unprojectWorldCloud()
{
    unique_lock<mutex> lock(mMutexCloud);
    Eigen::Quaternionf q = mpCloud->sensor_orientation_;
    Eigen::Vector4f t = mpCloud->sensor_origin_;

    Eigen::Isometry3f Twc(q);
    Twc.pretranslate(t.head<3>());

    PointCloudT::Ptr pWorldCloud(new PointCloudT);
    pcl::transformPointCloud(*mpCloud, *pWorldCloud, Twc.matrix());
    return pWorldCloud;
}

void Frame::setKF()
{
    unique_lock<mutex> lock(mMutexId);
    mbIsKF = true;
}

bool Frame::isKF()
{
    unique_lock<mutex> lock(mMutexId);
    return mbIsKF;
}

void Frame::addConnection(Frame::Ptr pFrame)
{
    unique_lock<mutex> lock(mMutexConnections);
    if (!mspConnectedKFs.count(pFrame))
        mspConnectedKFs.insert(pFrame);
}

set<Frame::Ptr> Frame::getConnectedKFs()
{
    unique_lock<mutex> lock(mMutexConnections);
    return mspConnectedKFs;
}

bool Frame::hasConnection(Frame::Ptr pFrame)
{
    unique_lock<mutex> lock(mMutexConnections);
    return mspConnectedKFs.count(pFrame);
}

void Frame::addLandmark(Landmark::Ptr pLM, const size_t& i)
{
    unique_lock<mutex> lock(mMutexFeatures);
    mvpLandmarks[i] = pLM;
}

Landmark::Ptr Frame::getLandmark(const size_t& i)
{
    unique_lock<mutex> lock(mMutexFeatures);
    return mvpLandmarks[i];
}
