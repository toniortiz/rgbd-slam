#include "Tracking.h"
#include "Converter.h"
#include "Core/Frame.h"
#include "Core/Map.h"
#include "Core/RGBDcamera.h"
#include "Drawer/MapDrawer.h"
#include "Drawer/Viewer.h"
#include "Features/Matcher.h"
#include "PlaceRecognition/LoopDetector.h"
#include "Solver/Gicp.h"
#include "Solver/Icp.h"
#include "Solver/PnPRansac.h"
#include "Solver/PnPSolver.h"
#include "Solver/PoseGraph.h"
#include "Solver/Ransac.h"
#include "Solver/SolverSE3.h"

using namespace std;

Tracking::Tracking(shared_ptr<DBoW3::Vocabulary> pVoc, Map::Ptr pMap)
    : mpCurFrame(nullptr)
    , mState(NOT_INITIALIZED)
    , mpVoc(pVoc)
    , mpMap(pMap)
    , mnAcumInliers(0)
    , mnMeanInliers(0)
{
    mpLoopDetector = make_shared<LoopDetector>(pVoc, pMap);
    mpLoopDetector->setInterval(100);

    // Launch pose graph thread
    mpPoseGraph = make_shared<PoseGraph>(this, mpLoopDetector, pMap);

    // Launch viewer thread
    mpMapDrawer = make_shared<MapDrawer>(pMap, mpPoseGraph);
    mpViewer = make_shared<Viewer>(mpMapDrawer, pMap, this);
}

cv::Mat Tracking::track(shared_ptr<Frame> newFrame)
{
    unique_lock<mutex> lck(mMutexTrack);
    mpCurFrame = newFrame;

    if (mState == NOT_INITIALIZED) {
        initialize();
    } else {
        if (mState == OK) {
            visualOdometry();
            updateLastFrame();

            mpCurFrame->mpReferenceKF = mpLastKeyFrame;

            // Update motion model
            mVelocity = mpCurFrame->getPose() * mpRefFrame.first->getPoseInverse();

            if (needKeyFrame())
                createKeyFrame();

            mpRefFrame.second = mpRefFrame.first;
            mpRefFrame.first = mpCurFrame;

        } else if (mState == LOST) {
            recover();
            mpRefFrame.second = mpRefFrame.first;
            mpRefFrame.first = mpCurFrame;
        }
    }

    updateRelativePose();

    unique_lock<mutex> lock2(mMutexImages);
    mImObs = mpCurFrame->drawTackedPoints();

    return mpCurFrame->getPose();
}

Tracking::TrackerState Tracking::getState() const
{
    return mState;
}

void Tracking::setTime(double meanTime)
{
    mpViewer->setMeanTrackingTime(meanTime);
}

void Tracking::setCurrentPose(const cv::Mat& pose)
{
    mpMapDrawer->setCurrentCameraPose(pose);
}

void Tracking::loopCandidates(const size_t& n)
{
    mpViewer->loopCandidates(n);
}

void Tracking::initialize()
{
    mpCurFrame->setPose(cv::Mat::eye(4, 4, CV_32F));

    for (size_t i = 0; i < mpCurFrame->N; ++i) {
        if (mpCurFrame->mvKeys3Dc[i].z > 0) {
            cv::Mat Xw = mpCurFrame->unprojectWorld(i);
            Landmark::Ptr pLM(new Landmark(Xw, mpCurFrame, i));
            pLM->addObservation(mpCurFrame, i);
            pLM->setColor(mpCurFrame->mvKeysColor[i]);

            mpMap->addLandmark(pLM);
            mpCurFrame->addLandmark(pLM, i);
        }
    }

    mpRefFrame.first = mpCurFrame;
    mpRefFrame.second = mpCurFrame;

    createKeyFrame();

    mState = OK;
}

void Tracking::visualOdometry()
{
    Frame::Ptr pRefFrame = mpRefFrame.first;

    Matcher matcher(0.9f);
    vector<cv::DMatch> vMatches, vInliers;
    matcher.match(pRefFrame, mpCurFrame, vMatches);

    RansacSE3 sac(200, 10, 3.0f, 4);
    bool b = sac.compute(pRefFrame, mpCurFrame, vMatches);

    // If the motion estimation agaisnt the reference frame fails, then the motion
    // estimation is tried with the second most recent frame. This simple heuristic
    // serves to eliminate drift in situations  where the camera viewpoint does not
    // vary significantly, a technique especially useful when hovering
    if (!b) {
        cout << "Match with second ref" << endl;
        vMatches.clear();
        pRefFrame = mpRefFrame.second;
        matcher.match(pRefFrame, mpCurFrame, vMatches);

        b = sac.compute(pRefFrame, mpCurFrame, vMatches);
    }

    if (sac.rmse >= 0.8f) {
        Eigen::Matrix4f guess = sac.mT21;
        Solver::Ptr solver(new Gicp(pRefFrame, mpCurFrame, sac.mvInliers, guess));
        static_cast<Gicp&>(*solver).setMaxCorrespondenceDistance(0.07);
        static_cast<Gicp&>(*solver).setMaximumIterations(10);
        b = solver->compute(vInliers);
    }
    vInliers = sac.mvInliers;

    {
        unique_lock<mutex> lock(mMutexStatistics);
        mnInliers = vInliers.size();
        mnAcumInliers += vInliers.size();
        mnMeanInliers = mnAcumInliers / mpCurFrame->id();
    }

    if (!b)
        recover();
}

bool Tracking::correct()
{
    unique_lock<mutex> lck(mMutexTrack);

    size_t n = mpMap->keyFramesInMap();
    Frame::Ptr pKF = nullptr;
    for (size_t i = 1; i < n; ++i) {
        pKF = mpMap->getKeyFrameAt(n - i);
        if (pKF != mpCurFrame)
            break;
    }

    Matcher matcher(0.9f);
    vector<cv::DMatch> vMatches;
    matcher.match(pKF, mpCurFrame, vMatches);

    RansacSE3 icp(200, 10, 3.0f, 4);
    bool b = icp.compute(pKF, mpCurFrame, vMatches);

    if (b) {
        cout << "Correct: " << pKF->id() << "-" << mpCurFrame->id() << endl;

        mpRefFrame.second = mpRefFrame.first;
        mpRefFrame.first = mpCurFrame;
        mState = OK;
        return true;
    } else
        return false;
}

void Tracking::recover()
{
    mpCurFrame->setPose(/*mVelocity **/ mpRefFrame.first->getPose());
    mState = OK;
}

double tnorm(const cv::Mat& T)
{
    cv::Mat t = T.rowRange(0, 3).col(3);
    return cv::norm(t);
}

double rnorm(const cv::Mat& T)
{
    cv::Mat R = T.rowRange(0, 3).colRange(0, 3);
    return acos(0.5 * (R.at<float>(0, 0) + R.at<float>(1, 1) + R.at<float>(2, 2) - 1.0));
}

bool Tracking::needKeyFrame()
{
    // New keyframes are added when the accumulated motion since the previous
    // keyframe exceeds either 10° in rotation or 20 cm in translation
    static const double mint = 0.20; // m
    static const double minr = 0.1745;

    cv::Mat delta = mpCurFrame->getPoseInverse() * mpLastKeyFrame->getPose();
    bool c1 = tnorm(delta) > mint;
    bool c2 = rnorm(delta) > minr;

    return c1 | c2;
}

void Tracking::createKeyFrame()
{
    mpCurFrame->computeBoW(mpVoc);
    mpLastKeyFrame = mpCurFrame;
    mpLastKeyFrame->setKF();
    mpCurFrame->mpReferenceKF = mpLastKeyFrame;

    mpLastKeyFrame->createCloud(6);
    mpLastKeyFrame->passThroughFilter("z", 0.5, 4.0);
    mpLastKeyFrame->downsampleCloud(0.04f);
    mpLastKeyFrame->statisticalFilterCloud(50, 1.0);

    mpPoseGraph->insertKeyFrame(mpLastKeyFrame);
}

void Tracking::updateLastFrame()
{
    Frame::Ptr pRef = mpRefFrame.first->mpReferenceKF;
    cv::Mat Tlr = mRelativeFramePoses.back();
    mpRefFrame.first->setPose(Tlr * pRef->getPose());
}

void Tracking::updateRelativePose()
{
    // Store frame pose information to retrieve the complete camera trajectory afterwards.
    cv::Mat Tcr = mpCurFrame->getPose() * mpCurFrame->mpReferenceKF->getPoseInverse();
    mRelativeFramePoses.push_back(Tcr);
    mReferences.push_back(mpLastKeyFrame);
    mFrameTimes.push_back(mpCurFrame->mTimeStamp);
}

void Tracking::shutdown()
{
    mpPoseGraph->shutdown();

    cout << "\nPress any key to continue: ";
    cout.flush();
    string answer;
    cin >> answer;

    mpViewer->shutdown();
}

void Tracking::saveKeyFrameTrajectory(const string& filename)
{
    vector<Frame::Ptr> vpKFs = mpMap->getAllKeyFrames();
    sort(vpKFs.begin(), vpKFs.end(), Frame::CompareId);

    ofstream fout(filename);
    fout << fixed;

    for (Frame::Ptr pKF : vpKFs)
        fout << (*pKF);

    fout.close();

    cout << "Saved: " << filename << endl;
}

void Tracking::saveCameraTrajectory(const string& filename)
{
    vector<Frame::Ptr> vpKFs = mpMap->getAllKeyFrames();
    sort(vpKFs.begin(), vpKFs.end(), Frame::CompareId);

    cv::Mat Two = vpKFs[0]->getPoseInverse();

    ofstream f;
    f.open(filename.c_str());
    f << fixed;

    list<Frame::Ptr>::iterator lRit = mReferences.begin();
    list<double>::iterator lT = mFrameTimes.begin();
    for (list<cv::Mat>::iterator lit = mRelativeFramePoses.begin(), lend = mRelativeFramePoses.end(); lit != lend; lit++, lRit++, lT++) {
        Frame::Ptr pKF = *lRit;
        cv::Mat Trw = cv::Mat::eye(4, 4, CV_32F);

        Trw = Trw * pKF->getPose() * Two;

        cv::Mat Tcw = (*lit) * Trw;
        cv::Mat Rwc = Tcw.rowRange(0, 3).colRange(0, 3).t();
        cv::Mat twc = -Rwc * Tcw.rowRange(0, 3).col(3);

        vector<float> q = Converter::toQuaternion(Rwc);

        f << setprecision(6) << *lT << " " << setprecision(9) << twc.at<float>(0) << " " << twc.at<float>(1) << " " << twc.at<float>(2)
          << " " << q[0] << " " << q[1] << " " << q[2] << " " << q[3] << endl;
    }
    f.close();

    cout << "Saved: " << filename << endl;
}

int Tracking::getMeanInliers()
{
    unique_lock<mutex> lock(mMutexStatistics);
    return mnMeanInliers;
}

int Tracking::getCurrentInliers()
{
    unique_lock<mutex> lock(mMutexStatistics);
    return mnInliers;
}

cv::Mat Tracking::getTrackedPointsImage()
{
    unique_lock<mutex> lock(mMutexImages);
    return mImObs.clone();
}
