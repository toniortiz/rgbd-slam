#include "tracking.h"
#include "Core/frame.h"
#include "Core/map.h"
#include "Core/rgbdcamera.h"
#include "Drawer/mapdrawer.h"
#include "Features/matcher.h"
#include "LoopClosing/loopclosing.h"
#include "Solver/gicp.h"
#include "Solver/icp.h"
#include "Solver/pnpsolver.h"
#include "Solver/posegraph.h"
#include "Solver/ransac.h"
#include "Solver/ransacpnp.h"
#include "Drawer/viewer.h"
#include "Solver/ricp.h"
#include "converter.h"

using namespace std;

Tracking::Tracking(shared_ptr<DBoW3::Vocabulary> pVoc, Map::Ptr pMap)
    : mpCurFrame(nullptr)
    , mState(NOT_INITIALIZED)
    , mpVoc(pVoc)
    , mpMap(pMap)
    , mnAcumInliers(0)
    , mnMeanInliers(0)
{
    mpLoopCloser = make_shared<LoopDetector>(pVoc, pMap);
    mpLoopCloser->setInterval(100);

    // Launch pose graph thread
    mpPoseGraph = make_shared<PoseGraph>(this, mpLoopCloser, pMap);

    // Launch viewer thread
    mpMapDrawer = make_shared<MapDrawer>(pMap, mpPoseGraph);
    mpViewer = make_shared<Viewer>(mpMapDrawer, pMap, this);

    mParams.verbose = 0;
    mParams.errorVersionVO = 0;
    mParams.errorVersionMap = 0;
    mParams.inlierThresholdEuclidean = 0.04;
    mParams.inlierThresholdReprojection = 2.0;
    mParams.inlierThresholdMahalanobis = 0.0002;
    mParams.minimalInlierRatioThreshold = 0.2;
    mParams.minimalNumberOfMatches = 15;
    mParams.usedPairs = 3;
    mParams.errorVersion = Ransac::EUCLIDEAN_ERROR;
}

cv::Mat Tracking::track(shared_ptr<Frame> newFrame)
{
    unique_lock<mutex> lck(mMutexTrack);
    mpCurFrame = newFrame;

    if (mState == NOT_INITIALIZED) {
        initialize();
    } else {
        if (mState == OK) {
            trackReference();
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
            Landmark::Ptr pLM(new Landmark(Xw, mpMap, mpCurFrame, i));
            pLM->addObservation(mpCurFrame->getId(), i);
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

void Tracking::trackReference()
{
    Frame::Ptr pRefFrame = mpRefFrame.first;

    Matcher matcher(0.9f);
    vector<cv::DMatch> vMatches, vInliers;
    matcher.match(pRefFrame, mpCurFrame, vMatches);

    RIcp sac(200, 10, 3.0f, 4);
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

    if (sac.rmse > 0.7f) {
        Eigen::Matrix4f guess = sac.mT21;
        Solver::Ptr solver(new Gicp(pRefFrame, mpCurFrame, sac.mvInliers, guess));
        static_cast<Gicp&>(*solver).setMaximumIterations(8);
        b = solver->compute(vInliers);
    }
    vInliers = sac.mvInliers;

    //    if (ransac->Compute(pF1, pF2, vMatches12)) {
    //        T = ransac->GetTransformation();
    //        float rmse = ransac->GetRMSE();
    //        mStatus = true;

    //        if (rmse * 10.0f > mMiu2) {
    //            if (icp->ComputeSubset(pF1, pF2, ransac->GetMatches()))
    //                T = icp->GetTransformation();
    //        } else if (rmse * 10.0f > mMiu1) {
    //            if (icp->Compute(pF1, pF2, ransac->GetMatches(), T))
    //                T = icp->GetTransformation();
    //        }

    //    } else {
    //        T = Eigen::Matrix4f::Identity();
    //        mStatus = false;
    //    }

    {
        unique_lock<mutex> lock(mMutexStatistics);
        mnInliers = vInliers.size();
        mnAcumInliers += vInliers.size();
        mnMeanInliers = mnAcumInliers / mpCurFrame->getId();
    }

    // Propagate Landmarks
    for (auto& inlier : sac.mvInliers) {
        Landmark::Ptr pLM = pRefFrame->getLandmark(inlier.queryIdx);

        if (!pLM) {
            cv::Mat Xw = mpCurFrame->unprojectWorld(inlier.trainIdx);
            pLM = make_shared<Landmark>(Xw, mpMap, mpCurFrame, inlier.trainIdx);
            pLM->addObservation(mpCurFrame->getId(), inlier.trainIdx);
            pLM->addObservation(pRefFrame->getId(), inlier.queryIdx);
            pLM->setColor(mpCurFrame->mvKeysColor[inlier.trainIdx]);

            mpCurFrame->addLandmark(pLM, inlier.trainIdx);
            pRefFrame->addLandmark(pLM, inlier.queryIdx);
        } else {
            mpCurFrame->addLandmark(pLM, inlier.trainIdx);
            pLM->addObservation(mpCurFrame->getId(), inlier.trainIdx);
        }
    }

    if (!b)
        recover();

    /*


    Solver::Ptr ransac(new Ransac(pRefFrame, mpCurFrame, vMatches, params));
    bool b = ransac->compute(vInliers);

    if (!b) {
        cout << "Match with second ref" << endl;
        vMatches.clear();
        pRefFrame = mpRefFrame.second;
        matcher.match(pRefFrame, mpCurFrame, vMatches);

        ransac.reset(new Ransac(pRefFrame, mpCurFrame, vMatches, params));
        b = ransac->compute(vInliers);
    }

    {
        unique_lock<mutex> lock(mMutexStatistics);
        mnInliers = vInliers.size();
        mnAcumInliers += vInliers.size();
        mnMeanInliers = mnAcumInliers / mpCurFrame->getId();
    }

    if (!b)
        recover();
*/
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

    RIcp icp(200, 10, 3.0f, 4);
    bool b = icp.compute(pKF, mpCurFrame, vMatches);

    if (b) {
        cout << "Correct: " << pKF->getId() << "-" << mpCurFrame->getId() << endl;

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
    mpCurFrame->mpReferenceKF = mpLastKeyFrame;

    //    mpLastKeyFrame->createCloud();
    //    mpLastKeyFrame->passThroughFilter("z", 0.5, 4.0);
    //    mpLastKeyFrame->downsampleCloud(0.04f);
    //    mpLastKeyFrame->statisticalFilterCloud(50, 1.0);

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
