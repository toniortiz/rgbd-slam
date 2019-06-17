#include "landmark.h"
#include "Features/matcher.h"
#include "frame.h"
#include "map.h"

using namespace std;

int Landmark::nNextId = 0;
mutex Landmark::mGlobalMutex;

Landmark::Landmark(const cv::Mat& Pos, Map::Ptr pMap, Frame::Ptr frame, const size_t& idxF)
    : nObs(0)
    , mnRefKFid(frame->getId())
    , mpMap(pMap)
{
    Pos.copyTo(mWorldPos);
    cv::Mat Ow = frame->getCameraCenter();
    mNormalVector = mWorldPos - Ow;
    mNormalVector = mNormalVector / cv::norm(mNormalVector);

    frame->mDescriptors.row(idxF).copyTo(mDescriptor);

    mnId = nNextId++;
}

void Landmark::setWorldPos(const cv::Mat& Pos)
{
    unique_lock<mutex> lock2(mGlobalMutex);
    unique_lock<mutex> lock(mMutexPos);
    Pos.copyTo(mWorldPos);
}

cv::Mat Landmark::getWorldPos()
{
    unique_lock<mutex> lock(mMutexPos);
    return mWorldPos.clone();
}

cv::Mat Landmark::getNormal()
{
    unique_lock<mutex> lock(mMutexPos);
    return mNormalVector.clone();
}

Landmark::KeyFrameID Landmark::getReferenceKeyFrameId()
{
    unique_lock<mutex> lock(mMutexFeatures);
    return mnRefKFid;
}

void Landmark::addObservation(KeyFrameID id, size_t obsId)
{
    unique_lock<mutex> lock(mMutexFeatures);
    if (mObservations.count(id))
        return;
    mObservations[id] = obsId;
    nObs++;
}

void Landmark::eraseObservation(KeyFrameID id)
{
    unique_lock<mutex> lock(mMutexFeatures);
    if (mObservations.count(id)) {
        nObs--;

        mObservations.erase(id);

        if (mnRefKFid == id)
            mnRefKFid = mObservations.begin()->first;
    }
}

map<Landmark::KeyFrameID, size_t> Landmark::getObservations()
{
    unique_lock<mutex> lock(mMutexFeatures);
    return mObservations;
}

list<pair<Landmark::KeyFrameID, size_t>> Landmark::getObservationsList()
{
    unique_lock<mutex> lock(mMutexFeatures);
    list<pair<KeyFrameID, size_t>> r;
    r.insert(r.end(), mObservations.begin(), mObservations.end());
    return r;
}

int Landmark::observations()
{
    unique_lock<mutex> lock(mMutexFeatures);
    return nObs;
}

cv::Vec3b Landmark::getColor()
{
    unique_lock<mutex> lock(mMutexFeatures);
    return mColor;
}

void Landmark::setColor(const cv::Vec3b& color)
{
    unique_lock<mutex> lock(mMutexFeatures);
    mColor = color;
}

cv::Mat Landmark::getDescriptor()
{
    unique_lock<mutex> lock(mMutexFeatures);
    return mDescriptor.clone();
}

int Landmark::getIndexInKeyFrame(KeyFrameID id)
{
    unique_lock<mutex> lock(mMutexFeatures);
    if (mObservations.count(id))
        return int(mObservations[id]);
    else
        return -1;
}

bool Landmark::isInKeyFrame(KeyFrameID id)
{
    unique_lock<mutex> lock(mMutexFeatures);
    return (mObservations.count(id));
}

void Landmark::updateNormalAndDepth()
{
    map<KeyFrameID, size_t> observations;
    KeyFrameID refKFid;
    cv::Mat Pos;
    {
        unique_lock<mutex> lock1(mMutexFeatures);
        unique_lock<mutex> lock2(mMutexPos);

        observations = mObservations;
        refKFid = mnRefKFid;
        Pos = mWorldPos.clone();
    }

    if (observations.empty())
        return;

    cv::Mat normal = cv::Mat::zeros(3, 1, CV_32F);
    int n = 0;
    for (auto& pr : observations) {
        Frame::Ptr pKF = mpMap->getKeyFrame(pr.first);
        cv::Mat Owi = pKF->getCameraCenter();
        cv::Mat normali = mWorldPos - Owi;
        normal = normal + normali / cv::norm(normali);
        n++;
    }

    Frame::Ptr pRefKFid = mpMap->getKeyFrame(refKFid);
    cv::Mat PC = Pos - pRefKFid->getCameraCenter();
    const float dist = cv::norm(PC);

    {
        unique_lock<mutex> lock3(mMutexPos);
        mNormalVector = normal / n;
    }
}

bool Landmark::operator==(const Landmark& pLM) const
{
    return mnId == pLM.mnId;
}
