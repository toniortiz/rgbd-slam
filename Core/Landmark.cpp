#include "Landmark.h"
#include "Features/Matcher.h"
#include "Frame.h"

using namespace std;

int Landmark::nNextId = 0;
mutex Landmark::mGlobalMutex;

Landmark::Landmark(const cv::Mat& Pos, Frame::Ptr frame, const size_t& idxF)
    : nObs(0)
{
    Pos.copyTo(mWorldPos);
    frame->mDescriptors.row(idxF).copyTo(mDescriptor);

    mnId = nNextId++;
}

Landmark::Landmark()
    : nObs(0)
{
}

Landmark::Ptr Landmark::create()
{
    Landmark::Ptr pLM(new Landmark());
    return pLM;
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

void Landmark::addObservation(KeyFramePtr pKF, size_t obsId)
{
    unique_lock<mutex> lock(mMutexFeatures);
    if (mObservations.count(pKF))
        return;
    mObservations[pKF] = obsId;
    nObs++;
}

void Landmark::eraseObservation(KeyFramePtr pKF)
{
    unique_lock<mutex> lock(mMutexFeatures);
    if (mObservations.count(pKF)) {
        nObs--;

        mObservations.erase(pKF);
    }
}

map<Landmark::KeyFramePtr, size_t> Landmark::getObservations()
{
    unique_lock<mutex> lock(mMutexFeatures);
    return mObservations;
}

list<pair<Landmark::KeyFramePtr, size_t>> Landmark::getObservationsList()
{
    unique_lock<mutex> lock(mMutexFeatures);
    list<pair<KeyFramePtr, size_t>> r;
    r.insert(r.end(), mObservations.begin(), mObservations.end());
    return r;
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

void Landmark::setDescriptor(cv::Mat& desc)
{
    unique_lock<mutex> lock(mMutexFeatures);
    desc.copyTo(mDescriptor);
}

cv::Mat Landmark::getDescriptor()
{
    unique_lock<mutex> lock(mMutexFeatures);
    return mDescriptor.clone();
}

int Landmark::id()
{
    unique_lock<mutex> lock(mMutexFeatures);
    return mnId;
}

int Landmark::obs()
{
    unique_lock<mutex> lock(mMutexFeatures);
    return nObs;
}

int Landmark::getIndexInKeyFrame(KeyFramePtr pKF)
{
    unique_lock<mutex> lock(mMutexFeatures);
    if (mObservations.count(pKF))
        return int(mObservations[pKF]);
    else
        return -1;
}

bool Landmark::isInKeyFrame(KeyFramePtr pKF)
{
    unique_lock<mutex> lock(mMutexFeatures);
    return (mObservations.count(pKF));
}

bool Landmark::operator==(const Landmark& pLM) const
{
    return mnId == pLM.mnId;
}
