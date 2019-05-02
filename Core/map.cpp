#include "map.h"

using namespace std;

Map::Map()
    : mnBigChangeIdx(0)
{
}

void Map::addKeyFrame(Frame::Ptr pKF)
{
    unique_lock<mutex> lock(mMutexMap);
    mpKeyFrames.insert({ pKF->getId(), pKF });
}

void Map::addLandmark(Landmark::Ptr pLM)
{
    unique_lock<mutex> lock(mMutexMap);
    mpLandmarks.insert({ pLM->mnId, pLM });
}

void Map::eraseKeyFrame(Frame::Ptr pKF)
{
    unique_lock<mutex> lock(mMutexMap);
    mpKeyFrames.erase(pKF->getId());
}

void Map::eraseLandmark(Landmark::Ptr pLM)
{
    unique_lock<mutex> lock(mMutexMap);
    mpLandmarks.erase(pLM->mnId);
}

Frame::Ptr Map::getKeyFrame(KeyFrameID id)
{
    unique_lock<mutex> lock(mMutexMap);

    ConstKeyFrameIt it = mpKeyFrames.find(id);
    if (it != mpKeyFrames.end())
        return it->second;
    else
        return nullptr;
}

Frame::Ptr Map::getKeyFrameAt(const size_t& index)
{
    unique_lock<mutex> lock(mMutexMap);
    KeyFrameIt it = mpKeyFrames.begin();
    std::advance(it, index);
    return it->second;
}

Landmark::Ptr Map::getLandmark(Map::LandmarkID id)
{
    unique_lock<mutex> lock(mMutexMap);
    ConstLandmarkIt it = mpLandmarks.find(id);
    if (it != mpLandmarks.end())
        return it->second;
    else
        return nullptr;
}

Landmark::Ptr Map::getLandmarkAt(const size_t& index)
{
    unique_lock<mutex> lock(mMutexMap);
    LandmarkIt it = mpLandmarks.begin();
    std::advance(it, index);
    return it->second;
}

Frame::Ptr Map::getKeyFrame(Frame::Ptr pKF)
{
    unique_lock<mutex> lock(mMutexMap);
    ConstKeyFrameIt it = mpKeyFrames.find(pKF->getId());
    if (it != mpKeyFrames.end())
        return it->second;
    else
        return nullptr;
}

Landmark::Ptr Map::getLandmark(Landmark::Ptr pLM)
{
    unique_lock<mutex> lock(mMutexMap);
    ConstLandmarkIt it = mpLandmarks.find(pLM->mnId);
    if (it != mpLandmarks.end())
        return it->second;
    else
        return nullptr;
}

vector<Frame::Ptr> Map::getAllKeyFrames()
{
    unique_lock<mutex> lock(mMutexMap);

    vector<Frame::Ptr> frames;
    frames.reserve(mpKeyFrames.size());
    for (ConstKeyFrameIt it = mpKeyFrames.begin(); it != mpKeyFrames.end(); it++)
        frames.push_back(it->second);

    return frames;
}

vector<Landmark::Ptr> Map::getAllLandmarks()
{
    unique_lock<mutex> lock(mMutexMap);

    vector<Landmark::Ptr> landmarks;
    landmarks.reserve(mpLandmarks.size());
    for (ConstLandmarkIt it = mpLandmarks.begin(); it != mpLandmarks.end(); it++)
        landmarks.push_back(it->second);

    return landmarks;
}

size_t Map::keyFramesInMap()
{
    unique_lock<mutex> lock(mMutexMap);
    return mpKeyFrames.size();
}

size_t Map::landmarksInMap()
{
    unique_lock<mutex> lock(mMutexMap);
    return mpLandmarks.size();
}

void Map::clear()
{
    unique_lock<mutex> lock(mMutexMap);
    mpKeyFrames.clear();
    mpLandmarks.clear();
}

void Map::informNewBigChange()
{
    unique_lock<mutex> lock(mMutexMap);
    mnBigChangeIdx++;
}

int Map::getLastBigChangeIdx()
{
    unique_lock<mutex> lock(mMutexMap);
    return mnBigChangeIdx;
}
