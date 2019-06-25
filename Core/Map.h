#ifndef MAP_H
#define MAP_H

#include "Frame.h"
#include "Landmark.h"
#include "System/Macros.h"
#include <set>
#include <vector>

class Map {
public:
    SMART_POINTER_TYPEDEFS(Map);

    typedef int KeyFrameID;
    typedef int LandmarkID;

    typedef std::map<KeyFrameID, Frame::Ptr> KeyFrameMap;
    typedef KeyFrameMap::iterator KeyFrameIt;
    typedef KeyFrameMap::const_iterator ConstKeyFrameIt;

    typedef std::map<LandmarkID, Landmark::Ptr> LandmarkMap;
    typedef LandmarkMap::iterator LandmarkIt;
    typedef LandmarkMap::const_iterator ConstLandmarkIt;

public:
    Map();

    void addKeyFrame(Frame::Ptr pKF);
    void addLandmark(Landmark::Ptr pLM);
    void eraseKeyFrame(Frame::Ptr pKF);
    void eraseLandmark(Landmark::Ptr pLM);

    Frame::Ptr getKeyFrame(KeyFrameID id);
    Frame::Ptr getKeyFrameAt(const size_t& index);
    Landmark::Ptr getLandmark(LandmarkID id);
    Landmark::Ptr getLandmarkAt(const size_t& index);
    Frame::Ptr getKeyFrame(Frame::Ptr pKF);
    Landmark::Ptr getLandmark(Landmark::Ptr pLM);

    std::vector<Frame::Ptr> getAllKeyFrames();
    std::vector<Landmark::Ptr> getAllLandmarks();

    size_t keyFramesInMap();
    size_t landmarksInMap();

    // Save or load the map from/to the file
    bool save(std::string path);
    bool load(std::string path);

    void clear();

    void informNewBigChange();
    int getLastBigChangeIdx();

protected:
    //    std::set<Frame::Ptr, Frame::Comparator> mspKeyFrames;
    KeyFrameMap mpKeyFrames;
    LandmarkMap mpLandmarks;

    // Index related to a big change in the map (graph optimization)
    int mnBigChangeIdx;

    std::mutex mMutexMap;
};

#endif // MAP_H
