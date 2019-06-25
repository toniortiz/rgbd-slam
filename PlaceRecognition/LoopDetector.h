#ifndef LOOPCLOSING_H
#define LOOPCLOSING_H

#include "System/Macros.h"
#include <DBoW3/Vocabulary.h>
#include <list>
#include <vector>

class Frame;
class Map;

class LoopDetector {
public:
    SMART_POINTER_TYPEDEFS(LoopDetector);

    typedef std::shared_ptr<Frame> FramePtr;
    typedef std::shared_ptr<DBoW3::Vocabulary> VocabularyPtr;
    typedef std::shared_ptr<Map> MapPtr;

public:
    LoopDetector(VocabularyPtr pVoc, MapPtr pMap, const int& interval = 10);

    static Ptr create(VocabularyPtr pVoc, MapPtr pMap);

    void add(FramePtr& pKF);

    std::vector<FramePtr> obtainCandidates(const FramePtr pKF);

    void setVocabulary(VocabularyPtr pVoc);
    void setMap(MapPtr pMap);

    void setInterval(int interval);

protected:
    VocabularyPtr mpVocabulary;
    MapPtr mpMap;

    int mnInterval;
    std::vector<std::list<FramePtr>> mvInvertedFile;
};

#endif // LOOPCLOSING_H
