#ifndef LOOPCLOSING_H
#define LOOPCLOSING_H

#include "System/macros.h"
#include <DBoW3/Vocabulary.h>
#include <vector>

class Frame;
class Map;

class LoopDetector {
public:
    SMART_POINTER_TYPEDEFS(LoopDetector);

public:
    LoopDetector(std::shared_ptr<DBoW3::Vocabulary> pVoc, std::shared_ptr<Map> pMap,
        const int& interval = 10);

    static Ptr create(std::shared_ptr<DBoW3::Vocabulary> pVoc, std::shared_ptr<Map> pMap);

    std::vector<std::shared_ptr<Frame>> obtainCandidates(const std::shared_ptr<Frame> pKF);

    void setVocabulary(std::shared_ptr<DBoW3::Vocabulary> pVoc);
    void setMap(std::shared_ptr<Map> pMap);

    void setInterval(int interval);

protected:
    std::shared_ptr<DBoW3::Vocabulary> mpVocabulary;

    std::shared_ptr<Map> mpMap;

    int mnInterval;
};

#endif // LOOPCLOSING_H
