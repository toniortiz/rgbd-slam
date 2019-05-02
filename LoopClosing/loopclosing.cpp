#include "loopclosing.h"
#include "Core/frame.h"
#include "Core/map.h"
#include <iostream>

using namespace std;

LoopDetector::LoopDetector(shared_ptr<DBoW3::Vocabulary> pVoc, Map::Ptr pMap,
    const int& interval)
    : mpVocabulary(pVoc)
    , mpMap(pMap)
    , mnInterval(interval)
{
}

LoopDetector::Ptr LoopDetector::create(shared_ptr<DBoW3::Vocabulary> pVoc, Map::Ptr pMap)
{
    return make_shared<LoopDetector>(pVoc, pMap);
}

vector<Frame::Ptr> LoopDetector::obtainCandidates(const Frame::Ptr pKF)
{
    vector<Frame::Ptr> possibleLoops;

    vector<Frame::Ptr> pKFs = mpMap->getAllKeyFrames();
    if (pKFs.empty())
        return possibleLoops;

    double minScore = numeric_limits<double>::max();
    size_t N = pKFs.size();
    for (size_t i = N - 1; i > N - 3; i--) {
        Frame::Ptr pKFi = pKFs[i];

        if (pKFi == pKF)
            continue;

        double score = mpVocabulary->score(pKFi->mBowVec, pKF->mBowVec);
        if (score < minScore)
            minScore = score;
    }

    if (minScore > 0.0) {
        for (Frame::Ptr pKFi : pKFs) {
            if (pKF == pKFi)
                continue;

            double score = mpVocabulary->score(pKF->mBowVec, pKFi->mBowVec);

            if (score > minScore && abs(pKFi->getId() - pKF->getId()) > mnInterval) {
                pKFi->mScore = score;
                possibleLoops.push_back(pKFi);
            }
        }
    }

    if (possibleLoops.size() > 5) {
        sort(possibleLoops.begin(), possibleLoops.end(), [&](const Frame::Ptr frame1, const Frame::Ptr frame2) {
            return frame1->mScore > frame2->mScore;
        });

        return vector<Frame::Ptr>(possibleLoops.begin(), possibleLoops.begin() + 5);
    } else {
        return possibleLoops;
    }
}

void LoopDetector::setVocabulary(shared_ptr<DBoW3::Vocabulary> pVoc)
{
    mpVocabulary = pVoc;
}

void LoopDetector::setMap(shared_ptr<Map> pMap)
{
    mpMap = pMap;
}

void LoopDetector::setInterval(int interval)
{
    mnInterval = interval;
}
