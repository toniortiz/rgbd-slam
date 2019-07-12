#include "LoopDetector.h"
#include "Core/Frame.h"
#include "Core/Map.h"
#include <iostream>

using namespace std;

LoopDetector::LoopDetector(VocabularyPtr pVoc, MapPtr pMap,
    const int& interval)
    : mpVocabulary(pVoc)
    , mpMap(pMap)
    , mnInterval(interval)
{
    mvInvertedFile.resize(pVoc->size());
}

LoopDetector::Ptr LoopDetector::create(VocabularyPtr pVoc, MapPtr pMap)
{
    return make_shared<LoopDetector>(pVoc, pMap);
}

void LoopDetector::add(FramePtr& pKF)
{
    for (const auto& [wordId, wordValue] : pKF->mBowVec)
        mvInvertedFile[wordId].push_back(pKF);
}

vector<LoopDetector::FramePtr> LoopDetector::obtainCandidates(const FramePtr pKF)
{
    vector<FramePtr> possibleLoops;
    list<FramePtr> lKFsSharingWords;

    set<FramePtr> spConnectedKFs = pKF->getConnectedKFs();
    if (spConnectedKFs.empty())
        return possibleLoops;

    // Compute minimum score with connected KFs
    double minScore = numeric_limits<double>::max();
    for (FramePtr pKFi : spConnectedKFs) {
        if (pKFi == pKF)
            continue;

        double score = mpVocabulary->score(pKFi->mBowVec, pKF->mBowVec);
        if (score < minScore)
            minScore = score;
    }

    // Search all KFs that share a word with query KF
    // Discard KFs connected to the query KF
    for (const auto& [wordId, wordValue] : pKF->mBowVec) {
        list<FramePtr>& lKFs = mvInvertedFile[wordId];

        for (FramePtr pKFi : lKFs) {
            if (pKFi == pKF)
                continue;
            if (pKFi->mnSharingWord == pKF->id())
                continue;
            if (spConnectedKFs.count(pKFi))
                continue;

            pKFi->mnSharingWord = pKF->id();
            lKFsSharingWords.push_back(pKFi);
        }
    }

    if (lKFsSharingWords.empty())
        return possibleLoops;

    for (Frame::Ptr pKFi : lKFsSharingWords) {
        double score = mpVocabulary->score(pKF->mBowVec, pKFi->mBowVec);

        if (score > minScore && abs(pKFi->id() - pKF->id()) > mnInterval) {
            pKFi->mScore = score;
            possibleLoops.push_back(pKFi);
        }
    }

        if (possibleLoops.size() > 5) {
            sort(possibleLoops.begin(), possibleLoops.end(), Frame::CompareBoWscore);
            return vector<Frame::Ptr>(possibleLoops.begin(), possibleLoops.begin() + 5);
        }

    return possibleLoops;
}

void LoopDetector::setVocabulary(VocabularyPtr pVoc)
{
    mpVocabulary = pVoc;
}

void LoopDetector::setMap(MapPtr pMap)
{
    mpMap = pMap;
}

void LoopDetector::setInterval(int interval)
{
    mnInterval = interval;
}
