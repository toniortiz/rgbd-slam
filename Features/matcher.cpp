#include "matcher.h"
#include "Core/frame.h"
#include "Core/landmark.h"
#include "Core/rgbdcamera.h"
#include "extractor.h"

using namespace std;

const int Matcher::TH_HIGH = 100;
const int Matcher::TH_LOW = 50;
const int Matcher::HISTO_LENGTH = 30;

Matcher::Matcher(float nnratio)
    : mfNNratio(nnratio)
{
    mpMatcher = cv::BFMatcher::create(Extractor::mNorm);
}

void Matcher::draw(Frame& ref, const Frame& cur, const vector<cv::DMatch>& m12, const int delay)
{
    try {
        cv::Mat out;
        const vector<cv::KeyPoint> vKP1 = ref.mvKeys;
        const vector<cv::KeyPoint> vKP2 = cur.mvKeys;
        cv::drawMatches(ref.mImGray, vKP1, cur.mImGray, vKP2, m12, out, cv::Scalar::all(-1),
            cv::Scalar::all(-1), vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

        cv::imshow("Matches", out);
        cv::waitKey(delay);
    } catch (cv::Exception& e) {
        cerr << e.what() << endl;
    }
}

int Matcher::projectionMatch(Frame::Ptr F1, Frame::Ptr F2, std::vector<cv::DMatch>& vMatches12, const float th)
{
    float fx = F1->mpCamera->fx();
    float fy = F1->mpCamera->fy();
    float cx = F1->mpCamera->cx();
    float cy = F1->mpCamera->cy();

    cv::Mat Rcw = F2->getRotation();
    cv::Mat tcw = F2->getTranslation();
    cv::Mat twc = F2->getCameraCenter();

    cv::Mat Rlw = F1->getRotation();
    cv::Mat tlw = F1->getTranslation();

    cv::Mat tlc = Rlw * twc + tlw;

    set<size_t> trainIdxs;

    for (size_t i1 = 0; i1 < F1->N; ++i1) {
        Landmark::Ptr pLM = F1->getLandmark(i1);

        if (!pLM)
            continue;
        if (F1->isOutlier(i1))
            continue;

        cv::Mat xw = pLM->getWorldPos();
        cv::Mat x3Dc = Rcw * xw + tcw;

        const float xc = x3Dc.at<float>(0);
        const float yc = x3Dc.at<float>(1);
        const float invzc = 1.0f / x3Dc.at<float>(2);

        if (invzc < 0)
            continue;

        float u = fx * xc * invzc + cx;
        float v = fy * yc * invzc + cy;

        if (u < F2->mnMinX || u > F2->mnMaxX)
            continue;
        if (v < F2->mnMinY || v > F2->mnMaxY)
            continue;

        vector<size_t> vIndices2 = F2->getFeaturesInArea(u, v, th);
        if (vIndices2.empty())
            continue;

        const cv::Mat d1 = pLM->getDescriptor();

        double bestDist = numeric_limits<double>::max();
        size_t bestIdx2;

        for (const auto& i2 : vIndices2) {
            if (!F2->isValidObs(i2))
                continue;
            if (trainIdxs.count(i2))
                continue;

            cv::Mat d2 = F2->mDescriptors.row(i2);

            double dist = descriptorDistance(d1, d2);
            if (dist < bestDist) {
                bestDist = dist;
                bestIdx2 = i2;
            }
        }

        if (bestDist <= TH_HIGH) {
            cv::DMatch m(i1, bestIdx2, bestDist);
            vMatches12.push_back(m);

            trainIdxs.insert(bestIdx2);
        }
    }

    return int(vMatches12.size());
}

int Matcher::match(Frame::Ptr ref, Frame::Ptr cur, vector<cv::DMatch>& vMatches12, const bool discardOutliers)
{
    vMatches12.clear();

    vector<vector<cv::DMatch>> matchesKnn;
    set<int> trainIdxs;

    mpMatcher->knnMatch(ref->mDescriptors, cur->mDescriptors, matchesKnn, 2);

    for (size_t i = 0; i < matchesKnn.size(); i++) {
        cv::DMatch& m1 = matchesKnn[i][0];
        cv::DMatch& m2 = matchesKnn[i][1];

        if (m1.distance < mfNNratio * m2.distance) {
            int i1 = m1.queryIdx;
            int i2 = m1.trainIdx;
            if (trainIdxs.count(i2))
                continue;

            if (discardOutliers) {
                if (ref->isOutlier(i1))
                    continue;
            }

            if (!ref->isValidObs(i1) || !cur->isValidObs(i2))
                continue;

            trainIdxs.insert(i2);
            vMatches12.push_back(m1);
        }
    }

    return int(vMatches12.size());
}

double Matcher::descriptorDistance(const cv::Mat& a, const cv::Mat& b)
{
    return cv::norm(a, b, Extractor::mNorm);
}
