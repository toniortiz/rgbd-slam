#include "ransac.h"
#include "Core/frame.h"
#include "Core/rgbdcamera.h"
#include "System/converter.h"
#include "System/random.h"
#include <Eigen/Core>
#include <Eigen/Dense>
#include <iostream>

using namespace std;

Ransac::Ransac(const Frame::Ptr F1, Frame::Ptr F2, const vector<cv::DMatch>& matches, Ransac::parameters params)
    : Solver(F1, F2, matches)
{
    mpCamera = nullptr;

    mRANSACParams.verbose = params.verbose;
    mRANSACParams.errorVersion = params.errorVersion;
    mRANSACParams.usedPairs = params.usedPairs;
    mRANSACParams.inlierThresholdEuclidean = params.inlierThresholdEuclidean;
    mRANSACParams.inlierThresholdReprojection = params.inlierThresholdReprojection;
    mRANSACParams.inlierThresholdMahalanobis = params.inlierThresholdMahalanobis;
    mRANSACParams.minimalInlierRatioThreshold = params.minimalInlierRatioThreshold;
    mRANSACParams.minimalNumberOfMatches = params.minimalNumberOfMatches;

    mRANSACParams.iterationCount = computeRANSACIteration(0.20);

    if (mRANSACParams.verbose > 0) {
        cout << "RANSACParams.verbose --> " << mRANSACParams.verbose
             << endl;
        cout << "RANSACParams.usedPairs --> " << mRANSACParams.usedPairs
             << endl;
        cout << "RANSACParams.errorVersion --> "
             << mRANSACParams.errorVersion << endl;
        cout << "RANSACParams.inlierThresholdEuclidean --> "
             << mRANSACParams.inlierThresholdEuclidean << endl;
        cout << "RANSACParams.inlierThresholdMahalanobis --> "
             << mRANSACParams.inlierThresholdMahalanobis << endl;
        cout << "RANSACParams.inlierThresholdReprojection --> "
             << mRANSACParams.inlierThresholdReprojection << endl;
        cout << "RANSACParams.minimalInlierRatioThreshold --> "
             << mRANSACParams.minimalInlierRatioThreshold << endl;
    }
}

bool Ransac::compute(vector<cv::DMatch>& inliers)
{
    Eigen::Matrix4f T = estimateTransformation(mF1->mvKeys3Dc, mF2->mvKeys3Dc, mMatches, inliers);

    for (const auto& m : inliers)
        mF2->setInlier(m.trainIdx);

    mF2->setPose(Converter::toMat<float, 4, 4>(T).inv() * mF1->getPose());

    if (T.isIdentity())
        return false;
    else
        return true;
}

Eigen::Matrix4f Ransac::estimateTransformation(vector<cv::Point3f> prevFeatures,
    vector<cv::Point3f> features, vector<cv::DMatch> matches, vector<cv::DMatch>& bestInlierMatches)
{
    if (mRANSACParams.verbose > 0)
        cout << "RANSAC: original matches.size() = " << matches.size() << endl;

    // We assume identity and 0% inliers
    Eigen::Matrix4f bestTransformationModel = Eigen::Matrix4f::Identity();
    double bestInlierRatio = 0.0;

    // Efficiently remove match if eith of features from the match has an invalid depth
    matches.erase(
        remove_if(matches.begin(), matches.end(),
            [&](const cv::DMatch& m) {
                int prevId = m.queryIdx, id = m.trainIdx;
                mF2->setOutlier(id);

                if (prevFeatures[prevId].z < 0.1f || prevFeatures[prevId].z > 6.0f
                    || features[id].z < 0.1f || features[id].z > 6.0f)
                    return true;
                return false;
            }),
        matches.end());

    // The set of matches is too small to make any sense
    if ((int)matches.size() < mRANSACParams.minimalNumberOfMatches) {
        bestInlierMatches.clear();
        return Eigen::Matrix4f::Identity();
    }

    if (mRANSACParams.verbose > 0)
        cout << "RANSAC: matches.size() = " << matches.size() << endl;

    // Main iteration loop
    for (int i = 0; i < mRANSACParams.iterationCount; i++) {

        // Randomly select matches
        if (mRANSACParams.verbose > 1)
            cout << "RANSAC: randomly sampling ids of matches"
                 << endl;
        vector<cv::DMatch> randomMatches = getRandomMatches(matches);

        // Compute model based on those matches
        if (mRANSACParams.verbose > 1)
            cout << "RANSAC: computing model based on matches"
                 << endl;
        Eigen::Matrix4f transformationModel;
        bool modelComputation = computeTransformationModel(prevFeatures,
            features, randomMatches, transformationModel);

        // TODO: Nothing happens here right now
        bool correctModel = true;

        // Model is correct and feasible
        if (correctModel && modelComputation) {

            // Evaluate the model
            if (mRANSACParams.verbose > 1)
                cout << "RANSAC: evaluating the model" << endl;
            vector<cv::DMatch> modelConsistentMatches;

            // Choose proper error computation version based on provided parameters
            float inlierRatio = 0;
            if ((mRANSACParams.errorVersion == EUCLIDEAN_ERROR)
                || (mRANSACParams.errorVersion == ADAPTIVE_ERROR)) {
                inlierRatio = computeMatchInlierRatioEuclidean(prevFeatures,
                    features, matches, transformationModel,
                    modelConsistentMatches);
            } else if (mRANSACParams.errorVersion == REPROJECTION_ERROR) {
                inlierRatio = computeInlierRatioReprojection(prevFeatures,
                    features, matches, transformationModel,
                    modelConsistentMatches);
            } else if (mRANSACParams.errorVersion
                == EUCLIDEAN_AND_REPROJECTION_ERROR) {
                inlierRatio = computeInlierRatioEuclideanAndReprojection(
                    prevFeatures, features, matches, transformationModel,
                    modelConsistentMatches);
            } else if (mRANSACParams.errorVersion == MAHALANOBIS_ERROR) {
                inlierRatio = computeInlierRatioMahalanobis(prevFeatures,
                    features, matches, transformationModel,
                    modelConsistentMatches);
            } else
                cout << "RANSAC: incorrect error version" << endl;

            // Save better model
            if (mRANSACParams.verbose > 1)
                cout << "RANSAC: saving best model" << endl;
            saveBetterModel(inlierRatio, transformationModel,
                modelConsistentMatches, bestInlierRatio,
                bestTransformationModel, bestInlierMatches);

            // Print achieved result
            if (mRANSACParams.verbose > 1)
                cout << "RANSAC: best model inlier ratio : "
                     << bestInlierRatio * 100.0 << "%" << endl;
        }
    }

    // Reestimate from inliers
    computeTransformationModel(prevFeatures, features, bestInlierMatches,
        bestTransformationModel, UMEYAMA);
    vector<cv::DMatch> newBestInlierMatches;
    computeMatchInlierRatioEuclidean(prevFeatures, features, bestInlierMatches,
        bestTransformationModel, newBestInlierMatches);
    newBestInlierMatches.swap(bestInlierMatches);

    // Test the number of inliers to the preset threshold
    if (bestInlierRatio < mRANSACParams.minimalInlierRatioThreshold) {
        bestTransformationModel = Eigen::Matrix4f::Identity();
        bestInlierMatches.clear();
    }

    // Final result
    if (mRANSACParams.verbose > 0) {
        cout << "RANSAC best model : inlierRatio = "
             << bestInlierRatio * 100.0 << "%" << endl;
        cout << "RANSAC best model : " << endl
             << bestTransformationModel << endl;
    }
    return bestTransformationModel;
}

vector<cv::DMatch> Ransac::getRandomMatches(const vector<cv::DMatch> matches)
{
    const int matchesSize = (int)matches.size();

    vector<cv::DMatch> chosenMatches;
    vector<bool> validIndex(matchesSize, true);

    // Loop until we found enough matches
    while ((int)chosenMatches.size() < mRANSACParams.usedPairs) {

        // Randomly sample one match
        int sampledMatchIndex = Random::randomInt(0, matchesSize - 1);

        // Check if the match was not already chosen or is not marked as wrong
        if (validIndex[sampledMatchIndex] == true) {

            // Add sampled match
            chosenMatches.push_back(matches[sampledMatchIndex]);

            // Prevent choosing it again
            validIndex[sampledMatchIndex] = false;
        }
    }

    return chosenMatches;
}

bool Ransac::computeTransformationModel(const vector<cv::Point3f> prevFeatures,
    const vector<cv::Point3f> features,
    const vector<cv::DMatch> matches,
    Eigen::Matrix4f& transformationModel, TransfEstimationType usedType)
{
    Eigen::MatrixXf prevFeaturesMatrix(matches.size(), 3);
    Eigen::MatrixXf featuresMatrix(matches.size(), 3);

    // Create matrices
    for (vector<cv::DMatch>::size_type j = 0; j < matches.size(); j++) {
        cv::DMatch p = matches[j];

        Eigen::Vector3f pf(prevFeatures[p.queryIdx].x, prevFeatures[p.queryIdx].y, prevFeatures[p.queryIdx].z);
        Eigen::Vector3f cf(features[p.trainIdx].x, features[p.trainIdx].y, features[p.trainIdx].z);

        prevFeaturesMatrix.block<1, 3>(j, 0) = pf;
        featuresMatrix.block<1, 3>(j, 0) = cf;
    }

    // Compute transformation
    if (usedType == UMEYAMA) {
        transformationModel = Eigen::umeyama(featuresMatrix.transpose(),
            prevFeaturesMatrix.transpose(), false);
    } else if (usedType == G2O) {

    } else {
        cout << "RANSAC: unrecognized transformation estimation" << endl;
    }

    // Check if it failed
    if (isnan(transformationModel(0, 0))) {
        transformationModel = Eigen::Matrix4f::Identity();
        return false;
    }
    return true;
}

float Ransac::computeMatchInlierRatioEuclidean(const vector<cv::Point3f> prevFeatures,
    const vector<cv::Point3f> features,
    const vector<cv::DMatch> matches,
    const Eigen::Matrix4f transformationModel,
    vector<cv::DMatch>& modelConsistentMatches)
{
    // Break into rotation (R) and translation (t)
    Eigen::Matrix3f R = transformationModel.block<3, 3>(0, 0);
    Eigen::Vector3f t = transformationModel.block<3, 1>(0, 3);

    int inlierCount = 0;
    // For all matches
    for (vector<cv::DMatch>::const_iterator it = matches.begin();
         it != matches.end(); ++it) {
        // Estimate location of feature from position one after transformation
        Eigen::Vector3f cf(features[it->trainIdx].x, features[it->trainIdx].y, features[it->trainIdx].z);
        Eigen::Vector3f estimatedOldPosition = R * cf + t;

        // Compute residual error and compare it to inlier threshold
        double threshold = mRANSACParams.inlierThresholdEuclidean;
        if (mRANSACParams.errorVersion == ADAPTIVE_ERROR)
            threshold *= prevFeatures[it->queryIdx].z;

        Eigen::Vector3f pf(prevFeatures[it->queryIdx].x, prevFeatures[it->queryIdx].y, prevFeatures[it->queryIdx].z);
        if ((estimatedOldPosition - pf).norm()
            < threshold) {
            inlierCount++;
            modelConsistentMatches.push_back(*it);
        }
    }

    // Percent of correct matches
    return float(inlierCount) / float(matches.size());
}

float Ransac::computeInlierRatioMahalanobis(const vector<cv::Point3f> prevFeatures,
    const vector<cv::Point3f> features,
    const vector<cv::DMatch> matches,
    const Eigen::Matrix4f transformationModel,
    vector<cv::DMatch>& modelConsistentMatches)
{
    // Break into rotation (R) and translation (t)
    Eigen::Matrix3f R = transformationModel.block<3, 3>(0, 0);
    Eigen::Vector3f t = transformationModel.block<3, 1>(0, 3);

    int inlierCount = 0;
    // For all matches
    for (vector<cv::DMatch>::const_iterator it = matches.begin();
         it != matches.end(); ++it) {
        // Estimate location of feature from position one after transformation
        Eigen::Vector3f cf(features[it->trainIdx].x, features[it->trainIdx].y, features[it->trainIdx].z);
        Eigen::Vector3f estimatedOldPosition = R * cf + t;

        // Compute residual error and compare it to inlier threshold
        Eigen::Matrix<double, 3, 3> cov;
        //sensorModel.computeCov(prevFeatures[it->queryIdx],cov);
        if (cov.determinant() != 0) {
            Eigen::Vector3f pf(prevFeatures[it->queryIdx].x, prevFeatures[it->queryIdx].y, prevFeatures[it->queryIdx].z);
            double distMah = (estimatedOldPosition - pf).transpose() * cov.cast<float>() * (estimatedOldPosition - pf);
            /*cout << "vec: " << estimatedOldPosition.x() << " -> " << prevFeatures[it->queryIdx].x() <<"\n";
             cout << "vec: " << estimatedOldPosition.y() << " -> " << prevFeatures[it->queryIdx].y() <<"\n";
             cout << "vec: " << estimatedOldPosition.z() << " -> " << prevFeatures[it->queryIdx].z() <<"\n";
             cout << "distMah: " << distMah << " distEucl: " << (estimatedOldPosition - prevFeatures[it->queryIdx]).norm() << "\n";
             cout << "distMahThr: " << RANSACParams.inlierThresholdMahalanobis << "\n";
             getchar();*/
            if (distMah < mRANSACParams.inlierThresholdMahalanobis) {
                inlierCount++;
                modelConsistentMatches.push_back(*it);
            }
        }
    }

    // Percent of correct matches
    return float(inlierCount) / float(matches.size());
}

float Ransac::computeInlierRatioReprojection(const vector<cv::Point3f> prevFeatures,
    const vector<cv::Point3f> features,
    const vector<cv::DMatch> matches,
    const Eigen::Matrix4f transformationModel,
    vector<cv::DMatch>& modelConsistentMatches)
{

    // Break into rotation (R) and translation (t)
    Eigen::Matrix3f R = transformationModel.block<3, 3>(0, 0);
    Eigen::Vector3f t = transformationModel.block<3, 1>(0, 3);

    // Break into rotation (R) and translation (t)
    Eigen::Matrix3f Rinv = transformationModel.inverse().block<3, 3>(0, 0);
    Eigen::Vector3f tinv = transformationModel.inverse().block<3, 1>(0, 3);

    int inlierCount = 0;
    // For all matches
    for (vector<cv::DMatch>::const_iterator it = matches.begin();
         it != matches.end(); ++it) {
        // Estimate location of feature from position one after transformation
        Eigen::Vector3f cf(features[it->trainIdx].x, features[it->trainIdx].y, features[it->trainIdx].z);
        Eigen::Vector3f pf(prevFeatures[it->queryIdx].x, prevFeatures[it->queryIdx].y, prevFeatures[it->queryIdx].z);

        Eigen::Vector3f estimatedOldPosition = R * cf + t;
        Eigen::Vector3f estimatedNewPosition = Rinv * pf + tinv;

        // Now project both features
        cv::Point2f predictedNew = mpCamera->project3Dto2D(estimatedNewPosition);
        cv::Point2f realNew = mpCamera->project3Dto2D(cf);

        cv::Point2f predictedOld = mpCamera->project3Dto2D(estimatedOldPosition);
        cv::Point2f realOld = mpCamera->project3Dto2D(pf);

        // Compute residual error and compare it to inlier threshold
        double error2D[2]{ cv::norm(predictedNew - realNew), cv::norm(predictedOld - realOld) };

        // Compute residual error and compare it to inlier threshold
        if (error2D[0] < mRANSACParams.inlierThresholdReprojection
            && error2D[1] < mRANSACParams.inlierThresholdReprojection) {

            inlierCount++;
            modelConsistentMatches.push_back(*it);
        }
    }

    // Percent of correct matches
    return float(inlierCount) / float(matches.size());
}

float Ransac::computeInlierRatioEuclideanAndReprojection(const vector<cv::Point3f> prevFeatures,
    const vector<cv::Point3f> features,
    const vector<cv::DMatch> matches,
    const Eigen::Matrix4f transformationModel,
    vector<cv::DMatch>& modelConsistentMatches)
{
    // Break into rotation (R) and translation (t)
    Eigen::Matrix3f R = transformationModel.block<3, 3>(0, 0);
    Eigen::Vector3f t = transformationModel.block<3, 1>(0, 3);

    // Break into rotation (R) and translation (t)
    Eigen::Matrix3f Rinv = transformationModel.inverse().block<3, 3>(0, 0);
    Eigen::Vector3f tinv = transformationModel.inverse().block<3, 1>(0, 3);

    int inlierCount = 0;
    // For all matches
    for (vector<cv::DMatch>::const_iterator it = matches.begin();
         it != matches.end(); ++it) {
        // Estimate location of feature from position one after transformation
        Eigen::Vector3f cf(features[it->trainIdx].x, features[it->trainIdx].y, features[it->trainIdx].z);
        Eigen::Vector3f pf(prevFeatures[it->queryIdx].x, prevFeatures[it->queryIdx].y, prevFeatures[it->queryIdx].z);

        Eigen::Vector3f estimatedOldPosition = R * cf + t;
        Eigen::Vector3f estimatedNewPosition = Rinv * pf + tinv;

        // Compute error3D
        double error3D = (estimatedOldPosition - pf).norm();

        // Now project both features
        cv::Point2f predictedNew = mpCamera->project3Dto2D(estimatedNewPosition);
        cv::Point2f realNew = mpCamera->project3Dto2D(cf);

        cv::Point2f predictedOld = mpCamera->project3Dto2D(estimatedOldPosition);
        cv::Point2f realOld = mpCamera->project3Dto2D(pf);

        // Compute residual error and compare it to inlier threshold
        double error2D[2]{ cv::norm(predictedNew - realNew), cv::norm(predictedOld - realOld) };

        //cout<<"ERROR : " << error3D << " " << error2D[0] << " " << error2D[1] << endl;

        // Compute residual error and compare it to inlier threshold
        if (error3D < mRANSACParams.inlierThresholdEuclidean
            && error2D[0] < mRANSACParams.inlierThresholdReprojection
            && error2D[1] < mRANSACParams.inlierThresholdReprojection) {
            //			cout<<"3D position: " << estimatedOldPosition.transpose() << "\t" << prevFeatures[it->queryIdx].transpose() << endl;
            //			cout<<"3D position: " << estimatedNewPosition.transpose() << "\t" << features[it->trainIdx].transpose() << endl;
            //			cout<<"ERROR : " << error3D << " " << (estimatedNewPosition - features[it->trainIdx]).norm() <<" "<< error2D[0] << " " << error2D[1] << endl;

            inlierCount++;
            modelConsistentMatches.push_back(*it);
        }
    }

    // Percent of correct matches
    return float(inlierCount) / float(matches.size());
}

inline void Ransac::saveBetterModel(const double inlierRatio,
    const Eigen::Matrix4f transformationModel,
    vector<cv::DMatch> modelConsistentMatches, double& bestInlierRatio,
    Eigen::Matrix4f& bestTransformationModel,
    vector<cv::DMatch>& bestInlierMatches)
{
    if (inlierRatio > bestInlierRatio) {
        // Save better model
        bestTransformationModel = transformationModel;
        bestInlierRatio = inlierRatio;
        bestInlierMatches.swap(modelConsistentMatches);

        // Update iteration count
        mRANSACParams.iterationCount = min(
            computeRANSACIteration(
                mRANSACParams.minimalInlierRatioThreshold),
            computeRANSACIteration(bestInlierRatio));
    }
}

inline int Ransac::computeRANSACIteration(double inlierRatio,
    double successProbability, int numberOfPairs)
{
    return int((log(1 - successProbability)
        / log(1 - pow(inlierRatio, numberOfPairs))));
}
