#include "mapdrawer.h"
#include "Core/frame.h"
#include "Core/landmark.h"
#include "Core/map.h"
#include "Solver/posegraph.h"
#include "System/converter.h"
#include "octomapdrawer.h"
#include <g2o/core/hyper_graph.h>

using namespace std;

MapDrawer::MapDrawer(Map::Ptr pMap, PoseGraph::Ptr pGraph)
    : mpMap(pMap)
    , mpGraph(pGraph)
    , mLastBigChange(0)
{
    mKeyFrameSize = 0.03f;
    mKeyFrameLineWidth = 1.0f;
    mGraphLineWidth = 0.9f;
    mCameraSize = 0.05f;
    mPointSize = 2.0f;
    mCameraLineWidth = 3.0f;

    mpOctomapDrawer = make_shared<OctomapDrawer>();
}

void MapDrawer::drawLandmarks()
{
    vector<Landmark::Ptr> vpLMs = mpMap->getAllLandmarks();
    if (vpLMs.empty())
        return;

    glPointSize(mPointSize);
    glBegin(GL_POINTS);
    glColor3f(1.0, 0.0, 0.0);

    for (Landmark::Ptr pLM : vpLMs) {
        cv::Mat pw = pLM->getWorldPos();
        cv::Vec3b bgr = pLM->getColor();

        glColor3f(bgr[2] / 255.0f, bgr[1] / 255.0f, bgr[0] / 255.0f);
        glVertex3f(pw.at<float>(0), pw.at<float>(1), pw.at<float>(2));
    }

    glEnd();
}

void MapDrawer::drawOctomap()
{
    vector<Frame::Ptr> vpKFs = mpMap->getAllKeyFrames();
    if (vpKFs.empty())
        return;

    // No big change
    if (mLastBigChange == mpMap->getLastBigChangeIdx()) {
        for (Frame::Ptr& pKF : vpKFs) {
            if (pKF->isValidCloud())
                mpOctomapDrawer->insertCloud(pKF);
        }
    } else {
        mLastBigChange = mpMap->getLastBigChangeIdx();
        mpOctomapDrawer->reset();

        for (Frame::Ptr& pKF : vpKFs) {
            if (pKF->isValidCloud())
                mpOctomapDrawer->insertCloud(pKF);
        }
    }

    mpOctomapDrawer->render();
}

void MapDrawer::drawPoseGraph(const bool bDrawVertices, const bool bDrawEdges)
{
    const float& w = mKeyFrameSize;
    const float h = w * 0.75f;
    const float z = w * 0.6f;

    /*
    vector<Frame::Ptr> vpKFs = mpMap->getAllKeyFrames();
    if (vpKFs.empty())
        return;

    Frame::Ptr oldKF = nullptr;

    for (Frame::Ptr pKF : vpKFs) {
        cv::Mat Twc = pKF->getPoseInverse().t();

        if (bDrawVertices) {
            glPushMatrix();

            glMultMatrixf(Twc.ptr<GLfloat>(0));

            glLineWidth(mKeyFrameLineWidth);
            glColor3f(0.0f, 0.0f, 0.0f);
            glBegin(GL_LINES);
            glVertex3f(0, 0, 0);
            glVertex3f(w, h, z);
            glVertex3f(0, 0, 0);
            glVertex3f(w, -h, z);
            glVertex3f(0, 0, 0);
            glVertex3f(-w, -h, z);
            glVertex3f(0, 0, 0);
            glVertex3f(-w, h, z);

            glVertex3f(w, h, z);
            glVertex3f(w, -h, z);

            glVertex3f(-w, h, z);
            glVertex3f(-w, -h, z);

            glVertex3f(-w, h, z);
            glVertex3f(w, h, z);

            glVertex3f(-w, -h, z);
            glVertex3f(w, -h, z);
            glEnd();

            glPopMatrix();
        }
        if (bDrawEdges) {
            if (!oldKF)
                oldKF = pKF;
            else {
                glLineWidth(mGraphLineWidth);
                glColor3f(0.0f, 0.75f, 1.0f);
                glBegin(GL_LINES);

                cv::Mat w0 = oldKF->getCameraCenter();
                cv::Mat w1 = pKF->getCameraCenter();

                glVertex3f(w0.at<float>(0), w0.at<float>(1), w0.at<float>(2));
                glVertex3f(w1.at<float>(0), w1.at<float>(1), w1.at<float>(2));
                glEnd();

                oldKF = pKF;
            }
        }
    }
*/

    vector<g2o::EdgeSE3*> vEdges = mpGraph->getEdges();
    if (vEdges.empty())
        return;

    for (g2o::EdgeSE3* edge : vEdges) {
        vector<g2o::HyperGraph::Vertex*> vertices = edge->vertices();

        for (size_t i = 0; i < vertices.size(); ++i) {
            g2o::VertexSE3* v0 = dynamic_cast<g2o::VertexSE3*>(vertices[i]);
            if (!v0)
                continue;

            Eigen::Matrix4f Ewc0 = v0->estimate().matrix().cast<float>();
            cv::Mat Twc0 = Converter::toMat<float, 4, 4>(Ewc0).t();

            if (bDrawVertices) {
                glPushMatrix();

                glMultMatrixf(Twc0.ptr<GLfloat>(0));

                glLineWidth(mKeyFrameLineWidth);
                glColor3f(0.0f, 0.0f, 0.0f);
                glBegin(GL_LINES);
                glVertex3f(0, 0, 0);
                glVertex3f(w, h, z);
                glVertex3f(0, 0, 0);
                glVertex3f(w, -h, z);
                glVertex3f(0, 0, 0);
                glVertex3f(-w, -h, z);
                glVertex3f(0, 0, 0);
                glVertex3f(-w, h, z);

                glVertex3f(w, h, z);
                glVertex3f(w, -h, z);

                glVertex3f(-w, h, z);
                glVertex3f(-w, -h, z);

                glVertex3f(-w, h, z);
                glVertex3f(w, h, z);

                glVertex3f(-w, -h, z);
                glVertex3f(w, -h, z);
                glEnd();

                glPopMatrix();
            }

            if (bDrawEdges) {
                if (i == vertices.size() - 1)
                    continue;

                g2o::VertexSE3* v1 = dynamic_cast<g2o::VertexSE3*>(vertices.at(i + 1));
                if (!v1)
                    continue;

                Eigen::Matrix4f Ewc1 = v1->estimate().matrix().cast<float>();
                cv::Mat Twc1 = Converter::toMat<float, 4, 4>(Ewc1);

                glLineWidth(mGraphLineWidth);
                glColor3f(/*0.0f, 0.75f, 1.0f*/ 1.0f, 0.0f, 0.0f);
                glBegin(GL_LINES);

                Twc0 = Twc0.t();
                cv::Mat w0 = Twc0.rowRange(0, 3).col(3);
                cv::Mat w1 = Twc1.rowRange(0, 3).col(3);

                glVertex3f(w0.at<float>(0), w0.at<float>(1), w0.at<float>(2));
                glVertex3f(w1.at<float>(0), w1.at<float>(1), w1.at<float>(2));
                glEnd();
            }
        }
    }
}

void MapDrawer::drawCurrentCamera(pangolin::OpenGlMatrix& Twc)
{
    const float& w = mCameraSize;
    const float h = w * 0.75f;
    const float z = w * 0.6f;

    glPushMatrix();

#ifdef HAVE_GLES
    glMultMatrixf(Twc.m);
#else
    glMultMatrixd(Twc.m);
#endif

    glLineWidth(mCameraLineWidth);
    glColor3f(0.0f, 0.0f, 1.0f);
    glBegin(GL_LINES);
    glVertex3f(0, 0, 0);
    glVertex3f(w, h, z);
    glVertex3f(0, 0, 0);
    glVertex3f(w, -h, z);
    glVertex3f(0, 0, 0);
    glVertex3f(-w, -h, z);
    glVertex3f(0, 0, 0);
    glVertex3f(-w, h, z);

    glVertex3f(w, h, z);
    glVertex3f(w, -h, z);

    glVertex3f(-w, h, z);
    glVertex3f(-w, -h, z);

    glVertex3f(-w, h, z);
    glVertex3f(w, h, z);

    glVertex3f(-w, -h, z);
    glVertex3f(w, -h, z);
    glEnd();

    glPopMatrix();
}

void MapDrawer::setCurrentCameraPose(const cv::Mat& Tcw)
{
    unique_lock<mutex> lock(mMutexCamera);
    mCameraPose = Tcw.clone();
}

void MapDrawer::getCurrentOpenGLCameraMatrix(pangolin::OpenGlMatrix& M)
{
    if (!mCameraPose.empty()) {
        cv::Mat Rwc(3, 3, CV_32F);
        cv::Mat twc(3, 1, CV_32F);
        {
            unique_lock<mutex> lock(mMutexCamera);
            Rwc = mCameraPose.rowRange(0, 3).colRange(0, 3).t();
            twc = -Rwc * mCameraPose.rowRange(0, 3).col(3);
        }

        M.m[0] = Rwc.at<float>(0, 0);
        M.m[1] = Rwc.at<float>(1, 0);
        M.m[2] = Rwc.at<float>(2, 0);
        M.m[3] = 0.0;

        M.m[4] = Rwc.at<float>(0, 1);
        M.m[5] = Rwc.at<float>(1, 1);
        M.m[6] = Rwc.at<float>(2, 1);
        M.m[7] = 0.0;

        M.m[8] = Rwc.at<float>(0, 2);
        M.m[9] = Rwc.at<float>(1, 2);
        M.m[10] = Rwc.at<float>(2, 2);
        M.m[11] = 0.0;

        M.m[12] = twc.at<float>(0);
        M.m[13] = twc.at<float>(1);
        M.m[14] = twc.at<float>(2);
        M.m[15] = 1.0;
    } else
        M.SetIdentity();
}
