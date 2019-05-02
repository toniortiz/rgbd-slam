#include "viewer.h"
#include "Core/map.h"
#include "System/tracking.h"
#include "mapdrawer.h"
#include <pangolin/pangolin.h>
#include <unistd.h>

using namespace std;

Viewer::Viewer(MapDrawer::Ptr pMapDrawer, Map::Ptr pMap)
    : mpMapDrawer(pMapDrawer)
    , mpMap(pMap)
    , mbFinishRequested(false)
    , mbFinished(true)
    , mbStopped(true)
    , mbStopRequested(false)
    , mMeanTrackTime(0.0)
    , mnLoopCandidates(0)
{
    float fps = 30.0f;
    if (fps < 1)
        fps = 30;
    mT = 1e3 / fps;

    mImageWidth = 640;
    mImageHeight = 480;

    mViewpointX = 0.0f;
    mViewpointY = -0.7f;
    mViewpointZ = -1.8f;
    mViewpointF = 500.0f;
}

void Viewer::run()
{
    mbFinished = false;
    mbStopped = false;

    pangolin::CreateWindowAndBind("Map Viewer", 1024, 768);

    // 3D Mouse handler requires depth testing to be enabled
    glEnable(GL_DEPTH_TEST);

    // Issue specific OpenGl we might need
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    pangolin::CreatePanel("menu").SetBounds(0.0, 1.0, 0.0, pangolin::Attach::Pix(175));
    pangolin::Var<bool> menuFollowCamera("menu.Follow Camera", true, true);
    pangolin::Var<bool> menuShowOctomap("menu.Show Octomap", true, true);
    pangolin::Var<bool> menuShowVertices("menu.Show Vertices", true, true);
    pangolin::Var<bool> menuShowEdges("menu.Show Edges", true, true);
    pangolin::Var<double> menuTime("menu.Track time:", 0);
    pangolin::Var<int> menuNodes("menu.Nodes:", 0);
    pangolin::Var<int> menuLoopCandidates("menu.LC:", 0);

    // Define Camera Render Object (for view / scene browsing)
    pangolin::OpenGlRenderState s_cam(
        pangolin::ProjectionMatrix(1024, 768, mViewpointF, mViewpointF, 512, 389, 0.1, 1000),
        pangolin::ModelViewLookAt(mViewpointX, mViewpointY, mViewpointZ, 0, 0, 0, 0.0, -1.0, 0.0));

    // Add named OpenGL viewport to window and provide 3D Handler
    pangolin::View& d_cam = pangolin::CreateDisplay()
                                .SetBounds(0.0, 1.0, pangolin::Attach::Pix(175), 1.0, -1024.0f / 768.0f)
                                .SetHandler(new pangolin::Handler3D(s_cam));

    pangolin::OpenGlMatrix Twc;
    Twc.SetIdentity();

    bool bFollow = true;

    while (true) {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        mpMapDrawer->getCurrentOpenGLCameraMatrix(Twc);

        if (menuFollowCamera && bFollow) {
            s_cam.Follow(Twc);
        } else if (menuFollowCamera && !bFollow) {
            s_cam.SetModelViewMatrix(pangolin::ModelViewLookAt(
                mViewpointX, mViewpointY, mViewpointZ, 0, 0, 0, 0.0, -1.0, 0.0));
            s_cam.Follow(Twc);
            bFollow = true;
        } else if (!menuFollowCamera && bFollow) {
            bFollow = false;
        }

        d_cam.Activate(s_cam);
        glClearColor(1.0f, 1.0f, 1.0f, 1.0f);

        mpMapDrawer->drawCurrentCamera(Twc);

        if (menuShowVertices || menuShowEdges)
            mpMapDrawer->drawPoseGraph(menuShowVertices, menuShowEdges);
        if (menuShowOctomap)
            mpMapDrawer->drawOctomap();

        menuNodes = int(mpMap->keyFramesInMap());

        {
            unique_lock<mutex> lock(mMutexUpdate);
            menuTime = mMeanTrackTime;
            menuLoopCandidates = mnLoopCandidates;
        }

        pangolin::FinishFrame();

        if (stop()) {
            while (isStopped()) {
                usleep(3000);
            }
        }

        if (checkFinish())
            break;
    }

    setFinish();
}

void Viewer::requestFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    mbFinishRequested = true;
}

bool Viewer::checkFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    return mbFinishRequested;
}

void Viewer::setFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    mbFinished = true;
}

bool Viewer::isFinished()
{
    unique_lock<mutex> lock(mMutexFinish);
    return mbFinished;
}

void Viewer::requestStop()
{
    unique_lock<mutex> lock(mMutexStop);
    if (!mbStopped)
        mbStopRequested = true;
}

bool Viewer::isStopped()
{
    unique_lock<mutex> lock(mMutexStop);
    return mbStopped;
}

bool Viewer::stop()
{
    unique_lock<mutex> lock(mMutexStop);
    unique_lock<mutex> lock2(mMutexFinish);

    if (mbFinishRequested)
        return false;
    else if (mbStopRequested) {
        mbStopped = true;
        mbStopRequested = false;
        return true;
    }

    return false;
}

void Viewer::release()
{
    unique_lock<mutex> lock(mMutexStop);
    mbStopped = false;
}

void Viewer::setMeanTrackingTime(const double& time)
{
    unique_lock<mutex> lock(mMutexUpdate);
    mMeanTrackTime = time;
}

void Viewer::loopCandidates(const size_t& n)
{
    unique_lock<mutex> lock(mMutexUpdate);
    mnLoopCandidates = int(n);
}

void Viewer::shutdown()
{
    requestFinish();
    if (!isFinished())
        usleep(5000);

    pangolin::BindToContext("Map Viewer");
}
