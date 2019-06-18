#include "viewer.h"
#include "Core/map.h"
#include "System/tracking.h"
#include "mapdrawer.h"
#include <pangolin/pangolin.h>
#include <unistd.h>

using namespace std;

Viewer::Viewer(MapDrawer::Ptr pMapDrawer, Map::Ptr pMap, Tracking* pTracker)
    : mpMapDrawer(pMapDrawer)
    , mpMap(pMap)
    , mpTracking(pTracker)
    , mbFinishRequested(false)
    , mbFinished(true)
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

    mRunThread = thread(&Viewer::run, this);
}

void Viewer::run()
{
    mbFinished = false;

    pangolin::CreateWindowAndBind("Viewer", 1024, 768);

    // 3D Mouse handler requires depth testing to be enabled
    glEnable(GL_DEPTH_TEST);

    // Issue specific OpenGl we might need
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    pangolin::CreatePanel("menu").SetBounds(0.0, 1.0, 0.0, pangolin::Attach::Pix(175));
    pangolin::Var<bool> menuFollowCamera("menu.Follow Camera", true, true);
    pangolin::Var<bool> menuShowLandmarks("menu.Landmarks", true, true);
    pangolin::Var<bool> menuShowOctomap("menu.Show Octomap", false, true);
    pangolin::Var<bool> menuShowVertices("menu.Show Vertices", true, true);
    pangolin::Var<bool> menuShowEdges("menu.Show Edges", true, true);
    pangolin::Var<double> menuTime("menu.Track time:", 0);
    pangolin::Var<int> menuNodes("menu.Nodes:", 0);
    pangolin::Var<int> menuLandmarks("menu.Landmarks:", 0);
    pangolin::Var<int> menuLoopCandidates("menu.LC:", 0);

    // Define Camera Render Object (for view / scene browsing)
    pangolin::OpenGlRenderState s_cam(
        pangolin::ProjectionMatrix(1024, 768, mViewpointF, mViewpointF, 512, 389, 0.1, 1000),
        pangolin::ModelViewLookAt(mViewpointX, mViewpointY, mViewpointZ, 0, 0, 0, 0.0, -1.0, 0.0));

    // Add named OpenGL viewport to window and provide 3D Handler
    pangolin::View& d_cam = pangolin::CreateDisplay()
                                .SetBounds(0.0, 1.0, pangolin::Attach::Pix(175), 1.0, -1024.0f / 768.0f)
                                .SetHandler(new pangolin::Handler3D(s_cam));

    pangolin::View& d_features = pangolin::Display("imgFeatures")
                                     .SetAspect(1024.0 / 768.0);
    pangolin::GlTexture texFeatures(mImageWidth, mImageHeight, GL_RGB, false, 0, GL_RGB, GL_UNSIGNED_BYTE);

    pangolin::CreateDisplay()
        .SetBounds(0.0, 0.3f, pangolin::Attach::Pix(175), 1.0)
        .SetLayout(pangolin::LayoutEqual)
        .AddDisplay(d_features);

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

        //drawHorizontalGrid();

        if (menuShowVertices || menuShowEdges)
            mpMapDrawer->drawPoseGraph(menuShowVertices, menuShowEdges);
        if (menuShowOctomap)
            mpMapDrawer->drawOctomap();
        if (menuShowLandmarks)
            mpMapDrawer->drawLandmarks();

        menuNodes = int(mpMap->keyFramesInMap());
        menuLandmarks = int(mpMap->landmarksInMap());

        {
            unique_lock<mutex> lock(mMutexUpdate);
            menuTime = mMeanTrackTime;
            menuLoopCandidates = mnLoopCandidates;
        }

        cv::Mat feats = mpTracking->getTrackedPointsImage();
        if (!feats.empty()) {
            texFeatures.Upload(feats.data, GL_BGR, GL_UNSIGNED_BYTE);
            d_features.Activate();
            glColor4f(1.0f, 1.0f, 1.0f, 1.0f);
            texFeatures.RenderToViewportFlipY();
        }

        pangolin::FinishFrame();

        if (checkFinish())
            break;

        usleep(3000);
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
    while (!isFinished())
        usleep(5000);

    pangolin::BindToContext("Viewer");

    if (mRunThread.joinable())
        mRunThread.join();
}

void Viewer::drawHorizontalGrid()
{
    Eigen::Matrix4f origin;
    origin << 0, 0, 1, 0, -1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 1;
    glPushMatrix();
    glMultTransposeMatrixf(origin.data());

    glLineWidth(1);
    glColor3f(0.5, 0.5, 0.5);
    glBegin(GL_LINES);

    constexpr float interval_ratio = 0.1;
    constexpr float grid_min = -100.0f * interval_ratio;
    constexpr float grid_max = 100.0f * interval_ratio;

    for (int x = -10; x <= 10; x += 1) {
        drawLine(x * 10.0f * interval_ratio, grid_min, 0, x * 10.0f * interval_ratio, grid_max, 0);
    }
    for (int y = -10; y <= 10; y += 1) {
        drawLine(grid_min, y * 10.0f * interval_ratio, 0, grid_max, y * 10.0f * interval_ratio, 0);
    }

    glEnd();

    glPopMatrix();
}

void Viewer::drawLine(const float x1, const float y1, const float z1, const float x2, const float y2, const float z2)
{
    glVertex3f(x1, y1, z1);
    glVertex3f(x2, y2, z2);
}
