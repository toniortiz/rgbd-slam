#include "Core/Frame.h"
#include "Core/Map.h"
#include "Features/Extractor.h"
#include "IO/DatasetCORBS.h"
#include "IO/DatasetICL.h"
#include "IO/DatasetTUM.h"
#include "System/Random.h"
#include "System/Tracking.h"
#include <iostream>
#include <opencv2/core/core.hpp>

using namespace std;

const string baseDir = "/home/antonio/Documents/M.C.C/Tesis/Dataset/TUM/rgbd_dataset_freiburg1_room/";
const string vocDir = "./Vocabulary/voc_TUM_FAST_BRIEF.yml.gz";

/* Available Datasets
 *  - DatasetTUM
 *  - DatasetICL
 *  - DatasetCORBS
*/
typedef DatasetTUM DatasetType;

int main()
{
    Random::initSeed();

    DatasetType::Ptr dataset(new DatasetType());
    dataset->open(baseDir);

    Extractor::Ptr extractor(new Extractor(Extractor::SVO, Extractor::BRIEF, Extractor::NORMAL));
    shared_ptr<DBoW3::Vocabulary> voc(new DBoW3::Vocabulary(vocDir));
    Map::Ptr pMap(new Map());
    Tracking tracker(voc, pMap);

    cout << static_cast<DatasetType&>(*dataset) << endl;

    cv::TickMeter tm;
    for (size_t ni = 0; ni < dataset->size(); ni++) {
        tm.start();
        Frame::Ptr frame = dataset->grabFrame(extractor, ni);

        tracker.track(frame);
        tracker.setCurrentPose(frame->getPose());

        tm.stop();
        tracker.setTime(tm.getTimeSec() / tm.getCounter());
    }

    tracker.shutdown();
    tracker.saveKeyFrameTrajectory("KeyFrameTrajectory.txt");
    tracker.saveCameraTrajectory("CameraTrajectory.txt");

    return 0;
}
