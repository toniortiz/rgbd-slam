#include "datasettumrgbd.h"
#include "Core/frame.h"
#include "Core/intrinsicmatrix.h"
#include "Core/rgbdcamera.h"
#include "Features/extractor.h"

using namespace std;

DatasetTUM::DatasetTUM()
    : Dataset("TUM")
    , mpCamera(nullptr)
{
}

DatasetTUM::~DatasetTUM() {}

bool DatasetTUM::isOpened() const { return (mAssociationFile.is_open() && mpCamera); }

Frame::Ptr DatasetTUM::grabFrame(Extractor::Ptr pExtractor, size_t i)
{
    if (isOpened()) {
        Frame::Ptr frame = mpCamera->createFrame(mBaseDir, mvImageFilenamesRGB[i], mvImageFilenamesD[i], mvTimestamps[i], pExtractor);
        return frame;
    } else
        return nullptr;
}

bool DatasetTUM::open(const string& dataset)
{
    mBaseDir = dataset;
    mAssociationFile.open(mBaseDir + "associations.txt");
    if (!mAssociationFile.is_open()) {
        cerr << "Can't open association file" << endl;
        return false;
    }

    detectCamera();

    while (!mAssociationFile.eof()) {
        string s;
        getline(mAssociationFile, s);
        if (!s.empty()) {
            stringstream ss;
            ss << s;
            double t;
            string sRGB, sD;
            ss >> t;
            mvTimestamps.push_back(t);
            ss >> sRGB;
            mvImageFilenamesRGB.push_back(sRGB);
            ss >> t;
            ss >> sD;
            mvImageFilenamesD.push_back(sD);
        }
    }
    return true;
}

size_t DatasetTUM::size() const { return mvImageFilenamesRGB.size(); }

bool DatasetTUM::detectCamera()
{
    // "/home/antonio/Documents/M.C.C/Tesis/Dataset/rgbd_dataset_freiburg1_desk2/
    string::size_type idx = mBaseDir.find("freiburg");
    IntrinsicMatrix* pIntrinsic = nullptr;

    char c = mBaseDir.at(idx + 8);
    switch (c) {
    case '1':
        pIntrinsic = new IntrinsicMatrix(517.306408f, 516.469215f, 318.643040f, 255.313989f);
        pIntrinsic->setDistortion(0.262383f, -0.953104f, 1.163314f, -0.005358f, 0.002628f);
        mpCamera = make_shared<RGBDcamera>(pIntrinsic, 40.0, 40.0, 5000.0, 30.0, 640, 480);
        break;

    case '2':
        pIntrinsic = new IntrinsicMatrix(520.908620f, 521.007327f, 325.141442f, 249.701764f);
        pIntrinsic->setDistortion(0.231222f, -0.784899f, 0.917205f, -0.003257f, -0.000105f);
        mpCamera = make_shared<RGBDcamera>(pIntrinsic, 40.0, 40.0, 5208.0, 30.0, 640, 480);
        break;

    case '3':
        pIntrinsic = new IntrinsicMatrix(535.4f, 539.2f, 320.1f, 247.6f);
        pIntrinsic->setDistortion(0, 0, 0, 0, 0);
        mpCamera = make_shared<RGBDcamera>(pIntrinsic, 40.0, 40.0, 5000.0, 30.0, 640, 480);
        break;
    }

    return true;
}

void DatasetTUM::print(ostream& out, const string& text) const
{
    Dataset::print(out, text);
    out << "Base Dir: " << mBaseDir << endl;
    out << "Size: " << size() << endl;
    out << *mpCamera << endl;
}

ostream& operator<<(ostream& out, const DatasetTUM& dataset)
{
    dataset.print(out, string(""));
    return out;
}
