#ifndef DATASET_H
#define DATASET_H

#include "System/macros.h"
#include <memory>
#include <string>

class Frame;
class Extractor;

class Dataset {
public:
    SMART_POINTER_TYPEDEFS(Dataset);

    enum DS {
        TUM = 0,
        ICL,
        D
    };

public:
    Dataset();
    Dataset(const std::string& name);
    virtual ~Dataset();

    std::string name() const;
    virtual bool isOpened() const;
    virtual std::shared_ptr<Frame> grabFrame(std::shared_ptr<Extractor> pExtractor, size_t i) = 0;
    virtual size_t size() const;
    virtual bool open(const std::string& dataset);
    virtual void print(std::ostream& out, const std::string& text) const;

    static Dataset::Ptr create(const DS& dataset);

    std::ostream& operator<<(std::ostream& out);

protected:
    std::string mName;
};

#endif // DATASET_H
