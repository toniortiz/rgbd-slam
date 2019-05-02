#ifndef MACROS_H
#define MACROS_H

#include <memory>

#define SMART_POINTER_TYPEDEFS(T)         \
    typedef std::unique_ptr<T> UniquePtr; \
    typedef std::shared_ptr<T> Ptr;       \
    typedef std::shared_ptr<const T> ConstPtr

#endif // MACROS_H
