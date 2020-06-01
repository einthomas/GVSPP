#ifndef CONCURRENTUNORDEREDSET_H
#define CONCURRENTUNORDEREDSET_H

#include <unordered_set>
#include <mutex>

template<class T>
class PVS {
public:
    PVS() {
    }

    const std::unordered_set<T> &getSet() {
        return set;
    }

    auto insert(T value) {
        writeMutex.lock();
        auto result = set.insert(value);
        writeMutex.unlock();
        return result;
    }

private:
    std::unordered_set<T> set;
    std::mutex writeMutex;
};

#endif // CONCURRENTUNORDEREDSET_H
