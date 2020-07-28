#ifndef CONCURRENTUNORDEREDSET_H
#define CONCURRENTUNORDEREDSET_H

#include <unordered_set>
#include <mutex>

template<class T>
class PVS {
public:
    std::vector<T> pvsVector;

    PVS() {
    }


    std::unordered_set<T> &getSet() {
        return set;
    }

    auto insert(T value, bool useMutex) {
        if (useMutex) {
            writeMutex.lock();
        }
        auto result = set.insert(value);
        if (useMutex) {
            writeMutex.unlock();
        }
        return result;
    }


private:
    std::unordered_set<T> set;
    std::mutex writeMutex;
};

#endif // CONCURRENTUNORDEREDSET_H
