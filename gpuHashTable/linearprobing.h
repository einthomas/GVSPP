#pragma once

// Based on https://github.com/nosferalatu/SimpleGPUHashTable
struct KeyValue
{
    int key;
};

//const int kHashTableCapacity = 128 * 1024 * 1024;

//const int kNumKeyValues = kHashTableCapacity / 2;

const int kEmpty = 0xffffffff;

class GPUHashSet {
public:
    int *hashSet;
    char *deviceInserted;
    int capacity;

    GPUHashSet(int capacity);
    ~GPUHashSet();
    void reset();
    void insert(const int *keys, int num_kvs);
    int* resize(int newSize);
};
