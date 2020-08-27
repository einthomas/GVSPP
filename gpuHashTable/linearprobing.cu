#include "stdio.h"
#include "stdint.h"
#include "vector"
#include "linearprobing.h"

GPUHashSet::GPUHashSet(int capacity)
    : capacity(capacity)
{
    // Allocate memory
    cudaMalloc(&hashSet, sizeof(int) * capacity);

    // Initialize hash table to empty
    static_assert(kEmpty == 0xffffffff, "memset expected kEmpty=0xffffffff");
    cudaMemset(hashSet, 0xff, sizeof(int) * capacity);

    cudaMalloc(&deviceInserted, sizeof(char) * capacity);
    cudaMemset(deviceInserted, 0x0, sizeof(char) * capacity);
}

GPUHashSet::~GPUHashSet() {
    cudaFree(hashSet);
    cudaFree(deviceInserted);
}

// 32 bit Murmur3 hash
__device__ int hash(int k, int capacity) {
    k ^= k >> 16;
    k *= 0x85ebca6b;
    k ^= k >> 13;
    k *= 0xc2b2ae35;
    k ^= k >> 16;
    return k & (capacity - 1);
}

void GPUHashSet::reset() {
    cudaMemset(hashSet, 0xff, sizeof(int) * capacity);
    cudaMemset(deviceInserted, 0x0, sizeof(char) * capacity);
}

__global__ void gpu_resize(int *hashSet, int *newHashSet, int size, int newSize) {
    unsigned int threadid = blockIdx.x * blockDim.x + threadIdx.x;
    if (threadid < size) {
        int key = hashSet[threadid];
        int slot = hash(key, newSize);
        while (true) {
            int prev = atomicCAS(&newHashSet[slot], kEmpty, key);
            if (prev == kEmpty || prev == key) {
                return;
            }

            slot = (slot + 1) & (newSize-1);
        }
    }
}

int* GPUHashSet::resize(int newSize) {
    int oldSize = capacity;

    // Create events for GPU timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);


    // Allocate memory
    int* newHashSet;
    cudaMalloc(&newHashSet, sizeof(int) * newSize);

    // Initialize hash table to empty
    static_assert(kEmpty == 0xffffffff, "memset expected kEmpty=0xffffffff");
    cudaMemset(newHashSet, 0xff, sizeof(int) * newSize);

    int mingridsize;
    int threadblocksize;
    cudaOccupancyMaxPotentialBlockSize(&mingridsize, &threadblocksize, gpu_resize, 0, 0);
    int gridsize = (capacity + threadblocksize - 1) / threadblocksize;
    //gpu_hashtable_insert<<<gridsize, threadblocksize>>>(pHashTable, device_keys, (int)num_kvs, deviceInserted);
    gpu_resize<<<gridsize, threadblocksize>>>(hashSet, newHashSet, capacity, newSize);

    capacity = newSize;

    cudaFree(hashSet);


    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    float seconds = milliseconds / 1000.0f;
    printf("resize hash table %d -> %d: %f ms\n", oldSize, newSize, milliseconds);

    return newHashSet;
}

// Insert the key/values in kvs into the hashtable
__global__ void gpu_hashtable_insert(
    int *hashtable, const int *keys, unsigned int numkvs, char *inserted, int hashTableCapacity
) {
    unsigned int threadid = blockIdx.x*blockDim.x + threadIdx.x;
    if (threadid < numkvs)
    {
        int key = keys[threadid];
        if (key < 0) {
            inserted[threadid] = 0;
            return;
        }

        //int value = values[threadid];
        int slot = hash(key, hashTableCapacity);

        while (true)
        {
            int prev = atomicCAS(&hashtable[slot], kEmpty, key);
            if (prev == kEmpty)
            {
                //hashtable[slot].value = value;
                inserted[threadid] = 1;
                return;
            }
            else if (prev == key)
            {
                inserted[threadid] = 0;
                return;
            }

            slot = (slot + 1) & (hashTableCapacity-1);
        }
    }

    /*
    // Set without hashing
    unsigned int threadid = blockIdx.x*blockDim.x + threadIdx.x;
    if (threadid < numkvs) {
        int key = keys[threadid];
        if (key < 0) {
            inserted[threadid] = 0;
            return;
        }

        int prev = atomicCAS(&hashtable[key], kEmpty, key);
        if (prev == kEmpty) {
            inserted[threadid] = 1;
        } else if (prev == key) {
            inserted[threadid] = 0;
        }
    }
    */
}

void GPUHashSet::insert(const int *keys, int num_kvs) {
    cudaMemset(deviceInserted, 0x0, sizeof(char) * capacity);

    // Have CUDA calculate the thread block size
    int mingridsize;
    int threadblocksize;
    cudaOccupancyMaxPotentialBlockSize(&mingridsize, &threadblocksize, gpu_hashtable_insert, 0, 0);

    // Create events for GPU timing
    /*
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    */

    // Insert all the keys into the hash table
    int gridsize = ((int)num_kvs + threadblocksize - 1) / threadblocksize;
    //gpu_hashtable_insert<<<gridsize, threadblocksize>>>(pHashTable, device_keys, (int)num_kvs, device_inserted, capacity);
    gpu_hashtable_insert<<<gridsize, threadblocksize>>>(hashSet, keys, (int)num_kvs, deviceInserted, capacity);

    /*
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    float seconds = milliseconds / 1000.0f;
    printf("    GPU inserted %d items in %f ms (%f million keys/second)\n",
        num_kvs, milliseconds, num_kvs / (double)seconds / 1000000.0f);
    */
}
