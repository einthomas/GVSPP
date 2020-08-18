#include "stdio.h"
#include "stdint.h"
#include "vector"
#include "linearprobing.h"

int kHashTableCapacity;

// 32 bit Murmur3 hash
__device__ int hash(int k, int hashTableCapacity)
{
    k ^= k >> 16;
    k *= 0x85ebca6b;
    k ^= k >> 13;
    k *= 0xc2b2ae35;
    k ^= k >> 16;
    return k & (hashTableCapacity-1);
}

// Create a hash table. For linear probing, this is just an array of KeyValues
int* create_hashtable(int capacity)
{
    kHashTableCapacity = capacity;

    // Allocate memory
    int* hashtable;
    cudaMalloc(&hashtable, sizeof(int) * kHashTableCapacity);

    // Initialize hash table to empty
    static_assert(kEmpty == 0xffffffff, "memset expected kEmpty=0xffffffff");
    cudaMemset(hashtable, 0xff, sizeof(int) * kHashTableCapacity);

    return hashtable;
}

__global__ void gpu_resize(int* hashTable, int* newHashTable, int size, int newSize)
{
    unsigned int threadid = blockIdx.x*blockDim.x + threadIdx.x;
    if (threadid < size)
    {
        int key = hashTable[threadid];
        int slot = hash(key, newSize);
        while (true)
        {
            int prev = atomicCAS(&newHashTable[slot], kEmpty, key);
            if (prev == kEmpty || prev == key)
            {
                return;
            }

            slot = (slot + 1) & (newSize-1);
        }
    }
}

int* resize(int* hashTable, int newSize)
{
    // Create events for GPU timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);


    // Allocate memory
    int* newHashTable;
    cudaMalloc(&newHashTable, sizeof(int) * newSize);

    // Initialize hash table to empty
    static_assert(kEmpty == 0xffffffff, "memset expected kEmpty=0xffffffff");
    cudaMemset(newHashTable, 0xff, sizeof(int) * newSize);

    int mingridsize;
    int threadblocksize;
    cudaOccupancyMaxPotentialBlockSize(&mingridsize, &threadblocksize, gpu_resize, 0, 0);
    int gridsize = (kHashTableCapacity + threadblocksize - 1) / threadblocksize;
    //gpu_hashtable_insert<<<gridsize, threadblocksize>>>(pHashTable, device_keys, (int)num_kvs, device_inserted);
    gpu_resize<<<gridsize, threadblocksize>>>(hashTable, newHashTable, kHashTableCapacity, newSize);

    kHashTableCapacity = newSize;

    cudaFree(hashTable);


    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    float seconds = milliseconds / 1000.0f;
    printf("resize hash table %d -> %d: %f ms\n", newSize / 2, newSize, milliseconds);

    return newHashTable;
}

// Insert the key/values in kvs into the hashtable
__global__ void gpu_hashtable_insert(int* hashtable, const int *keys, unsigned int numkvs, char* inserted, int hashTableCapacity)
{
    unsigned int threadid = blockIdx.x*blockDim.x + threadIdx.x;
    if (threadid < numkvs)
    {
        int key = keys[threadid];
        if (key == -1) {
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
}

void insert_hashtable(int* pHashTable, const int *keys, int num_kvs, char* inserted)
{
    /*
    // Copy the keys to the GPU
    int* device_keys;
    cudaMalloc(&device_keys, sizeof(int) * num_kvs);
    cudaMemcpy(device_keys, keys, sizeof(int) * num_kvs, cudaMemcpyHostToDevice);
    */

    char* device_inserted;
    cudaMalloc(&device_inserted, sizeof(char) * num_kvs);
    cudaMemcpy(device_inserted, inserted, sizeof(char) * num_kvs, cudaMemcpyHostToDevice);

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
    //gpu_hashtable_insert<<<gridsize, threadblocksize>>>(pHashTable, device_keys, (int)num_kvs, device_inserted, kHashTableCapacity);
    gpu_hashtable_insert<<<gridsize, threadblocksize>>>(pHashTable, keys, (int)num_kvs, device_inserted, kHashTableCapacity);

    /*
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);


    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    float seconds = milliseconds / 1000.0f;

    printf("    GPU inserted %d items in %f ms (%f million keys/second)\n",
        num_kvs, milliseconds, num_kvs / (double)seconds / 1000000.0f);
    */

    cudaMemcpy(inserted, device_inserted, sizeof(char) * num_kvs, cudaMemcpyDeviceToHost);

    cudaFree(device_inserted);
    //cudaFree(device_keys);
}

// Free the memory of the hashtable
void destroy_hashtable(KeyValue* pHashTable)
{
    cudaFree(pHashTable);
}
