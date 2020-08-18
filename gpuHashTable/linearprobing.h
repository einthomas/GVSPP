#pragma once

// Based on https://github.com/nosferalatu/SimpleGPUHashTable
struct KeyValue
{
    int key;
};

//const int kHashTableCapacity = 128 * 1024 * 1024;

//const int kNumKeyValues = kHashTableCapacity / 2;

const int kEmpty = 0xffffffff;

int* create_hashtable(int capacity);

//void insert_hashtable(KeyValue* hashtable, const KeyValue* kvs, uint32_t num_kvs, char* inserted);
void insert_hashtable(int* hashTable, const int *keys, int num_kvs, char* inserted);

void lookup_hashtable(KeyValue* hashtable, KeyValue* kvs, uint32_t num_kvs);

void delete_hashtable(KeyValue* hashtable, const KeyValue* kvs, uint32_t num_kvs);

std::vector<KeyValue> iterate_hashtable(KeyValue* hashtable);

void destroy_hashtable(KeyValue* hashtable);

int* resize(int* hashTable, int newSize);
