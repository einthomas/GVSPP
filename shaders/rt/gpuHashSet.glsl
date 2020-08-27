const int HASH_SET_EMPTY_ENTRY = -1;

// 32 bit Murmur3 hash
int hash(int k, int capacity) {
    k ^= k >> 16;
    k *= 0x85ebca6b;
    k ^= k >> 13;
    k *= 0xc2b2ae35;
    k ^= k >> 16;
    return k & (capacity - 1);
}

bool insert(const int key) {
    #if SET_TYPE == 0
        if (key < 0) {
            return false;
        }

        int prev = atomicCompSwap(set[key], HASH_SET_EMPTY_ENTRY, key);
        if (prev == HASH_SET_EMPTY_ENTRY) {
            #ifndef BULK_INSERT
                atomicAdd(pvsSize, 1);
            #endif
            return true;
        } else {
            return false;
        }
    #elif SET_TYPE == 1
        if (key < 0) {
            return false;
        }
        
        int slot = hash(key, pvsCapacity);
        while (true) {
            int prev = atomicCompSwap(set[slot], HASH_SET_EMPTY_ENTRY, key);
            if (prev == HASH_SET_EMPTY_ENTRY) {
                #ifndef BULK_INSERT
                    atomicAdd(pvsSize, 1);
                #endif
                return true;
            } else if (prev == key) {
                return false;
            }

            slot = (slot + 1) & (pvsCapacity - 1);
        }
    #endif
}
