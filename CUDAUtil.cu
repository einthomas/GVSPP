#include "CUDAUtil.h"

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/random.h>
#include <thrust/generate.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/unique.h>
#include <thrust/device_ptr.h>
#include <thrust/tuple.h>
#include <thrust/functional.h>
#include <thrust/set_operations.h>
#include <thrust/binary_search.h>
#include <thrust/gather.h>
#include <thrust/scatter.h>
#include <thrust/remove.h>

#include <thrust/tuple.h>
#include <thrust/iterator/zip_iterator.h>

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <chrono>
#include <vector>

#include "sample.h"

void sortByKey(thrust::device_ptr<int> triangleIDs, int size, thrust::device_ptr<int> sampleIndices);
int uniqueByKey(thrust::device_ptr<int> triangleIDs, int size, thrust::device_ptr<int> sampleIndices);
void findNewTriangles(
    thrust::device_ptr<int> devicePointerPVS, thrust::device_ptr<int> triangleIDs, thrust::device_ptr<Sample> samples, std::vector<Sample> &result, int pvsSize,
    int trianglesSize, thrust::device_ptr<int> sampleIndices
);
int setUnion(thrust::device_ptr<int> devicePointerPVS, thrust::device_ptr<int> triangleIDs, int sizeA, int sizeB);

int CUDAUtil::work(int *pvs, int *triangleIDKeys, Sample *sampleValues, std::vector<Sample> &result, int pvsSize, int triangleIDKeysSize) {
    thrust::device_vector<int> deviceSampleValueIndices(triangleIDKeysSize);
    thrust::sequence(deviceSampleValueIndices.begin(), deviceSampleValueIndices.end());
    thrust::device_ptr<int> sampleIndices = deviceSampleValueIndices.data();

    thrust::device_ptr<int> triangleIDs(triangleIDKeys);
    thrust::device_ptr<Sample> samples(sampleValues);

    thrust::device_ptr<int> devicePointerPVS(pvs);

    sortByKey(triangleIDs, triangleIDKeysSize, sampleIndices);

    int numTriangles = uniqueByKey(triangleIDs, triangleIDKeysSize, sampleIndices);

    if (triangleIDs[0] == -1) {
        triangleIDs++;
        sampleIndices++;
        numTriangles--;
    }

    if (numTriangles > 0) {
        findNewTriangles(devicePointerPVS, triangleIDs, samples, result, pvsSize, numTriangles, sampleIndices);
        if (result.size() > 0) {
            pvsSize = setUnion(devicePointerPVS, triangleIDs, pvsSize, numTriangles);
        }
    }

    return pvsSize;
}

void sortByKey(thrust::device_ptr<int> triangleIDs, int size, thrust::device_ptr<int> sampleIndices) {
    thrust::sort_by_key(triangleIDs, triangleIDs + size, sampleIndices);
    cudaDeviceSynchronize();
}

int uniqueByKey(thrust::device_ptr<int> triangleIDs, int size, thrust::device_ptr<int> sampleIndices) {
    auto newEnd = thrust::unique_by_key(triangleIDs, triangleIDs + size, sampleIndices);
    cudaDeviceSynchronize();
    return newEnd.first - triangleIDs;
}

int setUnion(thrust::device_ptr<int> devicePointerPVS, thrust::device_ptr<int> triangleIDs, int sizeA, int sizeB) {
    thrust::device_vector<int> result(sizeA + sizeB);
    auto newEnd = thrust::set_union(
        devicePointerPVS, devicePointerPVS + sizeA, triangleIDs, triangleIDs + sizeB,
        result.begin(), thrust::less<int>()
    );
    cudaDeviceSynchronize();

    int resultSize = newEnd - result.begin();
    thrust::copy(result.begin(), result.begin() + resultSize, devicePointerPVS);
    cudaDeviceSynchronize();

    return resultSize;
}

void findNewTriangles(
    thrust::device_ptr<int> devicePointerPVS, thrust::device_ptr<int> triangleIDs, thrust::device_ptr<Sample> samples, std::vector<Sample> &result, int pvsSize,
    int trianglesSize, thrust::device_ptr<int> devicePointerSampleValueIndices
) {
    // Search which triangles are already in the PVS
    thrust::device_vector<bool> stencil(trianglesSize);
    thrust::binary_search(
        devicePointerPVS, devicePointerPVS + pvsSize,
        triangleIDs, triangleIDs + trianglesSize,
        stencil.begin()
    );
    cudaDeviceSynchronize();

    // Count the number of triangles that are not in the PVS
    int numNewTriangles = thrust::count(stencil.begin(), stencil.end(), 0);
    cudaDeviceSynchronize();

    if (numNewTriangles > 0) {
        // Remove the indices referring to samples that are already in the PVS
        thrust::remove_if(devicePointerSampleValueIndices, devicePointerSampleValueIndices + trianglesSize, stencil.begin(), thrust::identity<int>());
        cudaDeviceSynchronize();

        // Store the new samples in a result vector
        thrust::device_vector<Sample> r(numNewTriangles);
        auto newEnd = thrust::gather(
            devicePointerSampleValueIndices, devicePointerSampleValueIndices + numNewTriangles,
            samples,
            r.begin()
        );
        cudaDeviceSynchronize();

        result.resize(numNewTriangles);
        thrust::copy(r.begin(), r.end(), result.begin());
        cudaDeviceSynchronize();
    }
}

__global__ void haltonKernel(int n, float *sequence, int startIndex) {
    int offset = blockIdx.x * blockDim.x + threadIdx.x;
    int bases[4] = { 2, 3, 5, 7 };

    for (int i = 0; i < 4; i++) {
        float f = 1.0f;
        float r = 0.0f;
        int k = startIndex + offset + 1;
        while (k > 0) {
            f /= bases[i];
            r = r + f * (k % bases[i]);
            k = int(k / bases[i]);
        }
        sequence[offset * 4 + i] = r;
    }
}

void CUDAUtil::generateHaltonSequence(int n, float *sequence, int startIndex) {
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    haltonKernel<<<numBlocks, blockSize>>>(blockSize, sequence, startIndex);
    cudaDeviceSynchronize();
}

int CUDAUtil::initCuda(uint8_t *vkDeviceUUID, size_t UUID_SIZE) {
    int current_device = 0;
    int device_count = 0;
    int devices_prohibited = 0;

    cudaDeviceProp deviceProp;
    cudaGetDeviceCount(&device_count);

    if (device_count == 0) {
        fprintf(stderr, "CUDA error: no devices supporting CUDA.\n");
        exit(EXIT_FAILURE);
    }

    // Find the GPU which is selected by Vulkan
    while (current_device < device_count) {
        cudaGetDeviceProperties(&deviceProp, current_device);

        if ((deviceProp.computeMode != cudaComputeModeProhibited)) {
            // Compare the cuda device UUID with vulkan UUID
            int ret = memcmp((void*)&deviceProp.uuid, vkDeviceUUID, UUID_SIZE);
            if (ret == 0) {
                cudaSetDevice(current_device);
                cudaGetDeviceProperties(&deviceProp, current_device);
                printf("GPU Device %d: \"%s\" with compute capability %d.%d\n\n",
                 current_device, deviceProp.name, deviceProp.major,
                 deviceProp.minor);

                return current_device;
            }

        } else {
          devices_prohibited++;
        }

        current_device++;
    }

    if (devices_prohibited == device_count) {
        fprintf(stderr, "CUDA error: No Vulkan-CUDA Interop capable GPU found.\n");
        exit(EXIT_FAILURE);
    }

    return -1;
}
