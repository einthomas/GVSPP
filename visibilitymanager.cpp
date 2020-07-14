#include <cstring>
#include <glm/gtx/string_cast.hpp>
#include <iostream>
#include <chrono>
#include <iterator>
#include <algorithm>
#include <random>

#include <unordered_set>

#include "vulkanutil.h"
#include "visibilitymanager.h"
#include "viewcell.h"
#include "sample.h"
#include "Vertex.h"

struct UniformBufferObject {
    alignas(16) glm::mat4 model;
    alignas(16) glm::mat4 view;
    alignas(16) glm::mat4 projection;
};

VisibilityManager::VisibilityManager()
    : statistics(RAY_COUNT_TERMINATION_THRESHOLD)
{
}

void VisibilityManager::init(
    VkPhysicalDevice physicalDevice, VkDevice logicalDevice,
    VkBuffer indexBuffer, const std::vector<uint32_t> &indices, VkBuffer vertexBuffer,
    const std::vector<Vertex> &vertices, const std::vector<VkBuffer> &uniformBuffers, int numThreads
) {
    this->logicalDevice = logicalDevice;
    this->physicalDevice = physicalDevice;
    this->numThreads = numThreads;

    tracedRays = 0;
    queueSubmitMutex = new std::mutex();

    gen.seed(rd());

    uint32_t computeQueueFamilyIndex = VulkanUtil::findQueueFamilies(
        physicalDevice, VK_QUEUE_COMPUTE_BIT, 0
    );
    vkGetDeviceQueue(logicalDevice, computeQueueFamilyIndex, 0, &computeQueue);

    commandPool.resize(numThreads);
    VkCommandPoolCreateInfo cmdPoolInfo = {};
    cmdPoolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    cmdPoolInfo.queueFamilyIndex = computeQueueFamilyIndex;
    cmdPoolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;    // Has to be set otherwise the command buffers can't be re-recorded
    for (int i = 0; i < numThreads; i++) {
        if (vkCreateCommandPool(logicalDevice, &cmdPoolInfo, nullptr, &commandPool[i])) {
            throw std::runtime_error("failed to create visibility manager command pool!");
        }
    }

    createBuffers(indices);

    haltonPoints.resize(numThreads);
    for (int i = 0; i < numThreads; i++) {
        generateHaltonPoints2d(RAYS_PER_ITERATION, i, 0);
        copyHaltonPointsToBuffer(i);
    }

    createViewCellBuffer();

    initRayTracing(indexBuffer, vertexBuffer, indices, vertices, uniformBuffers);
}

void VisibilityManager::addViewCell(glm::vec3 pos, glm::vec3 size, glm::vec3 normal) {
    viewCells.push_back(ViewCell(pos, size, normal));
}

void VisibilityManager::generateHaltonPoints2d(int n, int threadId, int offset) {
    int bases[4] = { 2, 3, 5, 7 };

    haltonPoints[threadId].clear();
    haltonPoints[threadId].resize(n);

    for (int k = 0; k < 4; k++) {
        double inverseBase = 1.0 / bases[k];
        double value = offset;

        for (int i = 0; i < n; i++) {
            double r = 1.0 - value - 1e-10;

            if (inverseBase < r) {
                value += inverseBase;
            } else {
                double h = inverseBase * inverseBase;
                double hh = inverseBase;
                while (h >= r) {
                    hh = h;
                    h *= inverseBase;
                }
                value += hh + h - 1.0;
            }

            haltonPoints[threadId][i][k] = value;
        }
    }

    /*
    haltonPoints[threadId].clear();
    haltonPoints[threadId].resize(n);

    auto start = std::chrono::steady_clock::now();
    float p, u, v, ip;
    int k, kk, pos, a;
    for (k = offset + threadId * n, pos = 0; k < n + offset + threadId * n; k++) {
        u = 0;
        for (p = 0.5, kk = k; kk; p *= 0.5, kk >>= 1) {
            if (kk & 1) {
                u += p;
            }
        }

        v = 0;
        ip = 1.0 / p2;
        for (p = ip, kk = k; kk; p *= ip, kk /= p2) {
            if ((a = kk % p2)) {
                v += a * p;
            }
        }

        haltonPoints[threadId][pos].x = u;
        haltonPoints[threadId][pos].y = v;

        pos++;
    }
    auto end = std::chrono::steady_clock::now();
    /*

    //std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << std::endl;

    //std::shuffle(haltonPoints[threadId].begin(), haltonPoints[threadId].end(), gen);      // TODO einkommentieren

    /*
    if (haltonPoints.size() == 0) {
        haltonPoints.resize(numThreads);
    }
    for (int i = 0; i < numThreads; i++) {
        haltonPoints[i].clear();
        haltonPoints[i].resize(n);

        float p, u, v, ip;
        int k, kk, pos, a;
        for (k = offset + i * n, pos = 0; k < n + offset + i * n; k++) {
            u = 0;
            for (p = 0.5, kk = k; kk; p *= 0.5, kk >>= 1) {
                if (kk & 1) {
                    u += p;
                }
            }

            v = 0;
            ip = 1.0 / p2;
            for (p = ip, kk = k; kk; p *= ip, kk /= p2) {
                if ((a = kk % p2)) {
                    v += a * p;
                }
            }

            haltonPoints[i][pos].x = u;
            haltonPoints[i][pos].y = v;
            pos++;
        }

        std::shuffle(haltonPoints[i].begin(), haltonPoints[i].end(), gen);
    }
    */
}

void VisibilityManager::copyHaltonPointsToBuffer(int threadId) {
    VkDeviceSize bufferSize;
    bufferSize = sizeof(haltonPoints[threadId][0]) * haltonPoints[threadId].size();

    // Create staging buffer using host-visible memory
    VkBuffer stagingBuffer;
    VkDeviceMemory stagingBufferMemory;
    VulkanUtil::createBuffer(
        physicalDevice, logicalDevice, bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, stagingBuffer, stagingBufferMemory,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
    );

    // Copy halton points to the staging buffer
    void *data;
    vkMapMemory(logicalDevice, stagingBufferMemory, 0, bufferSize, 0, &data);    // Map buffer memory into CPU accessible memory
    memcpy(data, haltonPoints[threadId].data(), (size_t) bufferSize);  // Copy 2d halton points to mapped memory
    vkUnmapMemory(logicalDevice, stagingBufferMemory);

    // Copy halton points from the staging buffer to the halton points buffer
    VulkanUtil::copyBuffer(
        logicalDevice, commandPool[threadId], computeQueue, stagingBuffer, haltonPointsBuffer[threadId],
        bufferSize, queueSubmitMutex
    );

    vkDestroyBuffer(logicalDevice, stagingBuffer, nullptr);
    vkFreeMemory(logicalDevice, stagingBufferMemory, nullptr);
}

void VisibilityManager::createViewCellBuffer() {
    viewCellBuffer.resize(numThreads);
    viewCellBufferMemory.resize(numThreads);

    VkDeviceSize bufferSize = sizeof(viewCells[0]) * viewCells.size();

    for (int i = 0; i < numThreads; i++) {
        // Create staging buffer using host-visible memory
        VkBuffer stagingBuffer;
        VkDeviceMemory stagingBufferMemory;
        VulkanUtil::createBuffer(
            physicalDevice,
            logicalDevice, bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, stagingBuffer, stagingBufferMemory,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
        );

        ViewCell viewCellTile = getViewCellTile(numThreads, 0, i);
        viewCells[0].tilePos = viewCellTile.tilePos;
        viewCells[0].tileSize = viewCellTile.tileSize;

        // Copy view cell to the staging buffer
        void *data;
        vkMapMemory(logicalDevice, stagingBufferMemory, 0, bufferSize, 0, &data);    // Map buffer memory into CPU accessible memory
        memcpy(data, viewCells.data(), (size_t) bufferSize);  // Copy vertex data to mapped memory
        //memcpy(data, &viewCell, (size_t) bufferSize);
        vkUnmapMemory(logicalDevice, stagingBufferMemory);

        // Create view cell buffer using GPU memory
        VulkanUtil::createBuffer(
            physicalDevice,
            logicalDevice, bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_RAY_TRACING_BIT_NV | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
            viewCellBuffer[i], viewCellBufferMemory[i], VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
        );

        // Copy halton points from the staging buffer to the view cell buffer
        VulkanUtil::copyBuffer(
            logicalDevice, commandPool[i], computeQueue, stagingBuffer, viewCellBuffer[i], bufferSize,
            queueSubmitMutex
        );

        vkDestroyBuffer(logicalDevice, stagingBuffer, nullptr);
        vkFreeMemory(logicalDevice, stagingBufferMemory, nullptr);
    }
}

/*
 * Possible number of thread counts (numThreads): 1, 2, 4, 6, 8, 9
 */
ViewCell VisibilityManager::getViewCellTile(int numThreads, int viewCellIndex, int threadId) {
    glm::vec3 up = glm::vec3(0.0, 1.0, 0.0);
    glm::vec3 viewCellRight = glm::normalize(glm::cross(viewCells[viewCellIndex].normal, up));     // TODO: Doesn't work if the viewcell normal is also (0, 1, 0)!
    glm::vec3 viewCellUp = glm::normalize(glm::cross(viewCellRight, viewCells[viewCellIndex].normal));

    ViewCell viewCell;
    viewCell.normal = viewCells[viewCellIndex].normal;
    if (numThreads == 1) {
        viewCell.tilePos = viewCells[viewCellIndex].pos;
        viewCell.tileSize = viewCells[viewCellIndex].size;
    } else if (numThreads % 2 == 0) {
        glm::vec2 split(
            -0.5 + 1.0f / numThreads + 1.0f / (numThreads * 0.5f) * (threadId % int(numThreads * 0.5f)),
            0.25f * (int(threadId / (numThreads * 0.5f)) == 0 ? 1.0f : -1.0f)
        );
        viewCell.tilePos = viewCells[viewCellIndex].pos + viewCellRight * split.x + viewCellUp * split.y;
        viewCell.tileSize = glm::vec3(1.0f / (numThreads * 0.5f), 0.5f, 1.0f) * viewCells[viewCellIndex].size;
    } else if (numThreads == 9) {
        glm::vec2 split(
            -0.5 + 1.0f / 6.0f + (1.0f / 3.0f) * (threadId % int(numThreads / 3.0f)),
            (1.0f / 3.0f) * (int(threadId / (numThreads / 3.0f)) - 1.0f)
        );
        viewCell.tilePos = viewCells[viewCellIndex].pos + viewCellRight * split.x + viewCellUp * split.y;
        viewCell.tileSize = glm::vec3(1.0f / 3.0f, 1.0f / 3.0f, 1.0f) * viewCells[viewCellIndex].size;
    }

    return viewCell;
}

void VisibilityManager::createBuffers(const std::vector<uint32_t> &indices) {
    randomSamplingOutputBuffer.resize(numThreads);
    randomSamplingOutputBufferMemory.resize(numThreads);

    randomSamplingOutputHostBuffer.resize(numThreads);
    randomSamplingOutputHostBufferMemory.resize(numThreads);

    triangleCounterBuffer.resize(numThreads);
    triangleCounterBufferMemory.resize(numThreads);

    absOutputBuffer.resize(numThreads);
    absOutputBufferMemory.resize(numThreads);

    absOutputHostBuffer.resize(numThreads);
    absOutputHostBufferMemory.resize(numThreads);

    absWorkingBuffer.resize(numThreads);
    absWorkingBufferMemory.resize(numThreads);

    edgeSubdivOutputBuffer.resize(numThreads);
    edgeSubdivOutputBufferMemory.resize(numThreads);

    edgeSubdivOutputHostBuffer.resize(numThreads);
    edgeSubdivOutputHostBufferMemory.resize(numThreads);

    edgeSubdivWorkingBuffer.resize(numThreads);
    edgeSubdivWorkingBufferMemory.resize(numThreads);

    haltonPointsBuffer.resize(numThreads);
    haltonPointsBufferMemory.resize(numThreads);

    randomSamplingOutputPointer.resize(numThreads);
    absOutputPointer.resize(numThreads);
    edgeSubdivOutputPointer.resize(numThreads);

    // Random sampling buffers
    //VkDeviceSize randomSamplingOutputBufferSize = sizeof(Sample) * (std::max(RAYS_PER_ITERATION, MAX_ABS_TRIANGLES_PER_ITERATION) * 9 * (std::pow(2, MAX_SUBDIVISION_STEPS) - 1) * 2 + 1);
    VkDeviceSize randomSamplingOutputBufferSize = sizeof(Sample) * RAYS_PER_ITERATION;
    VkDeviceSize absOutputBufferSize = sizeof(Sample) * MAX_ABS_TRIANGLES_PER_ITERATION * NUM_ABS_SAMPLES * NUM_REVERSE_SAMPLING_SAMPLES;
    //VkDeviceSize edgeSubdivOutputBufferSize = sizeof(Sample) * MAX_EDGE_SUBDIV_RAYS * (std::pow(2, MAX_SUBDIVISION_STEPS) - 1);
    VkDeviceSize edgeSubdivOutputBufferSize = sizeof(Sample) * MAX_ABS_TRIANGLES_PER_ITERATION * NUM_ABS_SAMPLES * (std::pow(2, MAX_SUBDIVISION_STEPS) - 1);
    for (int i = 0; i < numThreads; i++) {
        VulkanUtil::createBuffer(
            physicalDevice,
            logicalDevice, randomSamplingOutputBufferSize,
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_RAY_TRACING_BIT_NV | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            randomSamplingOutputBuffer[i], randomSamplingOutputBufferMemory[i], VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
        );
        VulkanUtil::createBuffer(
            physicalDevice,
            logicalDevice, randomSamplingOutputBufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT, randomSamplingOutputHostBuffer[i], randomSamplingOutputHostBufferMemory[i],
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT
        );
        vkMapMemory(logicalDevice, randomSamplingOutputHostBufferMemory[i], 0, randomSamplingOutputBufferSize, 0, &randomSamplingOutputPointer[i]);

        VulkanUtil::createBuffer(
            physicalDevice,
            logicalDevice, sizeof(unsigned int) * 2,
            VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_RAY_TRACING_BIT_NV | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            triangleCounterBuffer[i], triangleCounterBufferMemory[i], VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
        );

        // ABS buffers
        VulkanUtil::createBuffer(
            physicalDevice,
            logicalDevice, absOutputBufferSize,
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_RAY_TRACING_BIT_NV | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            absOutputBuffer[i], absOutputBufferMemory[i], VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
        );
        VulkanUtil::createBuffer(
            physicalDevice,
            logicalDevice, absOutputBufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT, absOutputHostBuffer[i], absOutputHostBufferMemory[i],
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT
        );
        vkMapMemory(logicalDevice, absOutputHostBufferMemory[i], 0, absOutputBufferSize, 0, &absOutputPointer[i]);
        VulkanUtil::createBuffer(
            physicalDevice,
            logicalDevice, sizeof(Sample) * MAX_ABS_TRIANGLES_PER_ITERATION,
            VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_RAY_TRACING_BIT_NV | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            absWorkingBuffer[i], absWorkingBufferMemory[i], VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
        );

        // Edge subdivision buffers
        VulkanUtil::createBuffer(
            physicalDevice,
            logicalDevice, edgeSubdivOutputBufferSize,
            //logicalDevice, sizeof(Sample) * MAX_EDGE_SUBDIV_RAYS * (std::pow(2, MAX_SUBDIVISION_STEPS) + 1),
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_RAY_TRACING_BIT_NV | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            edgeSubdivOutputBuffer[i], edgeSubdivOutputBufferMemory[i], VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
        );
        VulkanUtil::createBuffer(
            physicalDevice,
            logicalDevice, edgeSubdivOutputBufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT, edgeSubdivOutputHostBuffer[i], edgeSubdivOutputHostBufferMemory[i],
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT
        );
        vkMapMemory(logicalDevice, edgeSubdivOutputHostBufferMemory[i], 0, edgeSubdivOutputBufferSize, 0, &edgeSubdivOutputPointer[i]);
        VulkanUtil::createBuffer(
            physicalDevice,
            logicalDevice, sizeof(Sample) * MAX_EDGE_SUBDIV_RAYS * (std::pow(2, MAX_SUBDIVISION_STEPS) - 1),
            //logicalDevice, sizeof(Sample) * MAX_EDGE_SUBDIV_RAYS,
            VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_RAY_TRACING_BIT_NV | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            edgeSubdivWorkingBuffer[i], edgeSubdivWorkingBufferMemory[i], VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
        );

        // Create halton points buffer using GPU memory
        VulkanUtil::createBuffer(
            physicalDevice, logicalDevice, sizeof(haltonPoints[0][0]) * RAYS_PER_ITERATION,
            VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_RAY_TRACING_BIT_NV | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            haltonPointsBuffer[i], haltonPointsBufferMemory[i], VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
        );
    }

    VulkanUtil::createBuffer(
        physicalDevice,
        logicalDevice, sizeof(indices[0]) * indices.size(),
        VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
        pvsVisualizationBuffer, pvsVisualizationBufferMemory, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
    );
}

void VisibilityManager::createDescriptorSets(
    VkBuffer indexBuffer, VkBuffer vertexBuffer, const std::vector<VkBuffer> &uniformBuffers,
    int threadId
) {
    /*
    VkDescriptorSetAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = rtDescriptorPool;
    allocInfo.descriptorSetCount = 1;
    allocInfo.pSetLayouts = &rtDescriptorSetLayout;

    // Allocate descriptor sets
    if (vkAllocateDescriptorSets(
            logicalDevice, &allocInfo, &rtDescriptorSets
        ) != VK_SUCCESS
    ) {
        throw std::runtime_error("failed to allocate rt descriptor sets");
    }
    */

    std::array<VkWriteDescriptorSet, 9> descriptorWrites = {};

    VkWriteDescriptorSetAccelerationStructureNV asWriteInfo = {};
    asWriteInfo.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_NV;
    asWriteInfo.accelerationStructureCount = 1;
    asWriteInfo.pAccelerationStructures = &topLevelAS.as;
    descriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptorWrites[0].pNext = &asWriteInfo;
    descriptorWrites[0].dstSet = descriptorSet[threadId];
    descriptorWrites[0].dstBinding = 0;
    descriptorWrites[0].descriptorCount = 1;
    descriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_NV;

    VkDescriptorBufferInfo uniformBufferInfo = {};
    uniformBufferInfo.buffer = uniformBuffers[0];
    uniformBufferInfo.offset = 0;
    uniformBufferInfo.range = sizeof(UniformBufferObject);
    descriptorWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptorWrites[1].dstSet = descriptorSet[threadId];
    descriptorWrites[1].dstBinding = 1;
    descriptorWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    descriptorWrites[1].descriptorCount = 1;
    descriptorWrites[1].pBufferInfo = &uniformBufferInfo;

    VkDescriptorBufferInfo vertexBufferInfo = {};
    vertexBufferInfo.buffer = vertexBuffer;
    vertexBufferInfo.offset = 0;
    vertexBufferInfo.range = VK_WHOLE_SIZE;
    descriptorWrites[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptorWrites[2].dstSet = descriptorSet[threadId];
    descriptorWrites[2].dstBinding = 2;
    descriptorWrites[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    descriptorWrites[2].descriptorCount = 1;
    descriptorWrites[2].pBufferInfo = &vertexBufferInfo;

    VkDescriptorBufferInfo indexBufferInfo = {};
    indexBufferInfo.buffer = indexBuffer;
    indexBufferInfo.offset = 0;
    indexBufferInfo.range = VK_WHOLE_SIZE;
    descriptorWrites[3].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptorWrites[3].dstSet = descriptorSet[threadId];
    descriptorWrites[3].dstBinding = 3;
    descriptorWrites[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    descriptorWrites[3].descriptorCount = 1;
    descriptorWrites[3].pBufferInfo = &indexBufferInfo;

    VkDescriptorBufferInfo haltonPointsBufferInfo = {};
    haltonPointsBufferInfo.buffer = haltonPointsBuffer[threadId];
    haltonPointsBufferInfo.offset = 0;
    haltonPointsBufferInfo.range = VK_WHOLE_SIZE;
    descriptorWrites[4].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptorWrites[4].dstSet = descriptorSet[threadId];
    descriptorWrites[4].dstBinding = 4;
    descriptorWrites[4].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    descriptorWrites[4].descriptorCount = 1;
    descriptorWrites[4].pBufferInfo = &haltonPointsBufferInfo;

    VkDescriptorBufferInfo viewCellBufferInfo = {};
    viewCellBufferInfo.buffer = viewCellBuffer[threadId];
    viewCellBufferInfo.offset = 0;
    viewCellBufferInfo.range = VK_WHOLE_SIZE;
    descriptorWrites[5].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptorWrites[5].dstSet = descriptorSet[threadId];
    descriptorWrites[5].dstBinding = 5;
    descriptorWrites[5].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    descriptorWrites[5].descriptorCount = 1;
    descriptorWrites[5].pBufferInfo = &viewCellBufferInfo;

    VkDescriptorBufferInfo randomSamplingBufferInfo = {};
    randomSamplingBufferInfo.buffer = randomSamplingOutputBuffer[threadId];
    randomSamplingBufferInfo.offset = 0;
    randomSamplingBufferInfo.range = VK_WHOLE_SIZE;
    descriptorWrites[6].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptorWrites[6].dstSet = descriptorSet[threadId];
    descriptorWrites[6].dstBinding = 6;
    descriptorWrites[6].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    descriptorWrites[6].descriptorCount = 1;
    descriptorWrites[6].pBufferInfo = &randomSamplingBufferInfo;

    VkDescriptorBufferInfo trianglesBufferInfo = {};
    trianglesBufferInfo.buffer = absWorkingBuffer[threadId];
    trianglesBufferInfo.offset = 0;
    trianglesBufferInfo.range = VK_WHOLE_SIZE;
    descriptorWrites[7].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptorWrites[7].dstSet = descriptorSet[threadId];
    descriptorWrites[7].dstBinding = 7;
    descriptorWrites[7].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    descriptorWrites[7].descriptorCount = 1;
    descriptorWrites[7].pBufferInfo = &trianglesBufferInfo;

    VkDescriptorBufferInfo triangleCounterBufferInfo = {};
    triangleCounterBufferInfo.buffer = triangleCounterBuffer[threadId];
    triangleCounterBufferInfo.offset = 0;
    triangleCounterBufferInfo.range = VK_WHOLE_SIZE;
    descriptorWrites[8].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptorWrites[8].dstSet = descriptorSet[threadId];
    descriptorWrites[8].dstBinding = 8;
    descriptorWrites[8].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    descriptorWrites[8].descriptorCount = 1;
    descriptorWrites[8].pBufferInfo = &triangleCounterBufferInfo;

    vkUpdateDescriptorSets(
        logicalDevice,
        static_cast<uint32_t>(descriptorWrites.size()),
        descriptorWrites.data(),
        0,
        VK_NULL_HANDLE
    );


}

void VisibilityManager::initRayTracing(
    VkBuffer indexBuffer, VkBuffer vertexBuffer, const std::vector<uint32_t> &indices,
    const std::vector<Vertex> &vertices, const std::vector<VkBuffer> &uniformBuffers
) {
    rayTracingProperties = {};
    rayTracingProperties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PROPERTIES_NV;

    VkPhysicalDeviceProperties2 deviceProperties = {};
    deviceProperties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    deviceProperties.pNext = &rayTracingProperties;
    vkGetPhysicalDeviceProperties2(physicalDevice, &deviceProperties);

    // Get function pointers
    vkCreateAccelerationStructureNV = reinterpret_cast<PFN_vkCreateAccelerationStructureNV>(vkGetDeviceProcAddr(logicalDevice, "vkCreateAccelerationStructureNV"));
    vkDestroyAccelerationStructureNV = reinterpret_cast<PFN_vkDestroyAccelerationStructureNV>(vkGetDeviceProcAddr(logicalDevice, "vkDestroyAccelerationStructureNV"));
    vkBindAccelerationStructureMemoryNV = reinterpret_cast<PFN_vkBindAccelerationStructureMemoryNV>(vkGetDeviceProcAddr(logicalDevice, "vkBindAccelerationStructureMemoryNV"));
    vkGetAccelerationStructureHandleNV = reinterpret_cast<PFN_vkGetAccelerationStructureHandleNV>(vkGetDeviceProcAddr(logicalDevice, "vkGetAccelerationStructureHandleNV"));
    vkGetAccelerationStructureMemoryRequirementsNV = reinterpret_cast<PFN_vkGetAccelerationStructureMemoryRequirementsNV>(vkGetDeviceProcAddr(logicalDevice, "vkGetAccelerationStructureMemoryRequirementsNV"));
    vkCmdBuildAccelerationStructureNV = reinterpret_cast<PFN_vkCmdBuildAccelerationStructureNV>(vkGetDeviceProcAddr(logicalDevice, "vkCmdBuildAccelerationStructureNV"));
    vkCreateRayTracingPipelinesNV = reinterpret_cast<PFN_vkCreateRayTracingPipelinesNV>(vkGetDeviceProcAddr(logicalDevice, "vkCreateRayTracingPipelinesNV"));
    vkGetRayTracingShaderGroupHandlesNV = reinterpret_cast<PFN_vkGetRayTracingShaderGroupHandlesNV>(vkGetDeviceProcAddr(logicalDevice, "vkGetRayTracingShaderGroupHandlesNV"));
    vkCmdTraceRaysNV = reinterpret_cast<PFN_vkCmdTraceRaysNV>(vkGetDeviceProcAddr(logicalDevice, "vkCmdTraceRaysNV"));

    VkGeometryNV geometry = {};
    geometry.sType = VK_STRUCTURE_TYPE_GEOMETRY_NV;
    geometry.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_NV;
    geometry.geometry.triangles.sType = VK_STRUCTURE_TYPE_GEOMETRY_TRIANGLES_NV;
    geometry.geometry.triangles.vertexData = vertexBuffer;
    geometry.geometry.triangles.vertexOffset = 0;
    geometry.geometry.triangles.vertexCount = vertices.size();
    geometry.geometry.triangles.vertexStride = sizeof(Vertex);
    geometry.geometry.triangles.vertexFormat =VK_FORMAT_R32G32B32_SFLOAT;
    geometry.geometry.triangles.indexData = indexBuffer;
    geometry.geometry.triangles.indexOffset = 0;
    geometry.geometry.triangles.indexCount = indices.size();
    geometry.geometry.triangles.indexType = VK_INDEX_TYPE_UINT32;
    geometry.geometry.triangles.transformData = VK_NULL_HANDLE;
    geometry.geometry.triangles.transformOffset = 0;
    geometry.geometry.aabbs = {};
    geometry.geometry.aabbs.sType = { VK_STRUCTURE_TYPE_GEOMETRY_AABB_NV };
    geometry.flags = VK_GEOMETRY_OPAQUE_BIT_NV;

    createBottomLevelAS(&geometry);

    VkBuffer instanceBuffer;
    VkDeviceMemory instanceBufferMemory;

    /*
    glm::mat4x4 m = glm::rotate(
        glm::mat4(1.0f),
        glm::radians(0.0f),
        glm::vec3(0.0f, 1.0f, 0.0f)
    );
    */

    glm::mat4x4 model = glm::translate(
        glm::mat4(1.0f),
        glm::vec3(0.0f, 0.0f, 0.0f) * 0.5f
    );
    model = glm::transpose(model);

    glm::mat3x4 transform = {
        1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f, 0.0f,
    };
    memcpy(&transform, &model, sizeof(transform));

    GeometryInstance geometryInstance = {};
    geometryInstance.transform = transform;
    geometryInstance.instanceId = 0;
    geometryInstance.mask = 0xff;
    geometryInstance.instanceOffset = 0;
    geometryInstance.flags = VK_GEOMETRY_INSTANCE_TRIANGLE_CULL_DISABLE_BIT_NV;
    geometryInstance.accelerationStructureHandle = bottomLevelAS.handle;

    // Upload instance descriptions to the device
    VkDeviceSize instanceBufferSize = sizeof(GeometryInstance);
    VulkanUtil::createBuffer(
        physicalDevice,
        logicalDevice, instanceBufferSize,
        VK_BUFFER_USAGE_RAY_TRACING_BIT_NV,
        instanceBuffer,
        instanceBufferMemory,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
    );
    void *data;
    vkMapMemory(logicalDevice, instanceBufferMemory, 0, instanceBufferSize, 0, &data);
    memcpy(data, &geometryInstance, instanceBufferSize);
    vkUnmapMemory(logicalDevice, instanceBufferMemory);

    createTopLevelAS();

    // Build acceleration structures
    buildAS(instanceBuffer, &geometry);

    vkDestroyBuffer(logicalDevice, instanceBuffer, nullptr);
    vkFreeMemory(logicalDevice, instanceBufferMemory, nullptr);

    createCommandBuffers();
    createDescriptorPool();
    createDescriptorSetLayout();
    createABSDescriptorSetLayout();
    createEdgeSubdivDescriptorSetLayout();

    descriptorSet.resize(numThreads);
    descriptorSetABS.resize(numThreads);
    descriptorSetEdgeSubdiv.resize(numThreads);
    for (int i = 0; i < numThreads; i++) {   // TODO: Cleanup
        std::array<VkDescriptorSetLayout, 3> d = {
            descriptorSetLayout, descriptorSetLayoutABS, descriptorSetLayoutEdgeSubdiv
        };
        VkDescriptorSetAllocateInfo allocInfo = {};
        allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocInfo.descriptorPool = descriptorPool;
        allocInfo.descriptorSetCount = 3;
        allocInfo.pSetLayouts = d.data();

        // Allocate descriptor sets
        std::array<VkDescriptorSet, 3> dd = {
            descriptorSet[i], descriptorSetABS[i], descriptorSetEdgeSubdiv[i]
        };
        if (vkAllocateDescriptorSets(logicalDevice, &allocInfo, dd.data()) != VK_SUCCESS) {
            throw std::runtime_error("failed to allocate descriptor sets");
        }
        descriptorSet[i] = dd[0];
        descriptorSetABS[i] = dd[1];
        descriptorSetEdgeSubdiv[i] = dd[2];
    }

    for (int i = 0; i < numThreads; i++) {
        createDescriptorSets(indexBuffer, vertexBuffer, uniformBuffers, i);
    }
    createRandomSamplingPipeline();
    createShaderBindingTable(shaderBindingTable, shaderBindingTableMemory, pipeline);

    for (int i = 0; i < numThreads; i++) {
        createABSDescriptorSets(vertexBuffer, i);
    }
    createABSPipeline();
    createShaderBindingTable(shaderBindingTableABS, shaderBindingTableMemoryABS, pipelineABS);

    for (int i = 0; i < numThreads; i++) {
        createEdgeSubdivDescriptorSets(i);
    }
    createEdgeSubdivPipeline();
    createShaderBindingTable(shaderBindingTableEdgeSubdiv, shaderBindingTableMemoryEdgeSubdiv, pipelineEdgeSubdiv);

    // Calculate shader binding offsets
    bindingOffsetRayGenShader = rayTracingProperties.shaderGroupHandleSize * RT_SHADER_INDEX_RAYGEN;
    bindingOffsetMissShader = rayTracingProperties.shaderGroupHandleSize * RT_SHADER_INDEX_MISS;
    bindingOffsetHitShader = rayTracingProperties.shaderGroupHandleSize * RT_SHADER_INDEX_CLOSEST_HIT;
    bindingStride = rayTracingProperties.shaderGroupHandleSize;
}

void VisibilityManager::createBottomLevelAS(const VkGeometryNV *geometry) {
    // The bottom level acceleration structure contains the scene's geometry

    VkAccelerationStructureInfoNV accelerationStructureInfo = {};
    accelerationStructureInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_INFO_NV;
    accelerationStructureInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_NV;
    accelerationStructureInfo.instanceCount = 0;
    accelerationStructureInfo.geometryCount = 1;
    accelerationStructureInfo.pGeometries = geometry;
    accelerationStructureInfo.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_NV;

    VkAccelerationStructureCreateInfoNV accelerationStructureCI = {};
    accelerationStructureCI.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_NV;
    accelerationStructureCI.info = accelerationStructureInfo;
    vkCreateAccelerationStructureNV(logicalDevice, &accelerationStructureCI, nullptr, &bottomLevelAS.as);

    VkAccelerationStructureMemoryRequirementsInfoNV memoryRequirementsInfo = {};
    memoryRequirementsInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_MEMORY_REQUIREMENTS_INFO_NV;
    memoryRequirementsInfo.type = VK_ACCELERATION_STRUCTURE_MEMORY_REQUIREMENTS_TYPE_OBJECT_NV;
    memoryRequirementsInfo.accelerationStructure = bottomLevelAS.as;

    VkMemoryRequirements2 memoryRequirements2 = {};
    vkGetAccelerationStructureMemoryRequirementsNV(logicalDevice, &memoryRequirementsInfo, &memoryRequirements2);

    VkMemoryAllocateInfo memoryAllocateInfo = {};
    memoryAllocateInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    memoryAllocateInfo.allocationSize = memoryRequirements2.memoryRequirements.size;
    memoryAllocateInfo.memoryTypeIndex = VulkanUtil::findMemoryType(physicalDevice, memoryRequirements2.memoryRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT); //deviceLocalMemoryIndex;
    vkAllocateMemory(logicalDevice, &memoryAllocateInfo, nullptr, &bottomLevelAS.deviceMemory);

    VkBindAccelerationStructureMemoryInfoNV accelerationStructureMemoryInfo = {};
    accelerationStructureMemoryInfo.sType = VK_STRUCTURE_TYPE_BIND_ACCELERATION_STRUCTURE_MEMORY_INFO_NV;
    accelerationStructureMemoryInfo.accelerationStructure = bottomLevelAS.as;
    accelerationStructureMemoryInfo.memory = bottomLevelAS.deviceMemory;
    vkBindAccelerationStructureMemoryNV(logicalDevice, 1, &accelerationStructureMemoryInfo);

    vkGetAccelerationStructureHandleNV(logicalDevice, bottomLevelAS.as, sizeof(uint64_t), &bottomLevelAS.handle);
}

void VisibilityManager::createTopLevelAS() {
    VkAccelerationStructureInfoNV accelerationStructureInfo = {};
    accelerationStructureInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_INFO_NV;
    accelerationStructureInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_NV;
    accelerationStructureInfo.instanceCount = 1;
    accelerationStructureInfo.geometryCount = 0;
    accelerationStructureInfo.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_NV;

    VkAccelerationStructureCreateInfoNV accelerationStructureCI = {};
    accelerationStructureCI.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_NV;
    accelerationStructureCI.info = accelerationStructureInfo;
    vkCreateAccelerationStructureNV(logicalDevice, &accelerationStructureCI, nullptr, &topLevelAS.as);

    VkAccelerationStructureMemoryRequirementsInfoNV memoryRequirementsInfo = {};
    memoryRequirementsInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_MEMORY_REQUIREMENTS_INFO_NV;
    memoryRequirementsInfo.type = VK_ACCELERATION_STRUCTURE_MEMORY_REQUIREMENTS_TYPE_OBJECT_NV;
    memoryRequirementsInfo.accelerationStructure = topLevelAS.as;

    VkMemoryRequirements2 memoryRequirements2 = {};
    vkGetAccelerationStructureMemoryRequirementsNV(logicalDevice, &memoryRequirementsInfo, &memoryRequirements2);

    VkMemoryAllocateInfo memoryAllocateInfo = {};
    memoryAllocateInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    memoryAllocateInfo.allocationSize = memoryRequirements2.memoryRequirements.size;
    memoryAllocateInfo.memoryTypeIndex = VulkanUtil::findMemoryType(physicalDevice, memoryRequirements2.memoryRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT); //deviceLocalMemoryIndex;
    vkAllocateMemory(logicalDevice, &memoryAllocateInfo, nullptr, &topLevelAS.deviceMemory);

    VkBindAccelerationStructureMemoryInfoNV accelerationStructureMemoryInfo = {};
    accelerationStructureMemoryInfo.sType = VK_STRUCTURE_TYPE_BIND_ACCELERATION_STRUCTURE_MEMORY_INFO_NV;
    accelerationStructureMemoryInfo.accelerationStructure = topLevelAS.as;
    accelerationStructureMemoryInfo.memory = topLevelAS.deviceMemory;
    vkBindAccelerationStructureMemoryNV(logicalDevice, 1, &accelerationStructureMemoryInfo);

    vkGetAccelerationStructureHandleNV(logicalDevice, topLevelAS.as, sizeof(uint64_t), &topLevelAS.handle);
}

void VisibilityManager::buildAS(const VkBuffer instanceBuffer, const VkGeometryNV *geometry) {
    // Acceleration structure build requires some scratch space to store temporary information
    VkAccelerationStructureMemoryRequirementsInfoNV memoryRequirementsInfo = {};
    memoryRequirementsInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_MEMORY_REQUIREMENTS_INFO_NV;
    memoryRequirementsInfo.type = VK_ACCELERATION_STRUCTURE_MEMORY_REQUIREMENTS_TYPE_BUILD_SCRATCH_NV;

    // Memory requirement for the bottom level acceleration structure
    VkMemoryRequirements2 memReqBottomLevelAS;
    memoryRequirementsInfo.accelerationStructure = bottomLevelAS.as;
    vkGetAccelerationStructureMemoryRequirementsNV(logicalDevice, &memoryRequirementsInfo, &memReqBottomLevelAS);

    // Memory requirement for the top level acceleration structure
    VkMemoryRequirements2 memReqTopLevelAS;
    memoryRequirementsInfo.accelerationStructure = topLevelAS.as;
    vkGetAccelerationStructureMemoryRequirementsNV(logicalDevice, &memoryRequirementsInfo, &memReqTopLevelAS);

    // Create temporary buffer
    const VkDeviceSize tempBufferSize = std::max(memReqBottomLevelAS.memoryRequirements.size, memReqTopLevelAS.memoryRequirements.size);
    VkBuffer tempBuffer = {};
    VkDeviceMemory tempBufferMemory = {};
    VulkanUtil::createBuffer(
        physicalDevice,
        logicalDevice, tempBufferSize,
        VK_BUFFER_USAGE_RAY_TRACING_BIT_NV,
        tempBuffer,
        tempBufferMemory,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
    );

    VkCommandBuffer commandBuffer = VulkanUtil::beginSingleTimeCommands(logicalDevice, commandPool[0]);

    // Build bottom level acceleration structure
    VkAccelerationStructureInfoNV buildInfo = {};
    buildInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_INFO_NV;
    buildInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_NV;
    buildInfo.geometryCount = 1;
    buildInfo.pGeometries = geometry;
    buildInfo.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_NV;
    vkCmdBuildAccelerationStructureNV(
        commandBuffer,
        &buildInfo,
        VK_NULL_HANDLE,
        0,
        VK_FALSE,
        bottomLevelAS.as,
        VK_NULL_HANDLE,
        tempBuffer,
        0
    );
    VkMemoryBarrier memoryBarrier = {};
    memoryBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    memoryBarrier.srcAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_NV | VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_NV;
    memoryBarrier.dstAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_NV | VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_NV;
    vkCmdPipelineBarrier(
        commandBuffer,
        VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_NV,
        VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_NV,
        0,
        1,
        &memoryBarrier,
        0,
        0,
        0,
        0
    );

    // Build top level acceleration structure
    buildInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_NV;
    buildInfo.pGeometries = 0;
    buildInfo.geometryCount = 0;
    buildInfo.instanceCount = 1;
    buildInfo.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_NV;
    vkCmdBuildAccelerationStructureNV(
        commandBuffer,
        &buildInfo,
        instanceBuffer,
        0,
        VK_FALSE,
        topLevelAS.as,
        VK_NULL_HANDLE,
        tempBuffer,
        0
    );
    vkCmdPipelineBarrier(
        commandBuffer,
        VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_NV,
        VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_NV,
        0,
        1,
        &memoryBarrier,
        0,
        0,
        0,
        0
    );

    VulkanUtil::endSingleTimeCommands(
        logicalDevice, commandBuffer, commandPool[0], computeQueue, queueSubmitMutex
    );

    vkDestroyBuffer(logicalDevice, tempBuffer, nullptr);
    vkFreeMemory(logicalDevice, tempBufferMemory, nullptr);
}

void VisibilityManager::createDescriptorSetLayout() {
    // Top level acceleration structure binding
    VkDescriptorSetLayoutBinding aslayoutBinding = {};
    aslayoutBinding.binding = 0;
    aslayoutBinding.descriptorCount = 1;
    aslayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_NV;
    aslayoutBinding.stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_NV;

    // Uniform binding
    VkDescriptorSetLayoutBinding uniformLayoutBinding = {};
    uniformLayoutBinding.binding = 1;
    uniformLayoutBinding.descriptorCount = 1;
    uniformLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    uniformLayoutBinding.stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_NV | VK_SHADER_STAGE_CLOSEST_HIT_BIT_NV;

    // Vertex array binding
    VkDescriptorSetLayoutBinding vertexLayoutBinding = {};
    vertexLayoutBinding.binding = 2;
    vertexLayoutBinding.descriptorCount = 1;
    vertexLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    vertexLayoutBinding.stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_NV | VK_SHADER_STAGE_CLOSEST_HIT_BIT_NV;

    // Index array binding
    VkDescriptorSetLayoutBinding indexLayoutBinding = {};
    indexLayoutBinding.binding = 3;
    indexLayoutBinding.descriptorCount = 1;
    indexLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    indexLayoutBinding.stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_NV | VK_SHADER_STAGE_CLOSEST_HIT_BIT_NV;

    // Halton points binding
    VkDescriptorSetLayoutBinding haltonPointsBinding = {};
    haltonPointsBinding.binding = 4;
    haltonPointsBinding.descriptorCount = 1;
    haltonPointsBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    haltonPointsBinding.stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_NV;

    // View cell uniform binding
    VkDescriptorSetLayoutBinding viewCellBinding = {};
    viewCellBinding.binding = 5;
    viewCellBinding.descriptorCount = 1;
    viewCellBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    viewCellBinding.stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_NV;

    // Random sampling sample output buffer binding
    VkDescriptorSetLayoutBinding randomSamplingOutputBinding = {};
    randomSamplingOutputBinding.binding = 6;
    randomSamplingOutputBinding.descriptorCount = 1;
    randomSamplingOutputBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    randomSamplingOutputBinding.stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_NV;

    // Triangle buffer
    VkDescriptorSetLayoutBinding triangleBufferBinding = {};
    triangleBufferBinding.binding = 7;
    triangleBufferBinding.descriptorCount = 1;
    triangleBufferBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    triangleBufferBinding.stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_NV;

    // Triangle counter buffer
    VkDescriptorSetLayoutBinding triangleCounterBufferBinding = {};
    triangleCounterBufferBinding.binding = 8;
    triangleCounterBufferBinding.descriptorCount = 1;
    triangleCounterBufferBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    triangleCounterBufferBinding.stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_NV;

    std::array<VkDescriptorSetLayoutBinding, 9> bindings = {
        aslayoutBinding,
        uniformLayoutBinding,
        vertexLayoutBinding,
        indexLayoutBinding,
        haltonPointsBinding,
        viewCellBinding,
        randomSamplingOutputBinding,
        triangleBufferBinding,
        triangleCounterBufferBinding
    };
    VkDescriptorSetLayoutCreateInfo layoutInfo = {};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
    layoutInfo.pBindings = bindings.data();

    if (vkCreateDescriptorSetLayout(
            logicalDevice, &layoutInfo, nullptr, &descriptorSetLayout
        ) != VK_SUCCESS
    ) {
        throw std::runtime_error("failed to create rt descriptor set layout");
    }

    pushConstantRange = {};
    pushConstantRange.stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_NV;
    pushConstantRange.size = sizeof(int) * 7;
    pushConstantRange.offset = 0;
}

void VisibilityManager::createDescriptorPool() {
    std::array<VkDescriptorPoolSize, 4> poolSizes = {};
    poolSizes[0].type = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_NV;
    poolSizes[0].descriptorCount = 1;
    poolSizes[1].type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    poolSizes[1].descriptorCount = 1;
    poolSizes[2].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    poolSizes[2].descriptorCount = 2;
    poolSizes[3].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    poolSizes[3].descriptorCount = 9;

    VkDescriptorPoolCreateInfo poolInfo = {};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
    poolInfo.pPoolSizes = poolSizes.data();
    poolInfo.maxSets = 3 * numThreads;

    if (vkCreateDescriptorPool(logicalDevice, &poolInfo, nullptr, &descriptorPool) != VK_SUCCESS) {
        throw std::runtime_error("failed to create rt descriptor pool");
    }
}

void VisibilityManager::createPipeline(
    std::array<VkPipelineShaderStageCreateInfo, 3> shaderStages, VkPipelineLayout *pipelineLayout,
    VkPipeline *pipeline, std::vector<VkDescriptorSetLayout> descriptorSetLayouts,
    std::vector<VkPushConstantRange> pushConstantRanges
) {
    VkPipelineLayoutCreateInfo pipelineLayoutInfo = {};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = static_cast<uint32_t>(descriptorSetLayouts.size());
    pipelineLayoutInfo.pSetLayouts = descriptorSetLayouts.data();
    pipelineLayoutInfo.pushConstantRangeCount = static_cast<uint32_t>(pushConstantRanges.size());
    pipelineLayoutInfo.pPushConstantRanges = pushConstantRanges.data();
    if (vkCreatePipelineLayout(
            logicalDevice, &pipelineLayoutInfo, nullptr, pipelineLayout
        ) != VK_SUCCESS
    ) {
        throw std::runtime_error("failed to create rt pipeline layout");
    }

    // Setup shader groups
    std::array<VkRayTracingShaderGroupCreateInfoNV, 3> groups = {};
    for (auto &group : groups) {
        group.sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_NV;
        group.generalShader = VK_SHADER_UNUSED_NV;
        group.closestHitShader = VK_SHADER_UNUSED_NV;
        group.anyHitShader = VK_SHADER_UNUSED_NV;
        group.intersectionShader = VK_SHADER_UNUSED_NV;
    }
    groups[RT_SHADER_INDEX_RAYGEN].type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_NV;
    groups[RT_SHADER_INDEX_RAYGEN].generalShader = RT_SHADER_INDEX_RAYGEN;
    groups[RT_SHADER_INDEX_MISS].type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_NV;
    groups[RT_SHADER_INDEX_MISS].generalShader = RT_SHADER_INDEX_MISS;
    groups[RT_SHADER_INDEX_CLOSEST_HIT].type = VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_NV;
    groups[RT_SHADER_INDEX_CLOSEST_HIT].generalShader = VK_SHADER_UNUSED_NV;
    groups[RT_SHADER_INDEX_CLOSEST_HIT].closestHitShader = RT_SHADER_INDEX_CLOSEST_HIT;

    // Setup ray tracing pipeline
    VkRayTracingPipelineCreateInfoNV rtPipelineInfo = {};
    rtPipelineInfo.sType = VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_NV;
    rtPipelineInfo.stageCount = static_cast<uint32_t>(shaderStages.size());
    rtPipelineInfo.pStages = shaderStages.data();
    rtPipelineInfo.groupCount = static_cast<uint32_t>(groups.size());
    rtPipelineInfo.pGroups = groups.data();
    rtPipelineInfo.maxRecursionDepth = 1;
    rtPipelineInfo.layout = *pipelineLayout;
    if (vkCreateRayTracingPipelinesNV(
            logicalDevice, VK_NULL_HANDLE, 1, &rtPipelineInfo, nullptr, pipeline
        ) != VK_SUCCESS
    ) {
        throw std::runtime_error("failed to create rt pipeline");
    }
}

void VisibilityManager::createRandomSamplingPipeline() {
    // Load shaders
    VkPipelineShaderStageCreateInfo rayGenShaderStageInfo = {};
    rayGenShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    rayGenShaderStageInfo.stage = VK_SHADER_STAGE_RAYGEN_BIT_NV;
    rayGenShaderStageInfo.module = VulkanUtil::createShader(logicalDevice, "shaders/rt/raytrace.rgen.spv");
    rayGenShaderStageInfo.pName = "main";

    VkPipelineShaderStageCreateInfo rayClosestHitShaderStageInfo = {};
    rayClosestHitShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    rayClosestHitShaderStageInfo.stage = VK_SHADER_STAGE_CLOSEST_HIT_BIT_NV;
    rayClosestHitShaderStageInfo.module = VulkanUtil::createShader(logicalDevice, "shaders/rt/raytrace.rchit.spv");
    rayClosestHitShaderStageInfo.pName = "main";

    VkPipelineShaderStageCreateInfo rayMissShaderStageInfo = {};
    rayMissShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    rayMissShaderStageInfo.stage = VK_SHADER_STAGE_MISS_BIT_NV;
    rayMissShaderStageInfo.module = VulkanUtil::createShader(logicalDevice, "shaders/rt/raytrace.rmiss.spv");
    rayMissShaderStageInfo.pName = "main";

    std::array<VkPipelineShaderStageCreateInfo, 3> shaderStages = {};
    shaderStages[RT_SHADER_INDEX_RAYGEN] = rayGenShaderStageInfo;
    shaderStages[RT_SHADER_INDEX_CLOSEST_HIT] = rayClosestHitShaderStageInfo;
    shaderStages[RT_SHADER_INDEX_MISS] = rayMissShaderStageInfo;

    createPipeline(shaderStages, &pipelineLayout, &pipeline, { descriptorSetLayout }, { pushConstantRange });

    vkDestroyShaderModule(logicalDevice, rayGenShaderStageInfo.module, nullptr);
    vkDestroyShaderModule(logicalDevice, rayClosestHitShaderStageInfo.module, nullptr);
    vkDestroyShaderModule(logicalDevice, rayMissShaderStageInfo.module, nullptr);
}

void VisibilityManager::createABSPipeline() {
    // Load shaders
    VkPipelineShaderStageCreateInfo rayGenShaderStageInfo = {};
    rayGenShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    rayGenShaderStageInfo.stage = VK_SHADER_STAGE_RAYGEN_BIT_NV;
    rayGenShaderStageInfo.module = VulkanUtil::createShader(logicalDevice, "shaders/rt/raytrace_abs.rgen.spv");
    rayGenShaderStageInfo.pName = "main";

    VkPipelineShaderStageCreateInfo rayClosestHitShaderStageInfo = {};
    rayClosestHitShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    rayClosestHitShaderStageInfo.stage = VK_SHADER_STAGE_CLOSEST_HIT_BIT_NV;
    rayClosestHitShaderStageInfo.module = VulkanUtil::createShader(logicalDevice, "shaders/rt/raytrace_abs.rchit.spv");
    rayClosestHitShaderStageInfo.pName = "main";

    VkPipelineShaderStageCreateInfo rayMissShaderStageInfo = {};
    rayMissShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    rayMissShaderStageInfo.stage = VK_SHADER_STAGE_MISS_BIT_NV;
    rayMissShaderStageInfo.module = VulkanUtil::createShader(logicalDevice, "shaders/rt/raytrace_abs.rmiss.spv");
    rayMissShaderStageInfo.pName = "main";

    std::array<VkPipelineShaderStageCreateInfo, 3> shaderStages = {};
    shaderStages[RT_SHADER_INDEX_RAYGEN] = rayGenShaderStageInfo;
    shaderStages[RT_SHADER_INDEX_CLOSEST_HIT] = rayClosestHitShaderStageInfo;
    shaderStages[RT_SHADER_INDEX_MISS] = rayMissShaderStageInfo;

    createPipeline(
        shaderStages, &pipelineABSLayout, &pipelineABS,
        { descriptorSetLayout, descriptorSetLayoutABS }, { }
    );

    vkDestroyShaderModule(logicalDevice, rayGenShaderStageInfo.module, nullptr);
    vkDestroyShaderModule(logicalDevice, rayClosestHitShaderStageInfo.module, nullptr);
    vkDestroyShaderModule(logicalDevice, rayMissShaderStageInfo.module, nullptr);
}

void VisibilityManager::createEdgeSubdivPipeline() {
    // Load shaders
    VkPipelineShaderStageCreateInfo rayGenShaderStageInfo = {};
    rayGenShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    rayGenShaderStageInfo.stage = VK_SHADER_STAGE_RAYGEN_BIT_NV;
    if (USE_EDGE_SUBDIV_CPU) {
        rayGenShaderStageInfo.module = VulkanUtil::createShader(logicalDevice, "shaders/rt/raytrace_subdiv_cpu_calc.rgen.spv");
    } else {
        rayGenShaderStageInfo.module = VulkanUtil::createShader(logicalDevice, "shaders/rt/raytrace_subdiv.rgen.spv");
    }
    rayGenShaderStageInfo.pName = "main";

    VkPipelineShaderStageCreateInfo rayClosestHitShaderStageInfo = {};
    rayClosestHitShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    rayClosestHitShaderStageInfo.stage = VK_SHADER_STAGE_CLOSEST_HIT_BIT_NV;
    rayClosestHitShaderStageInfo.module = VulkanUtil::createShader(logicalDevice, "shaders/rt/raytrace_abs.rchit.spv");
    rayClosestHitShaderStageInfo.pName = "main";

    VkPipelineShaderStageCreateInfo rayMissShaderStageInfo = {};
    rayMissShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    rayMissShaderStageInfo.stage = VK_SHADER_STAGE_MISS_BIT_NV;
    rayMissShaderStageInfo.module = VulkanUtil::createShader(logicalDevice, "shaders/rt/raytrace_abs.rmiss.spv");
    rayMissShaderStageInfo.pName = "main";

    std::array<VkPipelineShaderStageCreateInfo, 3> shaderStages = {};
    shaderStages[RT_SHADER_INDEX_RAYGEN] = rayGenShaderStageInfo;
    shaderStages[RT_SHADER_INDEX_CLOSEST_HIT] = rayClosestHitShaderStageInfo;
    shaderStages[RT_SHADER_INDEX_MISS] = rayMissShaderStageInfo;

    createPipeline(
        shaderStages, &pipelineEdgeSubdivLayout, &pipelineEdgeSubdiv,
        { descriptorSetLayout, descriptorSetLayoutABS, descriptorSetLayoutEdgeSubdiv }, { }
    );

    vkDestroyShaderModule(logicalDevice, rayGenShaderStageInfo.module, nullptr);
    vkDestroyShaderModule(logicalDevice, rayClosestHitShaderStageInfo.module, nullptr);
    vkDestroyShaderModule(logicalDevice, rayMissShaderStageInfo.module, nullptr);
}

void VisibilityManager::createEdgeSubdivDescriptorSetLayout() {
    VkDescriptorSetLayoutBinding edgeSubdivOutputBinding = {};
    edgeSubdivOutputBinding.binding = 0;
    edgeSubdivOutputBinding.descriptorCount = 1;
    edgeSubdivOutputBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    edgeSubdivOutputBinding.stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_NV;

    VkDescriptorSetLayoutBinding edgeSubdivWorkingBufferBinding = {};
    edgeSubdivWorkingBufferBinding.binding = 1;
    edgeSubdivWorkingBufferBinding.descriptorCount = 1;
    edgeSubdivWorkingBufferBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    edgeSubdivWorkingBufferBinding.stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_NV;

    std::array<VkDescriptorSetLayoutBinding, 2> bindings = {
        edgeSubdivOutputBinding,
        edgeSubdivWorkingBufferBinding
    };
    VkDescriptorSetLayoutCreateInfo layoutInfo = {};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
    layoutInfo.pBindings = bindings.data();

    if (vkCreateDescriptorSetLayout(
            logicalDevice, &layoutInfo, nullptr, &descriptorSetLayoutEdgeSubdiv
        ) != VK_SUCCESS
    ) {
        throw std::runtime_error("failed to create rt descriptor set layout edge subdivision");
    }
}

void VisibilityManager::createABSDescriptorSetLayout() {
    VkDescriptorSetLayoutBinding triangleOutputBinding = {};
    triangleOutputBinding.binding = 0;
    triangleOutputBinding.descriptorCount = 1;
    triangleOutputBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    triangleOutputBinding.stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_NV;

    VkDescriptorSetLayoutBinding absWorkingBufferBinding = {};
    absWorkingBufferBinding.binding = 1;
    absWorkingBufferBinding.descriptorCount = 1;
    absWorkingBufferBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    absWorkingBufferBinding.stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_NV;

    std::array<VkDescriptorSetLayoutBinding, 2> bindings = {
        triangleOutputBinding,
        absWorkingBufferBinding
    };
    VkDescriptorSetLayoutCreateInfo layoutInfo = {};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
    layoutInfo.pBindings = bindings.data();

    if (vkCreateDescriptorSetLayout(
            logicalDevice, &layoutInfo, nullptr, &descriptorSetLayoutABS
        ) != VK_SUCCESS
    ) {
        throw std::runtime_error("failed to create rt descriptor set layout ABS");
    }
}

void VisibilityManager::createEdgeSubdivDescriptorSets(int threadId) {
    std::array<VkWriteDescriptorSet, 2> descriptorWrites = {};

    VkDescriptorBufferInfo edgeSubdivOutputBufferInfo = {};        // TODO: Move descriptor set creation to method
    edgeSubdivOutputBufferInfo.buffer = edgeSubdivOutputBuffer[threadId];
    edgeSubdivOutputBufferInfo.offset = 0;
    edgeSubdivOutputBufferInfo.range = VK_WHOLE_SIZE;
    descriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptorWrites[0].dstSet = descriptorSetEdgeSubdiv[threadId];
    descriptorWrites[0].dstBinding = 0;
    descriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    descriptorWrites[0].descriptorCount = 1;
    descriptorWrites[0].pBufferInfo = &edgeSubdivOutputBufferInfo;

    VkDescriptorBufferInfo edgeSubdivWorkingBufferInfo = {};
    edgeSubdivWorkingBufferInfo.buffer = edgeSubdivWorkingBuffer[threadId];
    edgeSubdivWorkingBufferInfo.offset = 0;
    edgeSubdivWorkingBufferInfo.range = VK_WHOLE_SIZE;
    descriptorWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptorWrites[1].dstSet = descriptorSetEdgeSubdiv[threadId];
    descriptorWrites[1].dstBinding = 1;
    descriptorWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    descriptorWrites[1].descriptorCount = 1;
    descriptorWrites[1].pBufferInfo = &edgeSubdivWorkingBufferInfo;

    vkUpdateDescriptorSets(
        logicalDevice,
        static_cast<uint32_t>(descriptorWrites.size()),
        descriptorWrites.data(),
        0,
        VK_NULL_HANDLE
    );
}

void VisibilityManager::createABSDescriptorSets(VkBuffer vertexBuffer, int threadId) {
    /*
    VkDescriptorSetAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = rtDescriptorPool;
    allocInfo.descriptorSetCount = 1;
    allocInfo.pSetLayouts = &rtDescriptorSetLayoutABS;

    // Allocate descriptor sets
    if (vkAllocateDescriptorSets(
            logicalDevice, &allocInfo, &rtDescriptorSetsABS
        ) != VK_SUCCESS
    ) {
        throw std::runtime_error("failed to allocate rt descriptor sets ABS");
    }
    */

    std::array<VkWriteDescriptorSet, 2> descriptorWrites = {};

    VkDescriptorBufferInfo absOutputBufferInfo = {};
    absOutputBufferInfo.buffer = absOutputBuffer[threadId];
    absOutputBufferInfo.offset = 0;
    absOutputBufferInfo.range = VK_WHOLE_SIZE;
    descriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptorWrites[0].dstSet = descriptorSetABS[threadId];
    descriptorWrites[0].dstBinding = 0;
    descriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    descriptorWrites[0].descriptorCount = 1;
    descriptorWrites[0].pBufferInfo = &absOutputBufferInfo;

    VkDescriptorBufferInfo absWorkingBufferInfo = {};
    absWorkingBufferInfo.buffer = absWorkingBuffer[threadId];
    absWorkingBufferInfo.offset = 0;
    absWorkingBufferInfo.range = VK_WHOLE_SIZE;
    descriptorWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptorWrites[1].dstSet = descriptorSetABS[threadId];
    descriptorWrites[1].dstBinding = 1;
    descriptorWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    descriptorWrites[1].descriptorCount = 1;
    descriptorWrites[1].pBufferInfo = &absWorkingBufferInfo;

    vkUpdateDescriptorSets(
        logicalDevice,
        static_cast<uint32_t>(descriptorWrites.size()),
        descriptorWrites.data(),
        0,
        VK_NULL_HANDLE
    );
}

ShaderExecutionInfo VisibilityManager::randomSample(int numRays, int threadId) {
    // Reset atomic triangle counter
    {
        VkDeviceSize bufferSize = sizeof(unsigned int) * 2;

        // Create staging buffer using host-visible memory
        VkBuffer stagingBuffer;
        VkDeviceMemory stagingBufferMemory;
        VulkanUtil::createBuffer(
            physicalDevice, logicalDevice, bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            stagingBuffer, stagingBufferMemory,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT
        );

        // Copy triangles data to the staging buffer
        unsigned int numTriangles[2] = { 0, 0 };
        void *data;
        vkMapMemory(logicalDevice, stagingBufferMemory, 0, bufferSize, 0, &data);    // Map buffer memory into CPU accessible memory
        memcpy(data, &numTriangles, (size_t) bufferSize);  // Copy vertex data to mapped memory
        vkUnmapMemory(logicalDevice, stagingBufferMemory);

        // Copy triangles data from the staging buffer to GPU-visible absWorkingBuffer
        VulkanUtil::copyBuffer(
            logicalDevice, commandPool[threadId], computeQueue, stagingBuffer,
            triangleCounterBuffer[threadId], bufferSize, queueSubmitMutex
        );

        vkDestroyBuffer(logicalDevice, stagingBuffer, nullptr);
        vkFreeMemory(logicalDevice, stagingBufferMemory, nullptr);
    }

    VkCommandBufferBeginInfo beginInfo = {};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    vkBeginCommandBuffer(commandBuffer[threadId], &beginInfo);
    vkCmdBindPipeline(commandBuffer[threadId], VK_PIPELINE_BIND_POINT_RAY_TRACING_NV, pipeline);
    vkCmdBindDescriptorSets(
        commandBuffer[threadId], VK_PIPELINE_BIND_POINT_RAY_TRACING_NV, pipelineLayout, 0, 1,
        &descriptorSet[threadId], 0, nullptr
    );

    if (faceIndices.count(numThreads)) {
        vkCmdPushConstants(
            commandBuffer[threadId],
            pipelineLayout,
            VK_SHADER_STAGE_RAYGEN_BIT_NV,
            0,
            sizeof(int) * faceIndices[numThreads][threadId].size(),
            faceIndices[numThreads][threadId].data()
        );
    }
    vkCmdTraceRaysNV(
        commandBuffer[threadId],
        shaderBindingTable, bindingOffsetRayGenShader,
        shaderBindingTable, bindingOffsetMissShader, bindingStride,
        shaderBindingTable, bindingOffsetHitShader, bindingStride,
        VK_NULL_HANDLE, 0, 0,
        numRays, 1, 1
    );
    vkEndCommandBuffer(commandBuffer[threadId]);

    VulkanUtil::executeCommandBuffer(
        logicalDevice, computeQueue, commandBuffer[threadId], commandBufferFence[threadId],
        queueSubmitMutex
    );

    // Get number of intersected triangles from the GPU
    unsigned int numTriangles = 0;
    {
        VkDeviceSize bufferSize = sizeof(unsigned int);

        VkBuffer hostBuffer;
        VkDeviceMemory hostBufferMemory;
        VulkanUtil::createBuffer(
            physicalDevice,
            logicalDevice, bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT, hostBuffer, hostBufferMemory,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT
        );

        VulkanUtil::copyBuffer(
            logicalDevice, commandPool[threadId], computeQueue, triangleCounterBuffer[threadId],
            hostBuffer, bufferSize, queueSubmitMutex
        );

        void *data;
        vkMapMemory(logicalDevice, hostBufferMemory, 0, bufferSize, 0, &data);
        unsigned int *n = (unsigned int*) data;
        numTriangles += n[0];

        vkUnmapMemory(logicalDevice, hostBufferMemory);
        vkDestroyBuffer(logicalDevice, hostBuffer, nullptr);
        vkFreeMemory(logicalDevice, hostBufferMemory, nullptr);
    }

    // Copy intersected triangles from VRAM to CPU accessible buffer
    {
        VkDeviceSize bufferSize = sizeof(Sample) * numTriangles;

        // Copy the intersected triangles GPU buffer to the host buffer
        VulkanUtil::copyBuffer(
            logicalDevice, commandPool[threadId], computeQueue, randomSamplingOutputBuffer[threadId],
            randomSamplingOutputHostBuffer[threadId], bufferSize, queueSubmitMutex
        );
        //std::cout << bufferSize << " " << (bufferSize / 1000.0f) / 1000.0f << "mb " << RAYS_PER_ITERATION << " " << numTriangles << std::endl;
    }

    return { numTriangles, (unsigned int) numRays };
}

ShaderExecutionInfo VisibilityManager::adaptiveBorderSample(const std::vector<Sample> &triangles, int threadId) {
    // Copy triangles vector to GPU accessible buffer
    {
        VkDeviceSize bufferSize = sizeof(triangles[0]) * triangles.size();

        // Create staging buffer using host-visible memory
        VkBuffer stagingBuffer;                 // TODO: Create staging buffer beforehand and reuse it
        VkDeviceMemory stagingBufferMemory;
        VulkanUtil::createBuffer(
            physicalDevice, logicalDevice, bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            stagingBuffer, stagingBufferMemory,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT
        );

        // Copy triangles data to the staging buffer
        void *data;
        vkMapMemory(logicalDevice, stagingBufferMemory, 0, bufferSize, 0, &data);    // Map buffer memory into CPU accessible memory
        memcpy(data, triangles.data(), (size_t) bufferSize);  // Copy vertex data to mapped memory
        vkUnmapMemory(logicalDevice, stagingBufferMemory);

        // Copy triangles data from the staging buffer to GPU-visible absWorkingBuffer
        VulkanUtil::copyBuffer(
            logicalDevice, commandPool[threadId], computeQueue, stagingBuffer, absWorkingBuffer[threadId],
            bufferSize, queueSubmitMutex
        );

        vkDestroyBuffer(logicalDevice, stagingBuffer, nullptr);
        vkFreeMemory(logicalDevice, stagingBufferMemory, nullptr);
    }

    // Reset atomic triangle counter
    {
        VkDeviceSize bufferSize = sizeof(unsigned int) * 2;

        // Create staging buffer using host-visible memory
        VkBuffer stagingBuffer;
        VkDeviceMemory stagingBufferMemory;
        VulkanUtil::createBuffer(
            physicalDevice, logicalDevice, bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            stagingBuffer, stagingBufferMemory,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT
        );

        // Copy triangles data to the staging buffer
        unsigned int numTriangles[2] = { 0, 0 };
        void *data;
        vkMapMemory(logicalDevice, stagingBufferMemory, 0, bufferSize, 0, &data);    // Map buffer memory into CPU accessible memory
        memcpy(data, &numTriangles, (size_t) bufferSize);  // Copy vertex data to mapped memory
        vkUnmapMemory(logicalDevice, stagingBufferMemory);

        // Copy triangles data from the staging buffer to GPU-visible absWorkingBuffer
        VulkanUtil::copyBuffer(
            logicalDevice, commandPool[threadId], computeQueue, stagingBuffer,
            triangleCounterBuffer[threadId], bufferSize, queueSubmitMutex
        );

        vkDestroyBuffer(logicalDevice, stagingBuffer, nullptr);
        vkFreeMemory(logicalDevice, stagingBufferMemory, nullptr);
    }

    // Record and execute a command buffer for running the actual ABS on the GPU
    VkCommandBufferBeginInfo beginInfo = {};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(commandBufferABS[threadId], &beginInfo);
    vkCmdBindPipeline(commandBufferABS[threadId], VK_PIPELINE_BIND_POINT_RAY_TRACING_NV, pipelineABS);
    vkCmdBindDescriptorSets(
        commandBufferABS[threadId], VK_PIPELINE_BIND_POINT_RAY_TRACING_NV, pipelineABSLayout, 0, 1,
        &descriptorSet[threadId], 0, nullptr
    );
    vkCmdBindDescriptorSets(
        commandBufferABS[threadId], VK_PIPELINE_BIND_POINT_RAY_TRACING_NV, pipelineABSLayout, 1, 1,
        &descriptorSetABS[threadId], 0, nullptr
    );
    vkCmdTraceRaysNV(
        commandBufferABS[threadId],
        shaderBindingTableABS, bindingOffsetRayGenShader,
        shaderBindingTableABS, bindingOffsetMissShader, bindingStride,
        shaderBindingTableABS, bindingOffsetHitShader, bindingStride,
        VK_NULL_HANDLE, 0, 0,
        triangles.size() * NUM_ABS_SAMPLES, 1, 1
    );

    vkEndCommandBuffer(commandBufferABS[threadId]);
    VulkanUtil::executeCommandBuffer(
        logicalDevice, computeQueue, commandBufferABS[threadId], commandBufferFence[threadId],
        queueSubmitMutex
    );

    // Get number of intersected triangles from the GPU
    unsigned int numTriangles = triangles.size() * NUM_ABS_SAMPLES;
    unsigned int numRays;
    {
        VkDeviceSize bufferSize = sizeof(unsigned int) * 2;

        VkBuffer hostBuffer;
        VkDeviceMemory hostBufferMemory;
        VulkanUtil::createBuffer(
            physicalDevice,
            logicalDevice, bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT, hostBuffer, hostBufferMemory,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT
        );

        VulkanUtil::copyBuffer(
            logicalDevice, commandPool[threadId], computeQueue, triangleCounterBuffer[threadId],
            hostBuffer, bufferSize, queueSubmitMutex
        );

        void *data;
        vkMapMemory(logicalDevice, hostBufferMemory, 0, bufferSize, 0, &data);
        unsigned int *n = (unsigned int*) data;
        numTriangles += n[0];
        numRays = n[1];

        vkUnmapMemory(logicalDevice, hostBufferMemory);
        vkDestroyBuffer(logicalDevice, hostBuffer, nullptr);
        vkFreeMemory(logicalDevice, hostBufferMemory, nullptr);
    }

    {
        //VkDeviceSize bufferSize = sizeof(Sample) * triangles.size() * 9 * 2;
        VkDeviceSize bufferSize = sizeof(Sample) * numTriangles;

        // Copy the intersected triangles GPU buffer to the host buffer
        VulkanUtil::copyBuffer(
            logicalDevice, commandPool[threadId], computeQueue, absOutputBuffer[threadId],
            absOutputHostBuffer[threadId], bufferSize, queueSubmitMutex
        );
        //std::cout << bufferSize << " " << (bufferSize / 1000.0f) / 1000.0f << std::endl;
    }

    return { numTriangles, numRays };
    //return triangles.size() * 9 * 2;
}

ShaderExecutionInfo VisibilityManager::edgeSubdivide(int numSamples, int threadId) {
    // Reset atomic triangle counter
    {
        VkDeviceSize bufferSize = sizeof(unsigned int) * 2;

        // Create staging buffer using host-visible memory
        VkBuffer stagingBuffer;
        VkDeviceMemory stagingBufferMemory;
        VulkanUtil::createBuffer(
            physicalDevice, logicalDevice, bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            stagingBuffer, stagingBufferMemory,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT
        );

        // Copy triangles data to the staging buffer
        unsigned int numTriangles[2] = { 0, 0 };
        void *data;
        vkMapMemory(logicalDevice, stagingBufferMemory, 0, bufferSize, 0, &data);    // Map buffer memory into CPU accessible memory
        memcpy(data, &numTriangles, (size_t) bufferSize);  // Copy vertex data to mapped memory
        vkUnmapMemory(logicalDevice, stagingBufferMemory);

        // Copy triangles data from the staging buffer to GPU-visible absWorkingBuffer
        VulkanUtil::copyBuffer(
            logicalDevice, commandPool[threadId], computeQueue, stagingBuffer,
            triangleCounterBuffer[threadId], bufferSize, queueSubmitMutex
        );

        vkDestroyBuffer(logicalDevice, stagingBuffer, nullptr);
        vkFreeMemory(logicalDevice, stagingBufferMemory, nullptr);
    }


    auto start = std::chrono::steady_clock::now();
    // Record and execute a command buffer for running the actual edge subdivision on the GPU
    VkCommandBufferBeginInfo beginInfo = {};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(commandBufferEdgeSubdiv[threadId], &beginInfo);
    vkCmdBindPipeline(commandBufferEdgeSubdiv[threadId], VK_PIPELINE_BIND_POINT_RAY_TRACING_NV, pipelineEdgeSubdiv);
    vkCmdBindDescriptorSets(
        commandBufferEdgeSubdiv[threadId], VK_PIPELINE_BIND_POINT_RAY_TRACING_NV, pipelineEdgeSubdivLayout, 0, 1,
        &descriptorSet[threadId], 0, nullptr
    );
    vkCmdBindDescriptorSets(
        commandBufferEdgeSubdiv[threadId], VK_PIPELINE_BIND_POINT_RAY_TRACING_NV, pipelineEdgeSubdivLayout, 1, 1,
        &descriptorSetABS[threadId], 0, nullptr
    );
    vkCmdBindDescriptorSets(
        commandBufferEdgeSubdiv[threadId], VK_PIPELINE_BIND_POINT_RAY_TRACING_NV, pipelineEdgeSubdivLayout, 2, 1,
        &descriptorSetEdgeSubdiv[threadId], 0, nullptr
    );
    vkCmdTraceRaysNV(
        commandBufferEdgeSubdiv[threadId],
        shaderBindingTableEdgeSubdiv, bindingOffsetRayGenShader,
        shaderBindingTableEdgeSubdiv, bindingOffsetMissShader, bindingStride,
        shaderBindingTableEdgeSubdiv, bindingOffsetHitShader, bindingStride,
        VK_NULL_HANDLE, 0, 0,
        numSamples, 1, 1
    );
    vkEndCommandBuffer(commandBufferEdgeSubdiv[threadId]);

    VulkanUtil::executeCommandBuffer(
        logicalDevice, computeQueue, commandBufferEdgeSubdiv[threadId],
        commandBufferFence[threadId], queueSubmitMutex
    );
    auto end = std::chrono::steady_clock::now();
    //std::cout << threadId << " edge subdiv " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms " << std::endl;

    // Get number of intersected triangles from the GPU
    unsigned int numTriangles;
    unsigned int numRays;
    {
        VkDeviceSize bufferSize = sizeof(unsigned int) * 2;

        VkBuffer hostBuffer;
        VkDeviceMemory hostBufferMemory;
        VulkanUtil::createBuffer(
            physicalDevice,
            logicalDevice, bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT, hostBuffer, hostBufferMemory,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT
        );

        VulkanUtil::copyBuffer(
            logicalDevice, commandPool[threadId], computeQueue, triangleCounterBuffer[threadId],
            hostBuffer, bufferSize, queueSubmitMutex
        );

        void *data;
        vkMapMemory(logicalDevice, hostBufferMemory, 0, bufferSize, 0, &data);
        unsigned int *n = (unsigned int*) data;
        numTriangles = n[0];
        numRays = n[1];

        vkUnmapMemory(logicalDevice, hostBufferMemory);
        vkDestroyBuffer(logicalDevice, hostBuffer, nullptr);
        vkFreeMemory(logicalDevice, hostBufferMemory, nullptr);
    }

    {
        //VkDeviceSize bufferSize = sizeof(Sample) * numSamples * int(pow(2, MAX_SUBDIVISION_STEPS) + 1);
        VkDeviceSize bufferSize = sizeof(Sample) * numTriangles;

        // Copy the intersected triangles GPU buffer to the host buffer
        VulkanUtil::copyBuffer(
            logicalDevice, commandPool[threadId], computeQueue, edgeSubdivOutputBuffer[threadId],
            edgeSubdivOutputHostBuffer[threadId], bufferSize, queueSubmitMutex
        );
        //std::cout << bufferSize << " " << (bufferSize / 1000.0f) / 1000.0f << std::endl;
    }

    //return numSamples * int(pow(2, MAX_SUBDIVISION_STEPS) + 1);
    return { numTriangles, numRays };
}

void VisibilityManager::createShaderBindingTable(
    VkBuffer &shaderBindingTable, VkDeviceMemory &shaderBindingTableMemory, VkPipeline &pipeline
) {
    const uint32_t bindingTableSize = rayTracingProperties.shaderGroupHandleSize * 3;
    VulkanUtil::createBuffer(
        physicalDevice,
        logicalDevice, bindingTableSize,
        VK_BUFFER_USAGE_RAY_TRACING_BIT_NV,
        shaderBindingTable,
        shaderBindingTableMemory,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
    );

    // Fetch all the shader handles used in the pipeline, so that they can be written in the SBT
    auto shaderHandleStorage = new uint8_t[bindingTableSize];
    vkGetRayTracingShaderGroupHandlesNV(
        logicalDevice, pipeline, 0, 3, bindingTableSize, shaderHandleStorage
    );

    void *vData;
    vkMapMemory(logicalDevice, shaderBindingTableMemory, 0, bindingTableSize, 0, &vData);

    auto* data = static_cast<uint8_t*>(vData);
    data += copyShaderIdentifier(data, shaderHandleStorage, RT_SHADER_INDEX_RAYGEN);
    data += copyShaderIdentifier(data, shaderHandleStorage, RT_SHADER_INDEX_MISS);
    data += copyShaderIdentifier(data, shaderHandleStorage, RT_SHADER_INDEX_CLOSEST_HIT);

    vkUnmapMemory(logicalDevice, shaderBindingTableMemory);
}

VkDeviceSize VisibilityManager::copyShaderIdentifier(
    uint8_t *data, const uint8_t *shaderHandleStorage, uint32_t groupIndex
) {
    // Copy shader identifier to "data"
    const uint32_t shaderGroupHandleSize = rayTracingProperties.shaderGroupHandleSize;
    memcpy(data, shaderHandleStorage + groupIndex * shaderGroupHandleSize, shaderGroupHandleSize);
    data += shaderGroupHandleSize;

    return shaderGroupHandleSize;
}

void VisibilityManager::rayTrace(const std::vector<uint32_t> &indices, int threadId) {
    auto startTotal = std::chrono::steady_clock::now();

    auto start = std::chrono::steady_clock::now();
    auto end = std::chrono::steady_clock::now();
    auto randomSamplingTime = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    auto absTime = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    auto edgeSubdivTime = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    auto haltonTime = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    std::vector<Sample> absSampleQueue;
    size_t newTriangles;
    for (int i = 0; true; i++) {
        // if USE_TERMINATION_CRITERION is set to true, terminate if less than
        // NEW_TRIANGLE_TERMINATION_THRESHOLD new triangles have been found for
        // RAY_COUNT_TERMINATION_THRESHOLD rays

        if (tracedRays >= RAY_COUNT_TERMINATION_THRESHOLD) {
            //numRays = 0;
            if (USE_TERMINATION_CRITERION && pvs.getSet().size() - newTriangles < NEW_TRIANGLE_TERMINATION_THRESHOLD) {
                break;
            } else {
                tracedRays = 0;
            }
        }

        newTriangles = pvs.getSet().size();

        // Execute random sampling
        start = std::chrono::steady_clock::now();
        ShaderExecutionInfo randomSampleInfo = randomSample(RAYS_PER_ITERATION, threadId);
        statistics.entries.back().numShaderExecutions += RAYS_PER_ITERATION;
        statistics.entries.back().rnsRays += randomSampleInfo.numRays;
        //tracedRays += RAYS_PER_ITERATION;
        //unsigned int numTriangles = RAYS_PER_ITERATION;
        Sample *outputSamples = (Sample*)randomSamplingOutputPointer[threadId];
        for (int i = 0; i < randomSampleInfo.numTriangles; i++) {
            //std::cout << outputSamples[i] << std::endl;
            auto result = pvs.insert(outputSamples[i].triangleID, numThreads > 1);
            if (result.second) {
                statistics.entries.back().rnsTris++;
                statistics.entries.back().newTriangles++;

                // If the current triangle ID was inserted into the PVS, insert it into the new samples
                // vector as well
                absSampleQueue.push_back(outputSamples[i]);
            }
        }
        statistics.entries.back().pvsSize = pvs.getSet().size();
        statistics.update();
        end = std::chrono::steady_clock::now();
        randomSamplingTime += std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

        // Adaptive Border Sampling. ABS is executed for a maximum of MAX_ABS_RAYS rays at a time as
        // long as there are a number of MIN_ABS_RAYS unprocessed triangles left
        while (absSampleQueue.size() >= MIN_ABS_TRIANGLES_PER_ITERATION) {
            //std::cout << absSampleQueue.size() << std::endl;

                    start = std::chrono::steady_clock::now();
            // Get a maximum of MAX_ABS_RAYS triangles for which ABS will be run at a time
            int numAbsRays = std::min(MAX_ABS_TRIANGLES_PER_ITERATION, absSampleQueue.size());
            std::vector<Sample> absWorkingVector;
            absWorkingVector.reserve(numAbsRays);
            for (int i = absSampleQueue.size() - numAbsRays; i < absSampleQueue.size(); i++) {
                absWorkingVector.emplace_back(absSampleQueue[i]);
            }
            absSampleQueue.erase(absSampleQueue.end() - numAbsRays, absSampleQueue.end());

            // Execute ABS
            ShaderExecutionInfo absInfo = adaptiveBorderSample(absWorkingVector, threadId);
            statistics.entries.back().numShaderExecutions += absWorkingVector.size() * NUM_ABS_SAMPLES;
            statistics.entries.back().absRays += absWorkingVector.size() * NUM_ABS_SAMPLES;
            statistics.entries.back().rsRays += absInfo.numRays - absWorkingVector.size() * NUM_ABS_SAMPLES;

            // Insert the newly found triangles into the PVS and into the newSamples set for which
            // ABS is going to be executed again
            Sample *absOutputSamples = (Sample*)absOutputPointer[threadId];
            for (int i = 0; i < absInfo.numTriangles; i++) {
                //std::cout << i << " " << absOutputSamples[i] << std::endl;
                if (absOutputSamples[i].triangleID != -1) {
                    auto result = pvs.insert(absOutputSamples[i].triangleID, numThreads > 1);
                    if (result.second) {
                        if (i < absWorkingVector.size() * NUM_ABS_SAMPLES) {
                            statistics.entries.back().absTris++;
                        } else {
                            statistics.entries.back().rsTris++;
                        }
                        statistics.entries.back().newTriangles++;

                        // If the current triangle ID was inserted into the PVS, insert it into the new samples
                        // vector as well
                        absSampleQueue.push_back(absOutputSamples[i]);
                    }
                }
            }
            statistics.entries.back().pvsSize = pvs.getSet().size();
            statistics.update();
                    end = std::chrono::steady_clock::now();
                    absTime += std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();


                    start = std::chrono::steady_clock::now();
            ShaderExecutionInfo edgeSubdivideInfo = edgeSubdivide(absWorkingVector.size() * NUM_ABS_SAMPLES, threadId);
            statistics.entries.back().numShaderExecutions += absWorkingVector.size() * NUM_ABS_SAMPLES;
            statistics.entries.back().edgeSubdivRays += edgeSubdivideInfo.numRays;
            Sample *edgeSubdivOutputSamples = (Sample*)edgeSubdivOutputPointer[threadId];
            for (int i = 0; i < edgeSubdivideInfo.numTriangles; i++) {
                if (edgeSubdivOutputSamples[i].triangleID != -1) {
                    auto result = pvs.insert(edgeSubdivOutputSamples[i].triangleID, numThreads > 1);
                    if (result.second) {
                        statistics.entries.back().edgeSubdivTris++;
                        statistics.entries.back().newTriangles++;

                        // If the current triangle ID was inserted into the PVS, insert it into the new samples
                        // vector as well
                        absSampleQueue.push_back(edgeSubdivOutputSamples[i]);
                    }
                }
            }
            statistics.entries.back().pvsSize = pvs.getSet().size();
                    end = std::chrono::steady_clock::now();
                    edgeSubdivTime += std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
            statistics.update();
        }

        statistics.print();

        break;

                start = std::chrono::steady_clock::now();
        generateHaltonPoints2d(RAYS_PER_ITERATION, threadId, RAYS_PER_ITERATION * i);
        copyHaltonPointsToBuffer(threadId);
                end = std::chrono::steady_clock::now();
                haltonTime += std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    }

    auto endTotal = std::chrono::steady_clock::now();
    std::cout << "Total time: " << std::chrono::duration_cast<std::chrono::milliseconds>(endTotal - startTotal).count() << "ms" << std::endl;
    std::cout << "Random sampling time: " << randomSamplingTime << "ms" << std::endl;
    std::cout << "ABS + Reverse sampling time: " << absTime << "ms" << std::endl;
    std::cout << "Edge subdivision time: " << edgeSubdivTime << "ms" << std::endl;
    std::cout << "Halton time: " << haltonTime << "ms" << std::endl;
}

void VisibilityManager::releaseResources() {
    for (int i = 0; i < numThreads; i++) {
        vkDestroyBuffer(logicalDevice, haltonPointsBuffer[i], nullptr);
        vkFreeMemory(logicalDevice, haltonPointsBufferMemory[i], nullptr);

        vkDestroyBuffer(logicalDevice, viewCellBuffer[i], nullptr);
        vkFreeMemory(logicalDevice, viewCellBufferMemory[i], nullptr);

        vkDestroyBuffer(logicalDevice, randomSamplingOutputBuffer[i], nullptr);
        vkFreeMemory(logicalDevice, randomSamplingOutputBufferMemory[i], nullptr);

        vkUnmapMemory(logicalDevice, randomSamplingOutputHostBufferMemory[i]);
        vkDestroyBuffer(logicalDevice, randomSamplingOutputHostBuffer[i], nullptr);
        vkFreeMemory(logicalDevice, randomSamplingOutputHostBufferMemory[i], nullptr);

        vkDestroyBuffer(logicalDevice, absOutputBuffer[i], nullptr);
        vkFreeMemory(logicalDevice, absOutputBufferMemory[i], nullptr);
        vkDestroyBuffer(logicalDevice, absWorkingBuffer[i], nullptr);
        vkFreeMemory(logicalDevice, absWorkingBufferMemory[i], nullptr);

        vkDestroyBuffer(logicalDevice, edgeSubdivOutputBuffer[i], nullptr);
        vkFreeMemory(logicalDevice, edgeSubdivOutputBufferMemory[i], nullptr);
        vkDestroyBuffer(logicalDevice, edgeSubdivWorkingBuffer[i], nullptr);
        vkFreeMemory(logicalDevice, edgeSubdivWorkingBufferMemory[i], nullptr);

        vkDestroyFence(logicalDevice, commandBufferFence[i], nullptr);

        VkCommandBuffer commandBuffers[] = {
            commandBuffer[i],
            commandBufferABS[i],
            commandBufferEdgeSubdiv[i]
        };
        vkFreeCommandBuffers(logicalDevice, commandPool[i], 3, commandBuffers);

        vkDestroyCommandPool(logicalDevice, commandPool[i], nullptr);
    }

    vkDestroyBuffer(logicalDevice, pvsVisualizationBuffer, nullptr);
    vkFreeMemory(logicalDevice, pvsVisualizationBufferMemory, nullptr);

    vkDestroyBuffer(logicalDevice, shaderBindingTable, nullptr);
    vkFreeMemory(logicalDevice, shaderBindingTableMemory, nullptr);
    vkDestroyBuffer(logicalDevice, shaderBindingTableABS, nullptr);
    vkFreeMemory(logicalDevice, shaderBindingTableMemoryABS, nullptr);
    vkDestroyBuffer(logicalDevice, shaderBindingTableEdgeSubdiv, nullptr);
    vkFreeMemory(logicalDevice, shaderBindingTableMemoryEdgeSubdiv, nullptr);

    vkDestroyDescriptorSetLayout(logicalDevice, descriptorSetLayout, nullptr);
    vkDestroyDescriptorSetLayout(logicalDevice, descriptorSetLayoutABS, nullptr);
    vkDestroyDescriptorSetLayout(logicalDevice, descriptorSetLayoutEdgeSubdiv, nullptr);

    vkDestroyPipeline(logicalDevice, pipeline, nullptr);
    vkDestroyPipelineLayout(logicalDevice, pipelineLayout, nullptr);
    vkDestroyPipeline(logicalDevice, pipelineABS, nullptr);
    vkDestroyPipelineLayout(logicalDevice, pipelineABSLayout, nullptr);
    vkDestroyPipeline(logicalDevice, pipelineEdgeSubdiv, nullptr);
    vkDestroyPipelineLayout(logicalDevice, pipelineEdgeSubdivLayout, nullptr);

    vkDestroyDescriptorPool(logicalDevice, descriptorPool, nullptr);

    vkDestroyAccelerationStructureNV(logicalDevice, topLevelAS.as, nullptr);
    vkFreeMemory(logicalDevice, topLevelAS.deviceMemory, nullptr);
    vkDestroyAccelerationStructureNV(logicalDevice, bottomLevelAS.as, nullptr);
    vkFreeMemory(logicalDevice, bottomLevelAS.deviceMemory, nullptr);
}

VkBuffer VisibilityManager::getPVSIndexBuffer(
    const std::vector<uint32_t> &indices, VkCommandPool commandPool, VkQueue queue, bool inverted
) {
    // Collect the vertex indices of the triangles in the PVS
    std::vector<uint32_t> pvsIndices;
    pvsIndices.reserve(pvs.getSet().size() * 3);
    if (inverted) {
        for (int i = 0; i < int(indices.size() / 3.0f); i++) {
            if (pvs.getSet().find(i) == pvs.getSet().end()) {
                pvsIndices.emplace_back(indices[3 * i]);
                pvsIndices.emplace_back(indices[3 * i + 1]);
                pvsIndices.emplace_back(indices[3 * i + 2]);
            }
        }
    } else {
        for (auto triangleID : pvs.getSet()) {
            if (triangleID != -1) {
                pvsIndices.emplace_back(indices[3 * triangleID]);
                pvsIndices.emplace_back(indices[3 * triangleID + 1]);
                pvsIndices.emplace_back(indices[3 * triangleID + 2]);
            }
        }
    }

    // Copy PVS data to GPU accessible pvs visualization buffer (has the same size as the index vector)
    VkDeviceSize bufferSize = sizeof(pvsIndices[0]) * pvsIndices.size();

    // Create staging buffer using host-visible memory
    VkBuffer stagingBuffer;
    VkDeviceMemory stagingBufferMemory;
    VulkanUtil::createBuffer(
        physicalDevice,
        logicalDevice, bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, stagingBuffer, stagingBufferMemory,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
    );

    // Copy PVS data to the staging buffer
    void *data;
    vkMapMemory(logicalDevice, stagingBufferMemory, 0, bufferSize, 0, &data);    // Map buffer memory into CPU accessible memory
    memcpy(data, pvsIndices.data(), (size_t) bufferSize);  // Copy vertex data to mapped memory
    vkUnmapMemory(logicalDevice, stagingBufferMemory);

    // Copy PVS data from the staging buffer to the GPU-visible PVS visualization buffer (used as an index buffer when drawing)
    VulkanUtil::copyBuffer(
        logicalDevice, commandPool, queue, stagingBuffer, pvsVisualizationBuffer,
        bufferSize//, queueSubmitMutex
    );

    vkDestroyBuffer(logicalDevice, stagingBuffer, nullptr);
    vkFreeMemory(logicalDevice, stagingBufferMemory, nullptr);

    return pvsVisualizationBuffer;
}

void VisibilityManager::createCommandBuffers() {
    commandBuffer.resize(numThreads);
    commandBufferABS.resize(numThreads);
    commandBufferEdgeSubdiv.resize(numThreads);
    commandBufferFence.resize(numThreads);

    for (int i = 0; i < numThreads; i++) {
        VkCommandBufferAllocateInfo allocInfo = {};
        allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocInfo.commandPool = commandPool[i];
        allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocInfo.commandBufferCount = 1;

        if (vkAllocateCommandBuffers(logicalDevice, &allocInfo, &commandBuffer[i]) != VK_SUCCESS) {
            throw std::runtime_error("failed to allocate rt command buffer!");
        }

        if (vkAllocateCommandBuffers(logicalDevice, &allocInfo, &commandBufferABS[i]) != VK_SUCCESS) {
            throw std::runtime_error("failed to allocate rt command buffer abs!");
        }

        if (vkAllocateCommandBuffers(logicalDevice, &allocInfo, &commandBufferEdgeSubdiv[i]) != VK_SUCCESS) {
            throw std::runtime_error("failed to allocate rt command buffer edge subdiv!");
        }

        // Create fences used to wait for command buffer execution completion after submitting them
        VkFenceCreateInfo fenceInfo;
        fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        fenceInfo.pNext = NULL;
        fenceInfo.flags = 0;
        vkCreateFence(logicalDevice, &fenceInfo, NULL, &commandBufferFence[i]);
    }
}
