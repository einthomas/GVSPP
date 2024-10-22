#include <cstring>
#include <glm/gtx/string_cast.hpp>
#include <iostream>
#include <chrono>
#include <iterator>
#include <algorithm>
#include <unordered_set>
#include <glm/vec3.hpp>

#include "vulkanutil.h"
#include "visibilitymanager.h"
#include "viewcell.h"
#include "sample.h"
#include "Vertex.h"

struct UniformBufferObjectMultiView {
    alignas(64) glm::mat4 model;
    alignas(64) glm::mat4 view;
    alignas(64) glm::mat4 projection;
};

VisibilityManager::VisibilityManager(
    long NEW_TRIANGLE_TERMINATION_THRESHOLD_COUNT,
    long NEW_TRIANGLE_TERMINATION_THRESHOLD,
    long RANDOM_RAYS_PER_ITERATION,
    long NUM_ABS_SAMPLES,
    long REVERSE_SAMPLING_NUM_SAMPLES_ALONG_EDGE,
    long MAX_BULK_INSERT_BUFFER_SIZE,
    int GPU_SET_TYPE,
    long INITIAL_HASH_SET_SIZE,
    VkPhysicalDevice physicalDevice,
    VkDevice logicalDevice,
    VkBuffer indexBuffer,
    const std::vector<uint32_t> &indices,
    VkBuffer vertexBuffer,
    const std::vector<Vertex> &vertices,
    const std::vector<VkBuffer> &uniformBuffers,
    std::array<uint8_t, VK_UUID_SIZE> deviceUUID,
    std::vector<ViewCell> viewCells,
    VkCommandPool graphicsCommandPool,
    VkQueue graphicsQueue,
    uint32_t frameBufferWidth,
    uint32_t frameBufferHeight,
    VkFormat depthFormat,
    bool visualizeFirstRays
):
    NEW_TRIANGLE_TERMINATION_THRESHOLD_COUNT(NEW_TRIANGLE_TERMINATION_THRESHOLD_COUNT),
    NEW_TRIANGLE_TERMINATION_THRESHOLD(NEW_TRIANGLE_TERMINATION_THRESHOLD),
    RANDOM_RAYS_PER_ITERATION(RANDOM_RAYS_PER_ITERATION),
    NUM_ABS_SAMPLES(NUM_ABS_SAMPLES + 9),
    NUM_REVERSE_SAMPLING_SAMPLES(REVERSE_SAMPLING_NUM_SAMPLES_ALONG_EDGE),
    MAX_BULK_INSERT_BUFFER_SIZE(MAX_BULK_INSERT_BUFFER_SIZE),
    GPU_SET_TYPE(GPU_SET_TYPE),
    MAX_TRIANGLE_COUNT(indices.size() / 3.0f),
    viewCells(viewCells),
    visualizeFirstRays(visualizeFirstRays)
{
    this->logicalDevice = logicalDevice;
    this->physicalDevice = physicalDevice;
    this->deviceUUID = deviceUUID;

    rayVertices.resize(viewCells.size());

    if (GPU_SET_TYPE == 0) {
        // PVS size when a GPU SET is used. Has to be equal to the number of triangles in the scene
        pvsBufferCapacity = MAX_TRIANGLE_COUNT;
    } else {
        if (INITIAL_HASH_SET_SIZE == 0) {
            // Initial PVS size when a GPU HASH SET is used. Has to be a power of 2
            pvsBufferCapacity = 1 << int(std::ceil(std::log2(MAX_TRIANGLE_COUNT / 2.0f)));
        } else {
            pvsBufferCapacity = INITIAL_HASH_SET_SIZE;
        }
    }

    tracedRays = 0;

    gen.seed(rd());

    uint32_t computeQueueFamilyIndex = VulkanUtil::findQueueFamilies(
        physicalDevice, VK_QUEUE_COMPUTE_BIT, 0
    );
    vkGetDeviceQueue(logicalDevice, computeQueueFamilyIndex, 0, &computeQueue);

    uint32_t transferQueueFamilyIndex = VulkanUtil::findQueueFamilies(
        physicalDevice, VK_QUEUE_TRANSFER_BIT, 0
    );

    vkGetDeviceQueue(logicalDevice, transferQueueFamilyIndex, 0, &transferQueue);

    VkCommandPoolCreateInfo cmdPoolInfo = {};
    cmdPoolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    cmdPoolInfo.queueFamilyIndex = computeQueueFamilyIndex;
    cmdPoolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;    // Has to be set otherwise the command buffers can't be re-recorded
    if (vkCreateCommandPool(logicalDevice, &cmdPoolInfo, nullptr, &commandPool)) {
        throw std::runtime_error("failed to create visibility manager command pool!");
    }

    {
        VkCommandPoolCreateInfo cmdPoolInfo = {};
        cmdPoolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        cmdPoolInfo.queueFamilyIndex = transferQueueFamilyIndex;
        cmdPoolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;    // Has to be set otherwise the command buffers can't be re-recorded
        if (vkCreateCommandPool(logicalDevice, &cmdPoolInfo, nullptr, &transferCommandPool)) {
            throw std::runtime_error("failed to create visibility manager transfer command pool!");
        }
    }

    createBuffers(indices);
    initRayTracing(indexBuffer, vertexBuffer, indices, vertices, uniformBuffers);
    generateHaltonSequence(RANDOM_RAYS_PER_ITERATION * 4.0f, rand() / float(RAND_MAX));
}

VisibilityManager::~VisibilityManager() {
    releaseResources();
}

void VisibilityManager::copyHaltonPointsToBuffer() {
    VkDeviceSize bufferSize;
    bufferSize = sizeof(haltonPoints[0]) * haltonPoints.size();

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
    memcpy(data, haltonPoints.data(), (size_t) bufferSize);  // Copy 2d halton points to mapped memory
    vkUnmapMemory(logicalDevice, stagingBufferMemory);

    // Copy halton points from the staging buffer to the halton points buffer
    VulkanUtil::copyBuffer(
        logicalDevice, transferCommandPool, transferQueue, stagingBuffer, haltonPointsBuffer,
        bufferSize
    );

    vkDestroyBuffer(logicalDevice, stagingBuffer, nullptr);
    vkFreeMemory(logicalDevice, stagingBufferMemory, nullptr);
}

void VisibilityManager::updateViewCellBuffer(int viewCellIndex) {
    VkDeviceSize viewCellBufferSize = sizeof(viewCells[viewCellIndex]) * viewCells.size();

    // Create staging buffer using host-visible memory
    VkBuffer stagingBuffer;
    VkDeviceMemory stagingBufferMemory;
    VulkanUtil::createBuffer(
        physicalDevice,
        logicalDevice, viewCellBufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, stagingBuffer, stagingBufferMemory,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT
    );

    void *data;
    vkMapMemory(logicalDevice, stagingBufferMemory, 0, viewCellBufferSize, 0, &data);
    memcpy(data, &viewCells[viewCellIndex], (size_t) viewCellBufferSize);
    vkUnmapMemory(logicalDevice, stagingBufferMemory);

    VulkanUtil::copyBuffer(
        logicalDevice, transferCommandPool, transferQueue, stagingBuffer, viewCellBuffer, viewCellBufferSize
    );

    vkDestroyBuffer(logicalDevice, stagingBuffer, nullptr);
    vkFreeMemory(logicalDevice, stagingBufferMemory, nullptr);
}

void VisibilityManager::resetPVSGPUBuffer() {
    VkDeviceSize pvsSize = sizeof(int) * pvsBufferCapacity;

    VkDeviceSize bufferSize = pvsSize;

    // Create staging buffer using host-visible memory
    VkBuffer stagingBuffer;
    VkDeviceMemory stagingBufferMemory;
    VulkanUtil::createBuffer(
        physicalDevice, logicalDevice, bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        stagingBuffer, stagingBufferMemory,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT
    );

    std::vector<int> vec(pvsBufferCapacity);
    std::fill(vec.begin(), vec.end(), -1);
    void *data;
    vkMapMemory(logicalDevice, stagingBufferMemory, 0, bufferSize, 0, &data);    // Map buffer memory into CPU accessible memory
    memcpy(data, vec.data(), (size_t) bufferSize);  // Copy vertex data to mapped memory
    vkUnmapMemory(logicalDevice, stagingBufferMemory);

    VulkanUtil::copyBuffer(
        logicalDevice, transferCommandPool, transferQueue, stagingBuffer,
        pvsBuffer, bufferSize
    );

    vkDestroyBuffer(logicalDevice, stagingBuffer, nullptr);
    vkFreeMemory(logicalDevice, stagingBufferMemory, nullptr);

    this->pvsSize = 0;
}

void VisibilityManager::resetAtomicBuffers() {
    VkDeviceSize bufferSize = sizeof(unsigned int) * 5;

    // Create staging buffer using host-visible memory
    VkBuffer stagingBuffer;
    VkDeviceMemory stagingBufferMemory;
    VulkanUtil::createBuffer(
        physicalDevice, logicalDevice, bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        stagingBuffer, stagingBufferMemory,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT
    );

    // Copy triangles data to the staging buffer
    unsigned int numTriangles[5] = { 0, 0, 0, 0, 0 };
    void *data;
    vkMapMemory(logicalDevice, stagingBufferMemory, 0, bufferSize, 0, &data);    // Map buffer memory into CPU accessible memory
    memcpy(data, &numTriangles, (size_t) bufferSize);  // Copy vertex data to mapped memory
    vkUnmapMemory(logicalDevice, stagingBufferMemory);

    // Copy triangles data from the staging buffer to GPU-visible absWorkingBuffer
    VulkanUtil::copyBuffer(
        logicalDevice, transferCommandPool, transferQueue, stagingBuffer,
        triangleCounterBuffer, bufferSize
    );

    vkDestroyBuffer(logicalDevice, stagingBuffer, nullptr);
    vkFreeMemory(logicalDevice, stagingBufferMemory, nullptr);
}

void VisibilityManager::resizeHashSetPVSBuffer(int newSize) {
    std::cout << "Resize PVS GPU hash set: " << pvsBufferCapacity << " -> " << newSize << std::endl;

    // Copy PVS buffer to host
    VkDeviceSize bufferSize = sizeof(int) * pvsBufferCapacity;

    VkBuffer hostBuffer;
    VkDeviceMemory hostBufferMemory;
    VulkanUtil::createBuffer(
        physicalDevice,
        logicalDevice, bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT, hostBuffer, hostBufferMemory,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT
    );

    VulkanUtil::copyBuffer(
        logicalDevice, transferCommandPool, transferQueue, pvsBuffer,
        hostBuffer, bufferSize
    );

    void *data;
    vkMapMemory(logicalDevice, hostBufferMemory, 0, bufferSize, 0, &data);
    int *pvsArray = (int*)data;


    // Destroy small PVS buffer and free memory
    vkDestroyBuffer(logicalDevice, pvsBuffer, nullptr);
    vkFreeMemory(logicalDevice, pvsBufferMemory, nullptr);


    // Create larger PVS buffer
    VulkanUtil::createBuffer(
        physicalDevice, logicalDevice, sizeof(int) * newSize,
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_RAY_TRACING_BIT_NV | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        pvsBuffer, pvsBufferMemory, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
    );
    int oldPVSBufferCapacity = pvsBufferCapacity;
    pvsBufferCapacity = newSize;

    int pvsSizeOld = pvsSize;
    resetPVSGPUBuffer();
    pvsSize = pvsSizeOld;


    // Update PVS capacity uniform
    void *pvsCapacityUniformData;
    vkMapMemory(logicalDevice, pvsCapacityUniformMemory, 0, sizeof(pvsBufferCapacity), 0, &pvsCapacityUniformData);
    memcpy(pvsCapacityUniformData, &pvsBufferCapacity, sizeof(pvsBufferCapacity));
    vkUnmapMemory(logicalDevice, pvsCapacityUniformMemory);


    // Re-insert PVS data
    if (pvsSize > 0) {
        // Create bulk insert buffer
        VkDeviceSize bufferSize = sizeof(int) * std::min(MAX_BULK_INSERT_BUFFER_SIZE, (long)oldPVSBufferCapacity);
        VulkanUtil::createBuffer(
            physicalDevice,
            logicalDevice, bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            pvsBulkInsertBuffer, pvsBulkInsertBufferMemory, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
        );
        createComputeDescriptorSets();


        // Copy PVS data to bulk insert buffer
        VkBuffer stagingBuffer;
        VkDeviceMemory stagingBufferMemory;
        VulkanUtil::createBuffer(
            physicalDevice, logicalDevice, bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            stagingBuffer, stagingBufferMemory,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT
        );
        void *data;
        vkMapMemory(logicalDevice, stagingBufferMemory, 0, bufferSize, 0, &data);    // Map buffer memory into CPU accessible memory

        for (int i = 0; i < std::ceil(oldPVSBufferCapacity / float(MAX_BULK_INSERT_BUFFER_SIZE)); i++) {
            int bufferSize;
            if (oldPVSBufferCapacity < (i + 1) * MAX_BULK_INSERT_BUFFER_SIZE) {
                bufferSize = (oldPVSBufferCapacity - i * MAX_BULK_INSERT_BUFFER_SIZE);
            } else {
                bufferSize = MAX_BULK_INSERT_BUFFER_SIZE;
            }
            VkDeviceSize bufferSizeDeviceSize = sizeof(int) * bufferSize;
            memcpy(data, pvsArray, (size_t) bufferSizeDeviceSize);  // Copy pvs data to mapped memory

            pvsArray += bufferSize;

            VulkanUtil::copyBuffer(
                logicalDevice, transferCommandPool, transferQueue, stagingBuffer, pvsBulkInsertBuffer,
                bufferSizeDeviceSize
            );


            // Insert data into the larger PVS buffer using a compute shader
            VkCommandBufferBeginInfo beginInfo = {};
            beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
            beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

            vkBeginCommandBuffer(commandBufferCompute, &beginInfo);
            vkCmdBindPipeline(commandBufferCompute, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineCompute);
            vkCmdBindDescriptorSets(
                commandBufferCompute, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineComputeLayout, 0, 1,
                &descriptorSetCompute, 0, nullptr
            );
            vkCmdDispatch(commandBufferCompute, bufferSize, 1, 1);
            vkEndCommandBuffer(commandBufferCompute);

            VulkanUtil::executeCommandBuffer(
                logicalDevice, computeQueue, commandBufferCompute, commandBufferFence
            );
        }

        vkUnmapMemory(logicalDevice, stagingBufferMemory);

        vkDestroyBuffer(logicalDevice, stagingBuffer, nullptr);
        vkFreeMemory(logicalDevice, stagingBufferMemory, nullptr);
    }

    // Update descriptor set
    std::array<VkWriteDescriptorSet, 1> descriptorWrites = {};
    VkDescriptorBufferInfo testBufferInfo = {};
    testBufferInfo.buffer = pvsBuffer;
    testBufferInfo.offset = 0;
    testBufferInfo.range = VK_WHOLE_SIZE;
    descriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptorWrites[0].dstSet = descriptorSet;
    descriptorWrites[0].dstBinding = 9;
    descriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    descriptorWrites[0].descriptorCount = 1;
    descriptorWrites[0].pBufferInfo = &testBufferInfo;
    vkUpdateDescriptorSets(
        logicalDevice,
        static_cast<uint32_t>(descriptorWrites.size()),
        descriptorWrites.data(),
        0,
        VK_NULL_HANDLE
    );

    vkDestroyBuffer(logicalDevice, hostBuffer, nullptr);
    vkFreeMemory(logicalDevice, hostBufferMemory, nullptr);

    vkDestroyBuffer(logicalDevice, pvsBulkInsertBuffer, nullptr);
    vkFreeMemory(logicalDevice, pvsBulkInsertBufferMemory, nullptr);
}

void VisibilityManager::generateHaltonSequence(int n, float rand) {
    // Generate Halton sequence of length n using a compute shader
    VkCommandBufferBeginInfo beginInfo = {};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    vkBeginCommandBuffer(commandBufferHaltonCompute, &beginInfo);
    vkCmdBindPipeline(commandBufferHaltonCompute, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineHaltonCompute);
    vkCmdBindDescriptorSets(
        commandBufferHaltonCompute, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineHaltonComputeLayout,
        0, 1, &descriptorSetHaltonCompute, 0, nullptr
    );
    vkCmdPushConstants(
        commandBufferHaltonCompute, pipelineHaltonComputeLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
        sizeof(float), &rand
    );
    vkCmdPushConstants(
        commandBufferHaltonCompute, pipelineHaltonComputeLayout, VK_SHADER_STAGE_COMPUTE_BIT, sizeof(float),
        sizeof(int), &n
    );
    vkCmdDispatch(commandBufferHaltonCompute, ((n / 4.0f) + 256 - 1) / 256.0f, 1, 1);
    vkEndCommandBuffer(commandBufferHaltonCompute);

    VulkanUtil::executeCommandBuffer(
        logicalDevice, computeQueue, commandBufferHaltonCompute, commandBufferFence
    );
}

void VisibilityManager::printAverageStatistics() {
    Statistics::printAverageStatistics(statistics);
}

void VisibilityManager::createBuffers(const std::vector<uint32_t> &indices) {
        // Random sampling buffers
    VkDeviceSize pvsSize = sizeof(int) * pvsBufferCapacity;

    VkDeviceSize haltonSize = sizeof(float) * RANDOM_RAYS_PER_ITERATION * 4;

    VkDeviceSize absOutputBufferSize;
    VkDeviceSize absWorkingBufferSize = sizeof(Sample) * MAX_ABS_TRIANGLES_PER_ITERATION;
    //absOutputBufferSize = sizeof(Sample) * std::min(MAX_ABS_TRIANGLES_PER_ITERATION * NUM_ABS_SAMPLES * NUM_REVERSE_SAMPLING_SAMPLES, MAX_TRIANGLE_COUNT);
    absOutputBufferSize = sizeof(Sample) * MAX_TRIANGLE_COUNT;
    VkDeviceSize randomSamplingOutputBufferSize = std::max(sizeof(Sample) * std::min(RANDOM_RAYS_PER_ITERATION, MAX_TRIANGLE_COUNT), absOutputBufferSize);

    VkDeviceSize viewCellBufferSize = sizeof(viewCells[0]) * viewCells.size();
    VulkanUtil::createBuffer(
        physicalDevice,
        logicalDevice, randomSamplingOutputBufferSize,
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_RAY_TRACING_BIT_NV | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        randomSamplingOutputBuffer, randomSamplingOutputBufferMemory, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT
    );
    vkMapMemory(logicalDevice, randomSamplingOutputBufferMemory, 0, randomSamplingOutputBufferSize, 0, &randomSamplingOutputPointer);

    VulkanUtil::createBuffer(
        physicalDevice,
        logicalDevice, sizeof(unsigned int) * 5,
        VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_RAY_TRACING_BIT_NV | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        triangleCounterBuffer, triangleCounterBufferMemory, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
    );

    // ABS buffers
    VulkanUtil::createBuffer(
        physicalDevice,
        logicalDevice, absWorkingBufferSize,
        VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_RAY_TRACING_BIT_NV | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        absWorkingBuffer, absWorkingBufferMemory, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
    );

    // Create halton points buffer using GPU memory
    VulkanUtil::createBuffer(
        physicalDevice, logicalDevice, haltonSize,
        VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_RAY_TRACING_BIT_NV | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        haltonPointsBuffer, haltonPointsBufferMemory, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
    );

    VulkanUtil::createBuffer(
        physicalDevice,
        logicalDevice, pvsSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_RAY_TRACING_BIT_NV | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, pvsBuffer, pvsBufferMemory,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
    );

    resetPVSGPUBuffer();
    resetAtomicBuffers();

    VulkanUtil::createBuffer(
        physicalDevice,
        logicalDevice, viewCellBufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_RAY_TRACING_BIT_NV | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
        viewCellBuffer, viewCellBufferMemory, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
    );


    {
        VulkanUtil::createBuffer(
            physicalDevice,
            logicalDevice, sizeof(int),
            VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
            pvsCapacityUniformBuffer, pvsCapacityUniformMemory,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
        );
        void *data;
        vkMapMemory(logicalDevice, pvsCapacityUniformMemory, 0, sizeof(pvsBufferCapacity), 0, &data);
        memcpy(data, &pvsBufferCapacity, sizeof(pvsBufferCapacity));
        vkUnmapMemory(logicalDevice, pvsCapacityUniformMemory);
    }

    // Reset atomic counters
    {
        VkDeviceSize bufferSize = sizeof(unsigned int) * 5;

        // Create staging buffer using host-visible memory
        VkBuffer stagingBuffer;
        VkDeviceMemory stagingBufferMemory;
        VulkanUtil::createBuffer(
            physicalDevice, logicalDevice, bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            stagingBuffer, stagingBufferMemory,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT
        );

        // Copy triangles data to the staging buffer
        unsigned int numTriangles[5] = { 0, 0, 0, 0, 0 };
        void *data;
        vkMapMemory(logicalDevice, stagingBufferMemory, 0, bufferSize, 0, &data);    // Map buffer memory into CPU accessible memory
        memcpy(data, &numTriangles, (size_t) bufferSize);  // Copy vertex data to mapped memory
        vkUnmapMemory(logicalDevice, stagingBufferMemory);

        // Copy triangles data from the staging buffer to GPU-visible absWorkingBuffer
        VulkanUtil::copyBuffer(
            logicalDevice, transferCommandPool, transferQueue, stagingBuffer,
            triangleCounterBuffer, bufferSize
        );

        vkDestroyBuffer(logicalDevice, stagingBuffer, nullptr);
        vkFreeMemory(logicalDevice, stagingBufferMemory, nullptr);
    }
}

void VisibilityManager::createDescriptorSets(
    VkBuffer indexBuffer, VkBuffer vertexBuffer, const std::vector<VkBuffer> &uniformBuffers
) {
    std::array<VkWriteDescriptorSet, 11> descriptorWrites = {};

    VkWriteDescriptorSetAccelerationStructureNV asWriteInfo = {};
    asWriteInfo.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_NV;
    asWriteInfo.accelerationStructureCount = 1;
    asWriteInfo.pAccelerationStructures = &topLevelAS.as;
    descriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptorWrites[0].pNext = &asWriteInfo;
    descriptorWrites[0].dstSet = descriptorSet;
    descriptorWrites[0].dstBinding = 0;
    descriptorWrites[0].descriptorCount = 1;
    descriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_NV;

    VkDescriptorBufferInfo uniformBufferInfo = {};
    uniformBufferInfo.buffer = uniformBuffers[0];
    uniformBufferInfo.offset = 0;
    uniformBufferInfo.range = sizeof(UniformBufferObjectMultiView);
    descriptorWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptorWrites[1].dstSet = descriptorSet;
    descriptorWrites[1].dstBinding = 1;
    descriptorWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    descriptorWrites[1].descriptorCount = 1;
    descriptorWrites[1].pBufferInfo = &uniformBufferInfo;

    VkDescriptorBufferInfo vertexBufferInfo = {};
    vertexBufferInfo.buffer = vertexBuffer;
    vertexBufferInfo.offset = 0;
    vertexBufferInfo.range = VK_WHOLE_SIZE;
    descriptorWrites[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptorWrites[2].dstSet = descriptorSet;
    descriptorWrites[2].dstBinding = 2;
    descriptorWrites[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    descriptorWrites[2].descriptorCount = 1;
    descriptorWrites[2].pBufferInfo = &vertexBufferInfo;

    VkDescriptorBufferInfo indexBufferInfo = {};
    indexBufferInfo.buffer = indexBuffer;
    indexBufferInfo.offset = 0;
    indexBufferInfo.range = VK_WHOLE_SIZE;
    descriptorWrites[3].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptorWrites[3].dstSet = descriptorSet;
    descriptorWrites[3].dstBinding = 3;
    descriptorWrites[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    descriptorWrites[3].descriptorCount = 1;
    descriptorWrites[3].pBufferInfo = &indexBufferInfo;

    VkDescriptorBufferInfo haltonPointsBufferInfo = {};
    haltonPointsBufferInfo.buffer = haltonPointsBuffer;
    haltonPointsBufferInfo.offset = 0;
    haltonPointsBufferInfo.range = VK_WHOLE_SIZE;
    descriptorWrites[4].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptorWrites[4].dstSet = descriptorSet;
    descriptorWrites[4].dstBinding = 4;
    descriptorWrites[4].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    descriptorWrites[4].descriptorCount = 1;
    descriptorWrites[4].pBufferInfo = &haltonPointsBufferInfo;

    VkDescriptorBufferInfo viewCellBufferInfo = {};
    viewCellBufferInfo.buffer = viewCellBuffer;
    viewCellBufferInfo.offset = 0;
    viewCellBufferInfo.range = VK_WHOLE_SIZE;
    descriptorWrites[5].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptorWrites[5].dstSet = descriptorSet;
    descriptorWrites[5].dstBinding = 5;
    descriptorWrites[5].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    descriptorWrites[5].descriptorCount = 1;
    descriptorWrites[5].pBufferInfo = &viewCellBufferInfo;

    VkDescriptorBufferInfo randomSamplingBufferInfo = {};
    randomSamplingBufferInfo.buffer = randomSamplingOutputBuffer;
    randomSamplingBufferInfo.offset = 0;
    randomSamplingBufferInfo.range = VK_WHOLE_SIZE;
    descriptorWrites[6].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptorWrites[6].dstSet = descriptorSet;
    descriptorWrites[6].dstBinding = 6;
    descriptorWrites[6].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    descriptorWrites[6].descriptorCount = 1;
    descriptorWrites[6].pBufferInfo = &randomSamplingBufferInfo;

    VkDescriptorBufferInfo trianglesBufferInfo = {};
    trianglesBufferInfo.buffer = absWorkingBuffer;
    trianglesBufferInfo.offset = 0;
    trianglesBufferInfo.range = VK_WHOLE_SIZE;
    descriptorWrites[7].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptorWrites[7].dstSet = descriptorSet;
    descriptorWrites[7].dstBinding = 7;
    descriptorWrites[7].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    descriptorWrites[7].descriptorCount = 1;
    descriptorWrites[7].pBufferInfo = &trianglesBufferInfo;

    VkDescriptorBufferInfo triangleCounterBufferInfo = {};
    triangleCounterBufferInfo.buffer = triangleCounterBuffer;
    triangleCounterBufferInfo.offset = 0;
    triangleCounterBufferInfo.range = VK_WHOLE_SIZE;
    descriptorWrites[8].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptorWrites[8].dstSet = descriptorSet;
    descriptorWrites[8].dstBinding = 8;
    descriptorWrites[8].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    descriptorWrites[8].descriptorCount = 1;
    descriptorWrites[8].pBufferInfo = &triangleCounterBufferInfo;

    VkDescriptorBufferInfo pvsBufferInfo = {};
    pvsBufferInfo.buffer = pvsBuffer;
    pvsBufferInfo.offset = 0;
    pvsBufferInfo.range = VK_WHOLE_SIZE;
    descriptorWrites[9].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptorWrites[9].dstSet = descriptorSet;
    descriptorWrites[9].dstBinding = 9;
    descriptorWrites[9].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    descriptorWrites[9].descriptorCount = 1;
    descriptorWrites[9].pBufferInfo = &pvsBufferInfo;

    VkDescriptorBufferInfo pvsBufferCapacityBufferInfo = {};
    pvsBufferCapacityBufferInfo.buffer = pvsCapacityUniformBuffer;
    pvsBufferCapacityBufferInfo.offset = 0;
    pvsBufferCapacityBufferInfo.range = VK_WHOLE_SIZE;
    descriptorWrites[10].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptorWrites[10].dstSet = descriptorSet;
    descriptorWrites[10].dstBinding = 10;
    descriptorWrites[10].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    descriptorWrites[10].descriptorCount = 1;
    descriptorWrites[10].pBufferInfo = &pvsBufferCapacityBufferInfo;

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
    geometry.geometry.triangles.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
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
    //geometryInstance.flags = VK_GEOMETRY_INSTANCE_TRIANGLE_CULL_DISABLE_BIT_NV;
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
    createComputeDescriptorSetLayout();
    createHaltonComputeDescriptorSetLayout();

    std::array<VkDescriptorSetLayout, 4> d = {
        descriptorSetLayout, descriptorSetLayoutABS,
        descriptorSetLayoutCompute, descriptorSetLayoutHaltonCompute
    };
    VkDescriptorSetAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = descriptorPool;
    allocInfo.descriptorSetCount = 4;
    allocInfo.pSetLayouts = d.data();

    // Allocate descriptor sets
    std::array<VkDescriptorSet, 4> dd = {
        descriptorSet, descriptorSetABS,
        descriptorSetCompute, descriptorSetHaltonCompute
    };
    if (vkAllocateDescriptorSets(logicalDevice, &allocInfo, dd.data()) != VK_SUCCESS) {
        throw std::runtime_error("failed to allocate descriptor sets");
    }
    descriptorSet = dd[0];
    descriptorSetABS = dd[1];
    descriptorSetCompute = dd[2];
    descriptorSetHaltonCompute = dd[3];

    createDescriptorSets(indexBuffer, vertexBuffer, uniformBuffers);
    createRandomSamplingPipeline();
    createShaderBindingTable(shaderBindingTable, shaderBindingTableMemory, pipeline);

    createABSDescriptorSets(vertexBuffer);
    createABSPipeline();
    createShaderBindingTable(shaderBindingTableABS, shaderBindingTableMemoryABS, pipelineABS);

    createHaltonComputeDescriptorSets();

    createComputePipeline();
    createHaltonComputePipeline();

    // Calculate shader binding offsets
    bindingOffsetRayGenShader = rayTracingProperties.shaderGroupHandleSize * RT_SHADER_INDEX_RAYGEN;
    bindingOffsetMissShader = rayTracingProperties.shaderGroupHandleSize * RT_SHADER_INDEX_MISS;
    bindingOffsetHitShader = rayTracingProperties.shaderGroupHandleSize * RT_SHADER_INDEX_CLOSEST_HIT;
    bindingStride = rayTracingProperties.shaderGroupHandleSize;
}

void VisibilityManager::createBottomLevelAS(const VkGeometryNV *geometry) {
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
    memoryAllocateInfo.memoryTypeIndex = VulkanUtil::findMemoryType(
        physicalDevice,
        memoryRequirements2.memoryRequirements.memoryTypeBits,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
    );
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
    memoryAllocateInfo.memoryTypeIndex = VulkanUtil::findMemoryType(
        physicalDevice,
        memoryRequirements2.memoryRequirements.memoryTypeBits,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
    );
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

    VkCommandBuffer commandBuffer = VulkanUtil::beginSingleTimeCommands(logicalDevice, commandPool);

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
        logicalDevice, commandBuffer, commandPool, computeQueue
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

    VkDescriptorSetLayoutBinding pvsBufferBinding = {};
    pvsBufferBinding.binding = 9;
    pvsBufferBinding.descriptorCount = 1;
    pvsBufferBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    pvsBufferBinding.stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_NV;

    VkDescriptorSetLayoutBinding pvsBufferCapacityBinding = {};
    pvsBufferCapacityBinding.binding = 10;
    pvsBufferCapacityBinding.descriptorCount = 1;
    pvsBufferCapacityBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    pvsBufferCapacityBinding.stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_NV;

    std::array<VkDescriptorSetLayoutBinding, 11> bindings = {
        aslayoutBinding,
        uniformLayoutBinding,
        vertexLayoutBinding,
        indexLayoutBinding,
        haltonPointsBinding,
        viewCellBinding,
        randomSamplingOutputBinding,
        triangleBufferBinding,
        triangleCounterBufferBinding,
        pvsBufferBinding,
        pvsBufferCapacityBinding
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
    poolInfo.maxSets = 5;

    if (vkCreateDescriptorPool(logicalDevice, &poolInfo, nullptr, &descriptorPool) != VK_SUCCESS) {
        throw std::runtime_error("failed to create rt descriptor pool");
    }
}

void VisibilityManager::createPipeline(
    const std::array<VkPipelineShaderStageCreateInfo, 3> &shaderStages, VkPipelineLayout &pipelineLayout,
    VkPipeline &pipeline, const std::vector<VkDescriptorSetLayout> &descriptorSetLayouts,
    std::vector<VkPushConstantRange> pushConstantRanges
) {
    VkPipelineLayoutCreateInfo pipelineLayoutInfo = {};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = static_cast<uint32_t>(descriptorSetLayouts.size());
    pipelineLayoutInfo.pSetLayouts = descriptorSetLayouts.data();
    pipelineLayoutInfo.pushConstantRangeCount = static_cast<uint32_t>(pushConstantRanges.size());
    pipelineLayoutInfo.pPushConstantRanges = pushConstantRanges.data();
    if (vkCreatePipelineLayout(
            logicalDevice, &pipelineLayoutInfo, nullptr, &pipelineLayout
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
    rtPipelineInfo.layout = pipelineLayout;
    if (vkCreateRayTracingPipelinesNV(
            logicalDevice, VK_NULL_HANDLE, 1, &rtPipelineInfo, nullptr, &pipeline
        ) != VK_SUCCESS
    ) {
        throw std::runtime_error("failed to create rt pipeline");
    }
}

void VisibilityManager::createRandomSamplingPipeline() {
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

    createPipeline(shaderStages, pipelineLayout, pipeline, { descriptorSetLayout }, { });

    vkDestroyShaderModule(logicalDevice, rayGenShaderStageInfo.module, nullptr);
    vkDestroyShaderModule(logicalDevice, rayClosestHitShaderStageInfo.module, nullptr);
    vkDestroyShaderModule(logicalDevice, rayMissShaderStageInfo.module, nullptr);
}

void VisibilityManager::createABSPipeline() {
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
    rayMissShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO; rayMissShaderStageInfo.stage = VK_SHADER_STAGE_MISS_BIT_NV;
    rayMissShaderStageInfo.module = VulkanUtil::createShader(logicalDevice, "shaders/rt/raytrace_abs.rmiss.spv");
    rayMissShaderStageInfo.pName = "main";

    std::array<VkPipelineShaderStageCreateInfo, 3> shaderStages = {};
    shaderStages[RT_SHADER_INDEX_RAYGEN] = rayGenShaderStageInfo;
    shaderStages[RT_SHADER_INDEX_CLOSEST_HIT] = rayClosestHitShaderStageInfo;
    shaderStages[RT_SHADER_INDEX_MISS] = rayMissShaderStageInfo;

    createPipeline(
        shaderStages, pipelineABSLayout, pipelineABS,
        { descriptorSetLayout, descriptorSetLayoutABS }, { }
    );

    vkDestroyShaderModule(logicalDevice, rayGenShaderStageInfo.module, nullptr);
    vkDestroyShaderModule(logicalDevice, rayClosestHitShaderStageInfo.module, nullptr);
    vkDestroyShaderModule(logicalDevice, rayMissShaderStageInfo.module, nullptr);
}

void VisibilityManager::createComputePipeline() {
    // Compute shader used for resizing the PVS buffer, if a hash set is used
    VkPipelineShaderStageCreateInfo pipelineShaderStageCreateInfo = {};
    pipelineShaderStageCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    pipelineShaderStageCreateInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    pipelineShaderStageCreateInfo.module = VulkanUtil::createShader(logicalDevice, "shaders/gpuHashSetBulkInsert.comp.spv");
    pipelineShaderStageCreateInfo.pName = "main";

    VkPipelineLayoutCreateInfo pipelineLayoutInfo = {};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = 1;
    pipelineLayoutInfo.pSetLayouts = &descriptorSetLayoutCompute;
    if (vkCreatePipelineLayout(
            logicalDevice, &pipelineLayoutInfo, nullptr, &pipelineComputeLayout
        ) != VK_SUCCESS
    ) {
        throw std::runtime_error("failed to create compute pipeline layout");
    }

    VkComputePipelineCreateInfo computePipelineCreateInfo = {};
    computePipelineCreateInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    computePipelineCreateInfo.stage = pipelineShaderStageCreateInfo;
    computePipelineCreateInfo.layout = pipelineComputeLayout;

    if (vkCreateComputePipelines(
            logicalDevice, 0, 1, &computePipelineCreateInfo, nullptr, &pipelineCompute
        ) != VK_SUCCESS
    ) {
        throw std::runtime_error("failed to create compute pipeline");
    }
}

void VisibilityManager::createHaltonComputeDescriptorSets() {
    std::array<VkWriteDescriptorSet, 1> descriptorWrites = {};

    VkDescriptorBufferInfo haltonPointsBufferInfo = {};
    haltonPointsBufferInfo.buffer = haltonPointsBuffer;
    haltonPointsBufferInfo.offset = 0;
    haltonPointsBufferInfo.range = VK_WHOLE_SIZE;
    descriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptorWrites[0].dstSet = descriptorSetHaltonCompute;
    descriptorWrites[0].dstBinding = 0;
    descriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    descriptorWrites[0].descriptorCount = 1;
    descriptorWrites[0].pBufferInfo = &haltonPointsBufferInfo;

    vkUpdateDescriptorSets(
        logicalDevice,
        static_cast<uint32_t>(descriptorWrites.size()),
        descriptorWrites.data(),
        0,
        VK_NULL_HANDLE
    );
}

void VisibilityManager::createHaltonComputeDescriptorSetLayout() {
    VkDescriptorSetLayoutBinding haltonPointsBinding = {};
    haltonPointsBinding.binding = 0;
    haltonPointsBinding.descriptorCount = 1;
    haltonPointsBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    haltonPointsBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    std::array<VkDescriptorSetLayoutBinding, 1> bindings = {
        haltonPointsBinding
    };
    VkDescriptorSetLayoutCreateInfo layoutInfo = {};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
    layoutInfo.pBindings = bindings.data();

    if (vkCreateDescriptorSetLayout(
            logicalDevice, &layoutInfo, nullptr, &descriptorSetLayoutHaltonCompute
        ) != VK_SUCCESS
    ) {
        throw std::runtime_error("failed to create descriptor set layout Halton compute");
    }
}

void VisibilityManager::createHaltonComputePipeline() {
    VkPipelineShaderStageCreateInfo pipelineShaderStageCreateInfo = {};
    pipelineShaderStageCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    pipelineShaderStageCreateInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    pipelineShaderStageCreateInfo.module = VulkanUtil::createShader(logicalDevice, "shaders/halton.comp.spv");
    pipelineShaderStageCreateInfo.pName = "main";

    std::array<VkPushConstantRange, 1> pushConstantRanges = {};
    pushConstantRanges[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pushConstantRanges[0].size = sizeof(glm::vec4) + sizeof(int);
    pushConstantRanges[0].offset = 0;

    VkPipelineLayoutCreateInfo pipelineLayoutInfo = {};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = 1;
    pipelineLayoutInfo.pSetLayouts = &descriptorSetLayoutHaltonCompute;
    pipelineLayoutInfo.pushConstantRangeCount = static_cast<uint32_t>(pushConstantRanges.size());
    pipelineLayoutInfo.pPushConstantRanges = pushConstantRanges.data();
    if (vkCreatePipelineLayout(
            logicalDevice, &pipelineLayoutInfo, nullptr, &pipelineHaltonComputeLayout
        ) != VK_SUCCESS
    ) {
        throw std::runtime_error("failed to create halton compute pipeline layout");
    }

    VkComputePipelineCreateInfo computePipelineCreateInfo = {};
    computePipelineCreateInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    computePipelineCreateInfo.stage = pipelineShaderStageCreateInfo;
    computePipelineCreateInfo.layout = pipelineHaltonComputeLayout;

    if (vkCreateComputePipelines(
            logicalDevice, 0, 1, &computePipelineCreateInfo, nullptr, &pipelineHaltonCompute
        ) != VK_SUCCESS
    ) {
        throw std::runtime_error("failed to create halton compute pipeline");
    }
}

void VisibilityManager::createABSDescriptorSetLayout() {
    VkDescriptorSetLayoutBinding absWorkingBufferBinding = {};
    absWorkingBufferBinding.binding = 0;
    absWorkingBufferBinding.descriptorCount = 1;
    absWorkingBufferBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    absWorkingBufferBinding.stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_NV;

    std::array<VkDescriptorSetLayoutBinding, 1> bindings = {
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

void VisibilityManager::createComputeDescriptorSets() {
    std::array<VkWriteDescriptorSet, 3> descriptorWrites = {};

    VkDescriptorBufferInfo pvsBulkInsertBufferInfo = {};
    pvsBulkInsertBufferInfo.buffer = pvsBulkInsertBuffer;
    pvsBulkInsertBufferInfo.offset = 0;
    pvsBulkInsertBufferInfo.range = VK_WHOLE_SIZE;
    descriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptorWrites[0].dstSet = descriptorSetCompute;
    descriptorWrites[0].dstBinding = 0;
    descriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    descriptorWrites[0].descriptorCount = 1;
    descriptorWrites[0].pBufferInfo = &pvsBulkInsertBufferInfo;

    VkDescriptorBufferInfo testBufferInfo = {};
    testBufferInfo.buffer = pvsBuffer;
    testBufferInfo.offset = 0;
    testBufferInfo.range = VK_WHOLE_SIZE;
    descriptorWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptorWrites[1].dstSet = descriptorSetCompute;
    descriptorWrites[1].dstBinding = 1;
    descriptorWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    descriptorWrites[1].descriptorCount = 1;
    descriptorWrites[1].pBufferInfo = &testBufferInfo;

    VkDescriptorBufferInfo pvsBufferCapacityBufferInfo = {};
    pvsBufferCapacityBufferInfo.buffer = pvsCapacityUniformBuffer;
    pvsBufferCapacityBufferInfo.offset = 0;
    pvsBufferCapacityBufferInfo.range = VK_WHOLE_SIZE;
    descriptorWrites[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptorWrites[2].dstSet = descriptorSetCompute;
    descriptorWrites[2].dstBinding = 2;
    descriptorWrites[2].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    descriptorWrites[2].descriptorCount = 1;
    descriptorWrites[2].pBufferInfo = &pvsBufferCapacityBufferInfo;

    vkUpdateDescriptorSets(
        logicalDevice,
        static_cast<uint32_t>(descriptorWrites.size()),
        descriptorWrites.data(),
        0,
        VK_NULL_HANDLE
    );
}

void VisibilityManager::createComputeDescriptorSetLayout() {
    VkDescriptorSetLayoutBinding pvsBulkInsertBufferBinding = {};
    pvsBulkInsertBufferBinding.binding = 0;
    pvsBulkInsertBufferBinding.descriptorCount = 1;
    pvsBulkInsertBufferBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    pvsBulkInsertBufferBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    VkDescriptorSetLayoutBinding testBufferBinding = {};
    testBufferBinding.binding = 1;
    testBufferBinding.descriptorCount = 1;
    testBufferBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    testBufferBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    VkDescriptorSetLayoutBinding pvsCapacityUniformBufferLayoutBinding = {};
    pvsCapacityUniformBufferLayoutBinding.binding = 2;
    pvsCapacityUniformBufferLayoutBinding.descriptorCount = 1;
    pvsCapacityUniformBufferLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    pvsCapacityUniformBufferLayoutBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    std::array<VkDescriptorSetLayoutBinding, 3> bindings = {
        pvsBulkInsertBufferBinding,
        testBufferBinding,
        pvsCapacityUniformBufferLayoutBinding
    };
    VkDescriptorSetLayoutCreateInfo layoutInfo = {};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
    layoutInfo.pBindings = bindings.data();

    if (vkCreateDescriptorSetLayout(
            logicalDevice, &layoutInfo, nullptr, &descriptorSetLayoutCompute
        ) != VK_SUCCESS
    ) {
        throw std::runtime_error("failed to create descriptor set layout compute");
    }
}

void VisibilityManager::createABSDescriptorSets(VkBuffer vertexBuffer) {
    std::array<VkWriteDescriptorSet, 1> descriptorWrites = {};

    VkDescriptorBufferInfo absWorkingBufferInfo = {};
    absWorkingBufferInfo.buffer = absWorkingBuffer;
    absWorkingBufferInfo.offset = 0;
    absWorkingBufferInfo.range = VK_WHOLE_SIZE;
    descriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptorWrites[0].dstSet = descriptorSetABS;
    descriptorWrites[0].dstBinding = 0;
    descriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    descriptorWrites[0].descriptorCount = 1;
    descriptorWrites[0].pBufferInfo = &absWorkingBufferInfo;

    vkUpdateDescriptorSets(
        logicalDevice,
        static_cast<uint32_t>(descriptorWrites.size()),
        descriptorWrites.data(),
        0,
        VK_NULL_HANDLE
    );
}

ShaderExecutionInfo VisibilityManager::randomSample(int numRays, int viewCellIndex) {
    VkCommandBufferBeginInfo beginInfo = {};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    vkBeginCommandBuffer(commandBuffer, &beginInfo);
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_RAY_TRACING_NV, pipeline);
    vkCmdBindDescriptorSets(
        commandBuffer, VK_PIPELINE_BIND_POINT_RAY_TRACING_NV, pipelineLayout, 0, 1,
        &descriptorSet, 0, nullptr
    );

    vkCmdTraceRaysNV(
        commandBuffer,
        shaderBindingTable, bindingOffsetRayGenShader,
        shaderBindingTable, bindingOffsetMissShader, bindingStride,
        shaderBindingTable, bindingOffsetHitShader, bindingStride,
        VK_NULL_HANDLE, 0, 0,
        numRays, 1, 1
    );
    vkEndCommandBuffer(commandBuffer);

    VulkanUtil::executeCommandBuffer(
        logicalDevice, computeQueue, commandBuffer, commandBufferFence
    );

    // Get number of intersected triangles from the GPU
    unsigned int numTriangles = 0;
    {
        VkDeviceSize bufferSize = sizeof(unsigned int) * 5;

        VkBuffer hostBuffer;
        VkDeviceMemory hostBufferMemory;
        VulkanUtil::createBuffer(
            physicalDevice,
            logicalDevice,
            bufferSize,
            VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            hostBuffer,
            hostBufferMemory,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT
        );

        VulkanUtil::copyBuffer(
            logicalDevice, transferCommandPool, transferQueue, triangleCounterBuffer,
            hostBuffer, bufferSize
        );

        void *data;
        vkMapMemory(logicalDevice, hostBufferMemory, 0, bufferSize, 0, &data);
        unsigned int *n = (unsigned int*) data;
        numTriangles += n[0];
        pvsSize = n[4];

        // Reset atomic counters
        for (int i = 0; i < 4; i++) {
            n[i] = 0;
        }

        VulkanUtil::copyBuffer(
            logicalDevice, transferCommandPool, transferQueue, hostBuffer,
            triangleCounterBuffer, bufferSize
        );

        vkUnmapMemory(logicalDevice, hostBufferMemory);
        vkDestroyBuffer(logicalDevice, hostBuffer, nullptr);
        vkFreeMemory(logicalDevice, hostBufferMemory, nullptr);
    }

    if (visualizeFirstRays) {
        Sample *s = (Sample*)randomSamplingOutputPointer;
        for (int i = 0; i < numTriangles; i++) {
            rayVertices[viewCellIndex].push_back({s[i].rayOrigin, glm::vec3(0.0f), glm::vec3(1.0f, 1.0f, 0.0f), glm::vec3(0.0f)});
            rayVertices[viewCellIndex].push_back({s[i].hitPos, glm::vec3(0.0f), glm::vec3(0.0f, 0.0f, 1.0f), glm::vec3(0.0f)});
        }
    }

    return { numTriangles, 0, (unsigned int) numRays, 0 };
}

ShaderExecutionInfo VisibilityManager::adaptiveBorderSample(const std::vector<Sample> &triangles, int viewCellIndex) {
    // Copy triangles vector to GPU accessible buffer
    {
        VkDeviceSize bufferSize = sizeof(triangles[0]) * triangles.size();

        // Create staging buffer using host-visible memory
        VkBuffer stagingBuffer;
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
            logicalDevice, transferCommandPool, transferQueue, stagingBuffer, absWorkingBuffer,
            bufferSize
        );

        vkDestroyBuffer(logicalDevice, stagingBuffer, nullptr);
        vkFreeMemory(logicalDevice, stagingBufferMemory, nullptr);
    }

    // Record and execute a command buffer for running the actual ABS on the GPU
    VkCommandBufferBeginInfo beginInfo = {};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(commandBufferABS, &beginInfo);
    vkCmdBindPipeline(commandBufferABS, VK_PIPELINE_BIND_POINT_RAY_TRACING_NV, pipelineABS);
    vkCmdBindDescriptorSets(
        commandBufferABS, VK_PIPELINE_BIND_POINT_RAY_TRACING_NV, pipelineABSLayout, 0, 1,
        &descriptorSet, 0, nullptr
    );
    vkCmdBindDescriptorSets(
        commandBufferABS, VK_PIPELINE_BIND_POINT_RAY_TRACING_NV, pipelineABSLayout, 1, 1,
        &descriptorSetABS, 0, nullptr
    );
    vkCmdTraceRaysNV(
        commandBufferABS,
        shaderBindingTableABS, bindingOffsetRayGenShader,
        shaderBindingTableABS, bindingOffsetMissShader, bindingStride,
        shaderBindingTableABS, bindingOffsetHitShader, bindingStride,
        VK_NULL_HANDLE, 0, 0,
        triangles.size() * NUM_ABS_SAMPLES, 1, 1
    );

    vkEndCommandBuffer(commandBufferABS);
    VulkanUtil::executeCommandBuffer(
        logicalDevice, computeQueue, commandBufferABS, commandBufferFence
    );

    // Get number of intersected triangles from the GPU
    unsigned int numTriangles;
    unsigned int numRsTriangles;
    unsigned int numRays = triangles.size() * NUM_ABS_SAMPLES;
    unsigned int numRsRays;
    {
        VkDeviceSize bufferSize = sizeof(unsigned int) * 5;

        VkBuffer hostBuffer;
        VkDeviceMemory hostBufferMemory;
        VulkanUtil::createBuffer(
            physicalDevice,
            logicalDevice, bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT, hostBuffer, hostBufferMemory,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT
        );

        VulkanUtil::copyBuffer(
            logicalDevice, transferCommandPool, transferQueue, triangleCounterBuffer,
            hostBuffer, bufferSize
        );

        void *data;
        vkMapMemory(logicalDevice, hostBufferMemory, 0, bufferSize, 0, &data);
        unsigned int *n = (unsigned int*) data;
        numRsTriangles = n[1];
        numTriangles = n[0] - numRsTriangles;       // In this case, n[0] contains the number of ALL triangles (rs and non-rs)
        numRsRays = n[3];
        pvsSize = n[4];

        // Reset atomic counters
        for (int i = 0; i < 4; i++) {
            n[i] = 0;
        }

        VulkanUtil::copyBuffer(
            logicalDevice, transferCommandPool, transferQueue, hostBuffer,
            triangleCounterBuffer, bufferSize
        );

        vkUnmapMemory(logicalDevice, hostBufferMemory);
        vkDestroyBuffer(logicalDevice, hostBuffer, nullptr);
        vkFreeMemory(logicalDevice, hostBufferMemory, nullptr);
    }

    if (visualizeFirstRays) {
        // Copy intersected triangles from VRAM to CPU accessible buffer
        Sample *s = (Sample*)randomSamplingOutputPointer;
        // Visualize ABS rays
        for (int i = 0; i < numTriangles + numRsTriangles; i++) {
            if (s[i].triangleID != -1) {
                rayVertices[viewCellIndex].push_back({
                    s[i].rayOrigin, glm::vec3(0.0f), glm::vec3(1.0f, 1.0f, 0.0f), glm::vec3(0.0f)
                });
                rayVertices[viewCellIndex].push_back({
                    s[i].hitPos, glm::vec3(0.0f), glm::vec3(0.0f, 0.0f, 1.0f), glm::vec3(0.0f)
                });
            }
        }
    }

    return { numTriangles, numRsTriangles, numRays, numRsRays};
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

// Compute the actual PVS
void VisibilityManager::rayTrace(int viewCellIndex) {
    updateViewCellBuffer(viewCellIndex);
    resetPVSGPUBuffer();
    resetAtomicBuffers();
    //gpuHashSet->reset();
    statistics.push_back(Statistics(1000000));
    int terminationThresholdCounter = 0;

    std::vector<Sample> absSampleQueue;
    size_t previousPVSSize;
    statistics.back().startOperation(VISIBILITY_SAMPLING);
    for (int i = 0; true; i++) {
        previousPVSSize = pvsSize;

        // Check GPU hash set size (if used)
        if (GPU_SET_TYPE == 1) {
            statistics.back().startOperation(GPU_HASH_SET_RESIZE);
            int potentialNewTriangles = std::min(RANDOM_RAYS_PER_ITERATION, MAX_TRIANGLE_COUNT - pvsSize);
            if (pvsBufferCapacity - pvsSize < potentialNewTriangles) {
                resizeHashSetPVSBuffer(1 << int(std::ceil(std::log2(pvsSize + potentialNewTriangles))));
            }
            statistics.back().endOperation(GPU_HASH_SET_RESIZE);
        }

        {
            // Execute random sampling
            statistics.back().startOperation(RANDOM_SAMPLING);
            ShaderExecutionInfo randomSampleInfo = randomSample(RANDOM_RAYS_PER_ITERATION / 1.0f, viewCellIndex);
            statistics.back().endOperation(RANDOM_SAMPLING);

            statistics.back().entries.back().numShaderExecutions += RANDOM_RAYS_PER_ITERATION;
            statistics.back().entries.back().rnsTris += randomSampleInfo.numTriangles;
            statistics.back().entries.back().rnsRays += randomSampleInfo.numRays;

            statistics.back().startOperation(RANDOM_SAMPLING_INSERT);

            if (randomSampleInfo.numTriangles > 0) {
                // Copy intersected triangles from VRAM to CPU accessible buffer
                Sample *s = (Sample*)randomSamplingOutputPointer;
                absSampleQueue.insert(absSampleQueue.end(), s, s + randomSampleInfo.numTriangles);
            }
            statistics.back().endOperation(RANDOM_SAMPLING_INSERT);
        }

        statistics.back().entries.back().pvsSize = pvsSize;
        statistics.back().update();

        // Adaptive Border Sampling. ABS is executed for a maximum of MAX_ABS_TRIANGLES_PER_ITERATION rays at a time as
        // long as there are unprocessed triangles left i.e. absSampleQueue is not empty
        while (absSampleQueue.size() >= MIN_ABS_TRIANGLES_PER_ITERATION) {
            // Get numAbsRays samples from the queue
            const int numAbsRays = std::min(MAX_ABS_TRIANGLES_PER_ITERATION, (long)absSampleQueue.size());
            std::vector<Sample> absWorkingVector;
            if (numAbsRays == absSampleQueue.size()) {
                absWorkingVector = absSampleQueue;
                absSampleQueue.clear();
            } else {
                absWorkingVector.reserve(numAbsRays);
                absWorkingVector.insert(
                    absWorkingVector.end(),
                    std::make_move_iterator(absSampleQueue.end() - numAbsRays),
                    std::make_move_iterator(absSampleQueue.end())
                );
                absSampleQueue.erase(absSampleQueue.end() - numAbsRays, absSampleQueue.end());
            }

            // Check GPU hash set size (if used)
            if (GPU_SET_TYPE == 1) {
                statistics.back().startOperation(GPU_HASH_SET_RESIZE);
                int potentialNewTriangles = std::min(
                    absWorkingVector.size() * NUM_ABS_SAMPLES * NUM_REVERSE_SAMPLING_SAMPLES,
                    (size_t)MAX_TRIANGLE_COUNT - pvsSize
                );
                if (pvsBufferCapacity - pvsSize < potentialNewTriangles) {
                    resizeHashSetPVSBuffer(1 << int(std::ceil(std::log2(pvsSize + potentialNewTriangles))));
                }
                statistics.back().endOperation(GPU_HASH_SET_RESIZE);
            }

            // Execute adaptive border sampling
            statistics.back().startOperation(ADAPTIVE_BORDER_SAMPLING);
            ShaderExecutionInfo absInfo = adaptiveBorderSample(absWorkingVector, viewCellIndex);
            statistics.back().endOperation(ADAPTIVE_BORDER_SAMPLING);

            statistics.back().entries.back().numShaderExecutions += absWorkingVector.size() * NUM_ABS_SAMPLES;
            statistics.back().entries.back().absRays += absInfo.numRays;
            statistics.back().entries.back().absRsRays += absInfo.numRsRays;
            statistics.back().entries.back().absTris += absInfo.numTriangles;
            statistics.back().entries.back().absRsTris += absInfo.numRsTriangles;

            if (absInfo.numTriangles + absInfo.numRsTriangles > 0) {
                // Copy intersected triangles from VRAM to CPU accessible buffer
                statistics.back().startOperation(ADAPTIVE_BORDER_SAMPLING_INSERT);
                Sample *s = (Sample*)randomSamplingOutputPointer;
                absSampleQueue.insert(absSampleQueue.end(), s, s + absInfo.numTriangles + absInfo.numRsTriangles);
                statistics.back().endOperation(ADAPTIVE_BORDER_SAMPLING_INSERT);
            }

            statistics.back().entries.back().pvsSize = pvsSize;
            statistics.back().update();
        }

        if (pvsSize - previousPVSSize < NEW_TRIANGLE_TERMINATION_THRESHOLD) {
            terminationThresholdCounter++;
        } else {
            terminationThresholdCounter = 0;
        }

        // Terminate, if no more than NEW_TRIANGLE_TERMINATION_THRESHOLD new triangles have been found during each of
        // the last NEW_TRIANGLE_TERMINATION_THRESHOLD_COUNT iterations
        if (terminationThresholdCounter == NEW_TRIANGLE_TERMINATION_THRESHOLD_COUNT) {
            statistics.back().endOperation(VISIBILITY_SAMPLING);
            statistics.back().print();
            break;
        }

        // Generate new Halton points
        statistics.back().startOperation(HALTON_GENERATION);
        generateHaltonSequence(RANDOM_RAYS_PER_ITERATION, rand() / float(RAND_MAX));
        statistics.back().endOperation(HALTON_GENERATION);
    }
}

void VisibilityManager::releaseResources() {
    vkDestroyBuffer(logicalDevice, haltonPointsBuffer, nullptr);
    vkFreeMemory(logicalDevice, haltonPointsBufferMemory, nullptr);

    vkDestroyBuffer(logicalDevice, viewCellBuffer, nullptr);
    vkFreeMemory(logicalDevice, viewCellBufferMemory, nullptr);

    vkDestroyBuffer(logicalDevice, randomSamplingOutputBuffer, nullptr);
    vkFreeMemory(logicalDevice, randomSamplingOutputBufferMemory, nullptr);
    vkUnmapMemory(logicalDevice, randomSamplingOutputHostBufferMemory);
    vkDestroyBuffer(logicalDevice, randomSamplingOutputHostBuffer, nullptr);
    vkFreeMemory(logicalDevice, randomSamplingOutputHostBufferMemory, nullptr);

    vkDestroyBuffer(logicalDevice, absWorkingBuffer, nullptr);
    vkFreeMemory(logicalDevice, absWorkingBufferMemory, nullptr);

    vkDestroyBuffer(logicalDevice, triangleCounterBuffer, nullptr);
    vkFreeMemory(logicalDevice, triangleCounterBufferMemory, nullptr);

    vkDestroyBuffer(logicalDevice, pvsBulkInsertBuffer, nullptr);
    vkFreeMemory(logicalDevice, pvsBulkInsertBufferMemory, nullptr);

    vkDestroyBuffer(logicalDevice, pvsBuffer, nullptr);
    vkFreeMemory(logicalDevice, pvsBufferMemory, nullptr);

    vkDestroyFence(logicalDevice, commandBufferFence, nullptr);

    VkCommandBuffer commandBuffers[] = {
        commandBuffer,
        commandBufferABS,
        commandBufferCompute,
        commandBufferHaltonCompute
    };
    vkFreeCommandBuffers(logicalDevice, commandPool, 4, commandBuffers);

    vkDestroyCommandPool(logicalDevice, commandPool, nullptr);

    vkDestroyBuffer(logicalDevice, pvsCapacityUniformBuffer, nullptr);
    vkFreeMemory(logicalDevice, pvsCapacityUniformMemory, nullptr);

    vkDestroyBuffer(logicalDevice, shaderBindingTable, nullptr);
    vkFreeMemory(logicalDevice, shaderBindingTableMemory, nullptr);
    vkDestroyBuffer(logicalDevice, shaderBindingTableABS, nullptr);
    vkFreeMemory(logicalDevice, shaderBindingTableMemoryABS, nullptr);

    vkDestroyDescriptorSetLayout(logicalDevice, descriptorSetLayout, nullptr);
    vkDestroyDescriptorSetLayout(logicalDevice, descriptorSetLayoutABS, nullptr);
    vkDestroyDescriptorSetLayout(logicalDevice, descriptorSetLayoutCompute, nullptr);
    vkDestroyDescriptorSetLayout(logicalDevice, descriptorSetLayoutHaltonCompute, nullptr);

    vkDestroyPipeline(logicalDevice, pipeline, nullptr);
    vkDestroyPipelineLayout(logicalDevice, pipelineLayout, nullptr);
    vkDestroyPipeline(logicalDevice, pipelineABS, nullptr);
    vkDestroyPipelineLayout(logicalDevice, pipelineABSLayout, nullptr);

    vkDestroyPipeline(logicalDevice, pipelineCompute, nullptr);
    vkDestroyPipelineLayout(logicalDevice, pipelineComputeLayout, nullptr);
    vkDestroyPipeline(logicalDevice, pipelineHaltonCompute, nullptr);
    vkDestroyPipelineLayout(logicalDevice, pipelineHaltonComputeLayout, nullptr);

    vkDestroyDescriptorPool(logicalDevice, descriptorPool, nullptr);

    vkDestroyAccelerationStructureNV(logicalDevice, topLevelAS.as, nullptr);
    vkFreeMemory(logicalDevice, topLevelAS.deviceMemory, nullptr);
    vkDestroyAccelerationStructureNV(logicalDevice, bottomLevelAS.as, nullptr);
    vkFreeMemory(logicalDevice, bottomLevelAS.deviceMemory, nullptr);
}

// Get the PVS from the GPU
void VisibilityManager::fetchPVS() {
    VkDeviceSize bufferSize = sizeof(int) * pvsBufferCapacity;

    VkBuffer hostBuffer;
    VkDeviceMemory hostBufferMemory;
    VulkanUtil::createBuffer(
        physicalDevice,
        logicalDevice, bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT, hostBuffer, hostBufferMemory,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT
    );

    VulkanUtil::copyBuffer(
        logicalDevice, transferCommandPool, transferQueue, pvsBuffer,
        hostBuffer, bufferSize
    );

    void *data;
    vkMapMemory(logicalDevice, hostBufferMemory, 0, bufferSize, 0, &data);

    pvs.pvsVector.clear();
    int* pvsArray = (int*)data;
    for (int i = 0; i < pvsBufferCapacity; i++) {
        if (pvsArray[i] >= 0) {
            pvs.pvsVector.push_back(pvsArray[i]);
        }
    }

    vkDestroyBuffer(logicalDevice, hostBuffer, nullptr);
    vkFreeMemory(logicalDevice, hostBufferMemory, nullptr);
}

void VisibilityManager::createCommandBuffers() {
    VkCommandBufferAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.commandPool = commandPool;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = 1;

    if (vkAllocateCommandBuffers(logicalDevice, &allocInfo, &commandBuffer) != VK_SUCCESS) {
        throw std::runtime_error("failed to allocate rt command buffer!");
    }

    if (vkAllocateCommandBuffers(logicalDevice, &allocInfo, &commandBufferABS) != VK_SUCCESS) {
        throw std::runtime_error("failed to allocate rt command buffer abs!");
    }

    if (vkAllocateCommandBuffers(logicalDevice, &allocInfo, &commandBufferCompute) != VK_SUCCESS) {
        throw std::runtime_error("failed to allocate command buffer compute!");
    }

    if (vkAllocateCommandBuffers(logicalDevice, &allocInfo, &commandBufferHaltonCompute) != VK_SUCCESS) {
        throw std::runtime_error("failed to allocate command buffer Halton compute!");
    }

    // Create fence used to wait for command buffer execution completion after submitting them
    VkFenceCreateInfo fenceInfo;
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fenceInfo.pNext = NULL;
    fenceInfo.flags = 0;
    vkCreateFence(logicalDevice, &fenceInfo, NULL, &commandBufferFence);
}
