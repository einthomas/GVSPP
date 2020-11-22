#pragma once

#include <vulkan/vulkan.h>
#include <vulkan/vulkan_core.h>
#include <vector>
#include <array>
#include <unordered_set>
#include <thread>
#include <mutex>
#include <atomic>
#include <random>
#include <map>

#include <glm/vec3.hpp>

#include "Vertex.h"
#include "sample.h"
#include "pvs.h"
#include "Statistics.h"
//#include "CUDAUtil.h"
#include "gpuHashTable/linearprobing.h"

#include <algorithm>
#include <math.h>
#include <cmath>

class ViewCell;

struct AccelerationStructure {
    VkDeviceMemory deviceMemory;
    VkAccelerationStructureNV as;
    uint64_t handle;
};

// Ray tracing geometry instance
struct GeometryInstance {
    glm::mat3x4 transform;
    uint32_t instanceId : 24;
    uint32_t mask : 8;
    uint32_t instanceOffset : 24;
    uint32_t flags : 8;
    uint64_t accelerationStructureHandle;
};

struct ShaderExecutionInfo {
    unsigned int numTriangles;
    unsigned int numRsTriangles;
    unsigned int numRays;
    unsigned int numRsRays;
};

class RasterVisibility;

class VisibilityManager {
public:
    std::vector<ViewCell> viewCells;
    PVS<int> pvs;
    std::vector<std::vector<Vertex>> rayVertices;
    bool visualizeRandomRays = false;
    bool visualizeABSRays = false;
    bool visualizeEdgeSubdivRays = false;
    VkQueue computeQueue;
    VkQueue transferQueue;
    std::vector<VkCommandPool> commandPool;
    VkCommandPool transferCommandPool;
    std::vector<Statistics> statistics;
    std::vector<VkBuffer> randomSamplingOutputBuffer;
    std::vector<VkBuffer> pvsBuffer;
    std::vector<VkBuffer> triangleCounterBuffer;

    VisibilityManager(
        bool USE_TERMINATION_CRITERION,
        bool USE_RECURSIVE_EDGE_SUBDIVISION,
        bool USE_HYBRID_VISIBILITY_SAMPLING,
        long RASTER_NUM_HEMICUBES,
        long NEW_TRIANGLE_TERMINATION_THRESHOLD_COUNT,
        long NEW_TRIANGLE_TERMINATION_THRESHOLD,
        long RANDOM_RAYS_PER_ITERATION,
        long ABS_MAX_SUBDIVISION_STEPS,
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
        int numThreads,
        std::array<uint8_t, VK_UUID_SIZE> deviceUUID,
        std::vector<ViewCell> viewCells,
        VkCommandPool graphicsCommandPool,
        VkQueue graphicsQueue,
        uint32_t frameBufferWidth,
        uint32_t frameBufferHeight,
        VkFormat depthFormat
    );
    ~VisibilityManager();

    template<int T>
    std::vector<glm::vec<T, float, glm::defaultp>> generateHaltonPoints2d(std::array<int, T> bases, int n, std::array<float, T> lastHaltonPoints) {
        std::vector<glm::vec<T, float, glm::defaultp>> haltonPoints;

        //int bases[4] = { 2, 3, 5, 7 };

        //haltonPoints.clear();
        haltonPoints.resize(n);


        /*
            This is the incremental version to generate the halton squence of
            quasi-random numbers of a given base. It has been taken from:
            Keller, Alexander. "Instant radiosity." Proceedings of the 24th annual conference on Computer graphics and interactive techniques. 1997.

            Train, Kenneth E. Discrete choice methods with simulation. Cambridge university press, 2009.
        */
        for (int k = 0; k < bases.size(); k++) {
            double inverseBase = 1.0 / bases[k];
            double value = 0.0;

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

                haltonPoints[i][k] = value;
            }
        }

        return haltonPoints;
    }
    void rayTrace(const std::vector<uint32_t> &indices, int threadId, int viewCellIndex);
    void releaseResources();
    void fetchPVS();
    void printAverageStatistics();

private:
    int pvsSize = 0;
    int numThreads;

    const bool USE_TERMINATION_CRITERION;
    const bool USE_RECURSIVE_EDGE_SUBDIVISION;
    const bool USE_HYBRID_VISIBILITY_SAMPLING;
    const long RASTER_NUM_HEMICUBES;
    const long NEW_TRIANGLE_TERMINATION_THRESHOLD;
    const long NEW_TRIANGLE_TERMINATION_THRESHOLD_COUNT;
    const long NUM_ABS_SAMPLES;
    const long NUM_REVERSE_SAMPLING_SAMPLES;
    const long MAX_BULK_INSERT_BUFFER_SIZE;
    const long MAX_TRIANGLE_COUNT;
    const int GPU_SET_TYPE;

    const long RANDOM_RAYS_PER_ITERATION;
    const int MIN_ABS_TRIANGLES_PER_ITERATION = 1;
    const long MAX_ABS_TRIANGLES_PER_ITERATION = 100000;
    const long ABS_MAX_SUBDIVISION_STEPS;
    const uint32_t RT_SHADER_INDEX_RAYGEN = 0;
    const uint32_t RT_SHADER_INDEX_MISS = 1;
    const uint32_t RT_SHADER_INDEX_CLOSEST_HIT = 2;

    unsigned int pvsBufferCapacity;

    GPUHashSet *gpuHashSet;
    //int* hashTablePVS;
    //char *inserted;
    //char *device_inserted;
    int hashTableCapacity;
    RasterVisibility *rasterVisibility;

    std::vector<std::vector<float>> haltonPoints;
    glm::vec4 lastHaltonPoints;
    std::random_device rd;
    std::mt19937 gen;
    std::mutex *queueSubmitMutex;
    std::atomic<int> tracedRays;

    VkPhysicalDevice physicalDevice;
    VkDevice logicalDevice;
    std::array<uint8_t, VK_UUID_SIZE> deviceUUID;

    std::vector<VkCommandBuffer> commandBuffer;
    std::vector<VkCommandBuffer> commandBufferABS;
    std::vector<VkCommandBuffer> commandBufferEdgeSubdiv;
    std::vector<VkCommandBuffer> commandBufferCompute;
    std::vector<VkCommandBuffer> commandBufferHaltonCompute;
    std::vector<VkFence> commandBufferFence;
    VkPhysicalDeviceRayTracingPropertiesNV rayTracingProperties;
    VkPipeline pipeline;
    VkPipelineLayout pipelineLayout;
    VkPipeline pipelineABS;
    VkPipelineLayout pipelineABSLayout;
    VkPipeline pipelineEdgeSubdiv;
    VkPipelineLayout pipelineEdgeSubdivLayout;
    VkPipeline pipelineCompute;
    VkPipelineLayout pipelineComputeLayout;
    VkPipeline pipelineHaltonCompute;
    VkPipelineLayout pipelineHaltonComputeLayout;

    VkDescriptorPool descriptorPool;
    std::vector<VkDescriptorSet> descriptorSet;
    VkDescriptorSetLayout descriptorSetLayout;
    std::vector<VkDescriptorSet> descriptorSetABS;
    VkDescriptorSetLayout descriptorSetLayoutABS;
    std::vector<VkDescriptorSet> descriptorSetEdgeSubdiv;
    VkDescriptorSetLayout descriptorSetLayoutEdgeSubdiv;
    std::vector<VkDescriptorSet> descriptorSetCompute;
    VkDescriptorSetLayout descriptorSetLayoutCompute;
    std::vector<VkDescriptorSet> descriptorSetHaltonCompute;
    VkDescriptorSetLayout descriptorSetLayoutHaltonCompute;

    VkImageView storageImageView;
    VkBuffer shaderBindingTable;
    VkDeviceMemory shaderBindingTableMemory;
    VkBuffer shaderBindingTableABS;
    VkDeviceMemory shaderBindingTableMemoryABS;
    VkBuffer shaderBindingTableEdgeSubdiv;
    VkDeviceMemory shaderBindingTableMemoryEdgeSubdiv;
    std::vector<VkBuffer> haltonPointsBuffer;
    std::vector<VkDeviceMemory> haltonPointsBufferMemory;
    std::vector<VkBuffer> viewCellBuffer;
    std::vector<VkDeviceMemory> viewCellBufferMemory;

    std::vector<VkDeviceMemory> randomSamplingOutputBufferMemory;
    std::vector<VkBuffer> absWorkingBuffer;
    std::vector<VkDeviceMemory> absWorkingBufferMemory;
    std::vector<void*> absOutputPointer;
    std::vector<VkBuffer> edgeSubdivOutputBuffer;
    std::vector<VkDeviceMemory> edgeSubdivOutputBufferMemory;
    std::vector<VkBuffer> edgeSubdivOutputHostBuffer;
    std::vector<VkDeviceMemory> edgeSubdivOutputHostBufferMemory;
    std::vector<void*> edgeSubdivOutputPointer;
    std::vector<VkDeviceMemory> triangleCounterBufferMemory;
    std::vector<VkBuffer> randomSamplingOutputHostBuffer;
    std::vector<VkDeviceMemory> randomSamplingOutputHostBufferMemory;
    std::vector<void*> randomSamplingOutputPointer;

    std::vector<VkBuffer> pvsBulkInsertBuffer;
    std::vector<VkDeviceMemory> pvsBulkInsertBufferMemory;

    std::vector<VkDeviceMemory> pvsBufferMemory;
    std::vector<void*> pvsPointer;
    std::vector<VkBuffer> pvsCapacityUniformBuffer;
    std::vector<VkDeviceMemory> pvsCapacityUniformMemory;
    /*
    int *pvsCuda;
    cudaExternalMemory_t pvsCudaMemory = {};
    float *haltonCuda;
    cudaExternalMemory_t haltonCudaMemory = {};

    Sample *randomSamplingOutputCuda;
    cudaExternalMemory_t randomSamplingOutputCudaMemory = {};
    Sample *absOutputCuda;
    cudaExternalMemory_t absOutputCudaMemory = {};
    Sample *edgeSubdivOutputCuda;
    cudaExternalMemory_t edgeSubdivOutputCudaMemory = {};
     */

    AccelerationStructure bottomLevelAS;
    AccelerationStructure topLevelAS;

    PFN_vkCreateAccelerationStructureNV vkCreateAccelerationStructureNV;
    PFN_vkDestroyAccelerationStructureNV vkDestroyAccelerationStructureNV;
    PFN_vkBindAccelerationStructureMemoryNV vkBindAccelerationStructureMemoryNV;
    PFN_vkGetAccelerationStructureHandleNV vkGetAccelerationStructureHandleNV;
    PFN_vkGetAccelerationStructureMemoryRequirementsNV vkGetAccelerationStructureMemoryRequirementsNV;
    PFN_vkCmdBuildAccelerationStructureNV vkCmdBuildAccelerationStructureNV;
    PFN_vkCreateRayTracingPipelinesNV vkCreateRayTracingPipelinesNV;
    PFN_vkGetRayTracingShaderGroupHandlesNV vkGetRayTracingShaderGroupHandlesNV;
    PFN_vkCmdTraceRaysNV vkCmdTraceRaysNV;

    VkDeviceSize bindingOffsetRayGenShader;
    VkDeviceSize bindingOffsetMissShader;
    VkDeviceSize bindingOffsetHitShader;
    VkDeviceSize bindingStride;

    void initRayTracing(
        VkBuffer indexBuffer, VkBuffer vertexBuffer, const std::vector<uint32_t> &indices,
        const std::vector<Vertex> &vertices, const std::vector<VkBuffer> &uniformBuffers
    );
    void createBottomLevelAS(const VkGeometryNV *geometry);
    void createTopLevelAS();
    void buildAS(const VkBuffer instanceBuffer, const VkGeometryNV *geometry);
    void createDescriptorSetLayout();
    void createDescriptorSets(
        VkBuffer indexBuffer, VkBuffer vertexBuffer, const std::vector<VkBuffer> &uniformBuffers,
        int threadId
    );
    void createDescriptorPool();
    void createPipeline(
        const std::array<VkPipelineShaderStageCreateInfo, 3> &shaderStages,
        VkPipelineLayout &pipelineLayout, VkPipeline &pipeline,
        const std::vector<VkDescriptorSetLayout> &descriptorSetLayouts,
        std::vector<VkPushConstantRange> pushConstantRanges
    );
    void createShaderBindingTable(
        VkBuffer &shaderBindingTable, VkDeviceMemory &shaderBindingTableMemory, VkPipeline &pipeline
    );
    VkDeviceSize copyShaderIdentifier(
        uint8_t* data, const uint8_t* shaderHandleStorage, uint32_t groupIndex
    );
    void createCommandBuffers();
    void copyHaltonPointsToBuffer(int threadId);
    void updateViewCellBuffer(int viewCellIndex);
    void createBuffers(const std::vector<uint32_t> &indices);
    void createDescriptorSets();
    void createRandomSamplingPipeline();
    void createABSDescriptorSetLayout();
    void createABSDescriptorSets(VkBuffer vertexBuffer, int threadId);
    void createABSPipeline();
    void createEdgeSubdivPipeline();
    void createEdgeSubdivDescriptorSetLayout();
    void createEdgeSubdivDescriptorSets(int threadId);
    void createComputeDescriptorSets(int threadId);
    void createComputeDescriptorSetLayout();
    void createComputePipeline();
    void createHaltonComputeDescriptorSets(int threadId);
    void createHaltonComputeDescriptorSetLayout();
    void createHaltonComputePipeline();
    ViewCell getViewCellTile(int numThreads, int viewCellIndex, int threadId);
    void resetPVSGPUBuffer();
    void resetAtomicBuffers();
    void resizePVSBuffer(int newSize);
    void generateHaltonSequence(int n, float rand);

    ShaderExecutionInfo randomSample(int numRays, int threadId, int viewCellIndex);
    ShaderExecutionInfo adaptiveBorderSample(const std::vector<Sample> &absWorkingVector, int threadId, int viewCellIndex);
    ShaderExecutionInfo edgeSubdivide(int numSamples, int threadId, int viewCellIndex);
};
