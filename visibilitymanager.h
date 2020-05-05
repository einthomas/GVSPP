#ifndef VISIBILITYMANAGER_H
#define VISIBILITYMANAGER_H

#include <vulkan/vulkan.h>
#include <vulkan/vulkan_core.h>
#include <vector>
#include <array>
#include <unordered_set>

#include "viewcell.h"
#include "Vertex.h"
#include "sample.h"

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

struct QueueFamilyIndices {
    std::optional<uint32_t> graphicsFamily;

    bool isComplete() {
        return graphicsFamily.has_value();
    }
};

class VisibilityManager {
public:
    VisibilityManager(int raysPerIteration);
    void init(
        int raysPerIteration, VkPhysicalDevice physicalDevice, VkDevice logicalDevice,
        VkCommandPool graphicsCommandPool, VkQueue graphicsQueue, VkBuffer indexBuffer,
        const std::vector<uint32_t> &indices, VkBuffer vertexBuffer,
        const std::vector<Vertex> &vertices, const std::vector<VkBuffer> &uniformBuffers,
        uint32_t deviceLocalMemoryIndex
    );
    void addViewCell(glm::vec3 pos, glm::vec2 size, glm::vec3 normal);
    void generateHaltonPoints(int n, int p2 = 7);
    void rayTrace(const std::vector<uint32_t> &indices);
    void releaseResources();

    VkBuffer pvsVisualizationBuffer;

private:
    int raysPerIteration;
    const size_t MAX_ABS_TRIANGLES_PER_ITERATION;
    const size_t MAX_EDGE_SUBDIV_RAYS;
    const size_t MAX_SUBDIVISION_STEPS = 3;     // TODO: Shouldn't have to be set separately in raytrace-subdiv.rgen
    //const size_t MIN_ABS_RAYS = size_t(floor(MAX_ABS_RAYS * 0.2));
    const size_t MIN_ABS_TRIANGLES_PER_ITERATION = 1;
    const uint32_t RT_SHADER_INDEX_RAYGEN = 0;
    const uint32_t RT_SHADER_INDEX_MISS = 1;
    const uint32_t RT_SHADER_INDEX_CLOSEST_HIT = 2;

    std::vector<glm::vec2> haltonPoints;
    std::vector<ViewCell> viewCells;
    std::unordered_set<int> pvs;

    VkPhysicalDevice physicalDevice;
    VkDevice logicalDevice;
    VkCommandPool graphicsCommandPool;
    VkQueue graphicsQueue;

    VkCommandPool commandPool;
    VkCommandBuffer commandBuffer;
    VkCommandBuffer commandBufferABS;
    VkCommandBuffer commandBufferEdgeSubdiv;
    VkFence commandBufferFence;
    VkPhysicalDeviceRayTracingPropertiesNV rayTracingProperties;
    VkPipeline pipeline;
    VkPipelineLayout pipelineLayout;
    VkPipeline pipelineABS;
    VkPipelineLayout pipelineABSLayout;
    VkPipeline pipelineEdgeSubdiv;
    VkPipelineLayout pipelineEdgeSubdivLayout;

    VkDescriptorPool descriptorPool;
    VkDescriptorSet descriptorSet;
    VkDescriptorSetLayout descriptorSetLayout;
    VkDescriptorSet descriptorSetABS;
    VkDescriptorSetLayout descriptorSetLayoutABS;
    VkDescriptorSet descriptorSetEdgeSubdiv;
    VkDescriptorSetLayout descriptorSetLayoutEdgeSubdiv;

    VkImageView storageImageView;
    VkBuffer shaderBindingTable;
    VkDeviceMemory shaderBindingTableMemory;
    VkBuffer shaderBindingTableABS;
    VkDeviceMemory shaderBindingTableMemoryABS;
    VkBuffer shaderBindingTableEdgeSubdiv;
    VkDeviceMemory shaderBindingTableMemoryEdgeSubdiv;
    VkBuffer haltonPointsBuffer;
    VkDeviceMemory haltonPointsBufferMemory;
    VkBuffer viewCellBuffer;
    VkDeviceMemory viewCellBufferMemory;
    VkBuffer intersectedTrianglesBuffer;
    VkDeviceMemory intersectedTrianglesBufferMemory;
    VkBuffer rayOriginBuffer;
    VkDeviceMemory rayOriginBufferMemory;
    VkBuffer absOutputBuffer;
    VkDeviceMemory absOutputBufferMemory;
    VkBuffer absWorkingBuffer;
    VkDeviceMemory absWorkingBufferMemory;
    VkDeviceMemory pvsVisualizationBufferMemory;
    VkBuffer edgeSubdivWorkingBuffer;
    VkDeviceMemory edgeSubdivWorkingBufferMemory;

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
        const std::vector<Vertex> &vertices, const std::vector<VkBuffer> &uniformBuffers,
        uint32_t deviceLocalMemoryIndex
    );
    void createBottomLevelAS(const VkGeometryNV *geometry, uint32_t deviceLocalMemoryIndex);
    void createTopLevelAS(uint32_t deviceLocalMemoryIndex);
    void buildAS(const VkBuffer instanceBuffer, const VkGeometryNV *geometry);
    void createDescriptorSetLayout();
    void createDescriptorSets(VkBuffer indexBuffer, VkBuffer vertexBuffer, const std::vector<VkBuffer> &uniformBuffers);
    void createDescriptorPool();
    void createPipeline(
        std::array<VkPipelineShaderStageCreateInfo, 3> shaderStages,
        VkPipelineLayout *pipelineLayout, VkPipeline *pipeline,
        std::vector<VkDescriptorSetLayout> descriptorSetLayouts
    );
    void createShaderBindingTable(
        VkBuffer &shaderBindingTable, VkDeviceMemory &shaderBindingTableMemory, VkPipeline &pipeline
    );
    VkDeviceSize copyShaderIdentifier(uint8_t* data, const uint8_t* shaderHandleStorage, uint32_t groupIndex);
    void createCommandBuffers();
    QueueFamilyIndices findQueueFamilies();
    void createHaltonPointsBuffer();
    void createViewCellBuffer();
    void createBuffers(const std::vector<uint32_t> &indices);
    void createDescriptorSets();
    void createRandomSamplingPipeline();
    void createABSDescriptorSetLayout();
    void createABSDescriptorSets(VkBuffer vertexBuffer);
    void createABSPipeline();
    void createEdgeSubdivPipeline();
    void createEdgeSubdivDescriptorSetLayout();
    void createEdgeSubdivDescriptorSets();

    std::vector<Sample> randomSample(int numRays);
    std::vector<Sample> adaptiveBorderSample(const std::vector<Sample> &absWorkingVector);
    std::vector<Sample> edgeSubdivide(const std::vector<Sample> &intersectedTriangles);
};

#endif // VISIBILITYMANAGER_H
