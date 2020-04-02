#ifndef RENDERER_H
#define RENDERER_H

#include <QVulkanWindow>
#include <unordered_map>
#include <set>
#include "Vertex.h"
#include "visibilitymanager.h"
#include "pvs.h"

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

class VulkanRenderer : public QVulkanWindowRenderer {
public:
    VulkanRenderer(QVulkanWindow *w);

    void initResources() override;
    void initSwapChainResources() override;
    void releaseSwapChainResources() override;
    void releaseResources() override;
    void startNextFrame() override;
    void togglePVSVisualzation();
    void saveWindowContentToImage();

private:
    QVulkanWindow *window;

    VkDescriptorPool descriptorPool;
    std::vector<VkDescriptorSet> descriptorSets;
    VkDescriptorSetLayout descriptorSetLayout;
    VkPipeline pipeline;
    VkPipelineLayout pipelineLayout;
    VkShaderModule fragShaderModule;
    VkShaderModule vertShaderModule;

    std::unordered_map<Vertex, uint32_t> uniqueVertices;
    std::vector<Vertex> vertices;
    std::vector<uint32_t> indices;
    VkBuffer vertexBuffer;
    VkDeviceMemory vertexBufferMemory;
    VkBuffer indexBuffer;
    VkDeviceMemory indexBufferMemory;

    std::vector<VkBuffer> uniformBuffers;
    std::vector<VkDeviceMemory> uniformBuffersMemory;

    VkImage textureImage;
    VkDeviceMemory textureImageMemory;
    VkImageView textureImageView;
    VkSampler textureSampler;

    PFN_vkCreateAccelerationStructureNV vkCreateAccelerationStructureNV;
    PFN_vkDestroyAccelerationStructureNV vkDestroyAccelerationStructureNV;
    PFN_vkBindAccelerationStructureMemoryNV vkBindAccelerationStructureMemoryNV;
    PFN_vkGetAccelerationStructureHandleNV vkGetAccelerationStructureHandleNV;
    PFN_vkGetAccelerationStructureMemoryRequirementsNV vkGetAccelerationStructureMemoryRequirementsNV;
    PFN_vkCmdBuildAccelerationStructureNV vkCmdBuildAccelerationStructureNV;
    PFN_vkCreateRayTracingPipelinesNV vkCreateRayTracingPipelinesNV;
    PFN_vkGetRayTracingShaderGroupHandlesNV vkGetRayTracingShaderGroupHandlesNV;
    PFN_vkCmdTraceRaysNV vkCmdTraceRaysNV;
    AccelerationStructure bottomLevelAS;
    AccelerationStructure topLevelAS;

    VkShaderModule createShader(const QString &name);
    void createGraphicsPipeline();
    void createVertexBuffer();
    void createIndexBuffer();
    void createBuffer(
        VkDeviceSize size, VkBufferUsageFlags usageFlags, VkBuffer &buffer, VkDeviceMemory &bufferMemory,
        VkMemoryPropertyFlags properties
    );
    void copyBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size);

    void loadModel();
    void createTextureImage();
    void createImage(
        uint32_t width, uint32_t height, VkFormat format, VkImageTiling tiling, VkImageUsageFlags usage,
        uint32_t memoryTypeIndex, VkImage& image, VkDeviceMemory& imageMemory
    );
    void createTextureImageView();
    void createTextureSampler();
    void transitionImageLayout(
        VkCommandBuffer commandBuffer, VkImage image, VkFormat format, VkImageLayout oldLayout,
        VkImageLayout newLayout,
        VkPipelineStageFlags sourceStage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
        VkPipelineStageFlags destinationStage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT
    );
    void copyBufferToImage(VkBuffer buffer, VkImage image, uint32_t width, uint32_t height);

    void createDescriptorSetLayout();
    void createDescriptorPool();
    void createDescriptorSets();
    void createUniformBuffers();
    void updateUniformBuffer(uint32_t swapChainImageIndex);

    VkCommandBuffer beginSingleTimeCommands();
    void endSingleTimeCommands(VkCommandBuffer commandBuffer);

    uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties);

    // RT
    const uint32_t RT_SHADER_INDEX_RAYGEN = 0;
    const uint32_t RT_SHADER_INDEX_MISS = 1;
    const uint32_t RT_SHADER_INDEX_CLOSEST_HIT = 2;
    const uint32_t RT_SHADER_INDEX_RAYGEN_ABS = 3;
    VkCommandBuffer rtCommandBuffer;
    VkPhysicalDeviceRayTracingPropertiesNV rayTracingProperties;
    VkPipeline rtPipeline;
    VkPipelineLayout rtPipelineLayout;
    VkPipeline rtPipelineABS;
    VkPipelineLayout rtPipelineABSLayout;
    VkDescriptorPool rtDescriptorPool;
    VkDescriptorSet rtDescriptorSets;
    VkDescriptorSetLayout rtDescriptorSetLayout;
    VkImage rtStorageImage;
    VkDeviceMemory rtStorageImageMemory;
    VkImageView rtStorageImageView;
    VkBuffer shaderBindingTable;
    VkDeviceMemory shaderBindingTableMemory;
    VkBuffer shaderBindingTableABS;
    VkDeviceMemory shaderBindingTableMemoryABS;
    void initRayTracing();
    void createBottomLevelAS(const VkGeometryNV *geometry);
    void createTopLevelAS();
    void buildAS(const VkBuffer instanceBuffer, const VkGeometryNV *geometry);
    void createRtDescriptorSetLayout();
    void createRtDescriptorSets();
    void createRtDescriptorPool();
    void createRtPipeline(
        std::array<VkPipelineShaderStageCreateInfo, 3> shaderStages,
        VkPipelineLayout *pipelineLayout, VkPipeline *pipeline,
        std::vector<VkDescriptorSetLayout> descriptorSetLayouts
    );
    void createShaderBindingTable(
        VkBuffer &shaderBindingTable, VkDeviceMemory &shaderBindingTableMemory, VkPipeline &pipeline
    );
    VkDeviceSize copyShaderIdentifier(uint8_t* data, const uint8_t* shaderHandleStorage, uint32_t groupIndex);
    void rayTrace();
    void createCommandBuffers();

    bool visualizePVS = false;

    // Visibility
    std::unordered_set<glm::uvec3> pvs;
    const int RAYS_PER_ITERATION_SQRT = 500;
    VkBuffer haltonPointsBuffer;
    VkDeviceMemory haltonPointsBufferMemory;
    VkBuffer viewCellBuffer;
    VkDeviceMemory viewCellBufferMemory;
    VkBuffer intersectedTrianglesBuffer;
    VkDeviceMemory intersectedTrianglesBufferMemory;
    VkBuffer absOutputBuffer;
    VkDeviceMemory absOutputBufferMemory;
    VkBuffer pvsVisualizationBuffer;
    VkDeviceMemory pvsVisualizationBufferMemory;
    VisibilityManager visibilityManager;
    VkDescriptorSet rtDescriptorSetsABS;
    VkDescriptorSetLayout rtDescriptorSetLayoutABS;
    void initVisibilityManager();
    void createHaltonPointsBuffer();
    void createViewCellBuffer();
    void createPVSBuffer();
    void executeCommandBuffer(VkCommandBuffer commandBuffer);
    void createRandomSamplingRtPipeline();
    void createABSRtDescriptorSetLayout();
    void createABSRtDescriptorSets();
    void createABSRtPipeline();
};

#endif // RENDERER_H
