#ifndef RENDERER_H
#define RENDERER_H

#include <QVulkanWindow>
#include <unordered_map>
#include <set>
#include <queue>
#include "vulkanutil.h"
#include "Vertex.h"
#include "visibilitymanager.h"
#include "pvs.h"

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



    //VkShaderModule createShader(const QString &name);
    void createGraphicsPipeline();
    void createVertexBuffer();
    void createIndexBuffer();

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

    uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties);

    // RT


    bool visualizePVS = false;

    // Visibility
    //std::unordered_set<glm::uvec3> pvs;


    const int RAYS_PER_ITERATION_SQRT = 40;
    //const size_t MAX_ABS_RAYS = RAYS_PER_ITERATION_SQRT * RAYS_PER_ITERATION_SQRT;
    //const size_t MIN_ABS_RAYS = size_t(floor(MAX_ABS_RAYS * 0.2));
    //const size_t MIN_ABS_RAYS = 1;

    VisibilityManager visibilityManager;
    VkDescriptorSet rtDescriptorSetsABS;
    VkDescriptorSetLayout rtDescriptorSetLayoutABS;
    void initVisibilityManager();
    void executeCommandBuffer(VkCommandBuffer commandBuffer);


    QueueFamilyIndices findQueueFamilies();
};

#endif // RENDERER_H
