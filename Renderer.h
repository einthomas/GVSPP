#ifndef RENDERER_H
#define RENDERER_H

//#include "qvulkanwindow.h"
//#include <QVulkanWindow>
#include <unordered_map>
#include <set>
#include <queue>
#include "GLFWVulkanWindow.h"
#include "vulkanutil.h"
#include "Vertex.h"
#include "visibilitymanager.h"

class VulkanRenderer { // : public QVulkanWindowRenderer {
public:
    glm::vec3 cameraPos;
    glm::vec3 cameraForward;
    glm::vec3 cameraRight;
    glm::vec3 cameraUp;

    VulkanRenderer(GLFWVulkanWindow *w);

    void initResources();
    void initSwapChainResources();
    void releaseSwapChainResources();
    void releaseResources();
    void startNextFrame(
        uint32_t swapChainImageIndex, VkFramebuffer framebuffer, VkCommandBuffer commandBuffer
    );
    void togglePVSVisualization();
    void toggleWholeModelVisualization();
    void nextCorner();
    void alignCameraWithViewCellNormal();
    void startVisibilityThread();

private:    
    //QVulkanWindow *window;
    GLFWVulkanWindow *window;

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
    int currentViewCellCornerView = 0;

    void createGraphicsPipeline();
    void createVertexBuffer();
    void createIndexBuffer();

    void loadModel();
    void createTextureImage();
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

    // Visibility
    bool visualizePVS = false;
    bool renderWholeModel = false;
    //std::thread visibilityThread;
    std::vector<std::thread> visibilityThreads;

    VisibilityManager visibilityManager;
    VkDescriptorSet rtDescriptorSetsABS;
    VkDescriptorSetLayout rtDescriptorSetLayoutABS;
    void initVisibilityManager();
};

#endif // RENDERER_H
