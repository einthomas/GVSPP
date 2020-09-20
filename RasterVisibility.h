#ifndef NIRENSTEINSAMPLER_H
#define NIRENSTEINSAMPLER_H

#include <string>
#include <vector>
#include <unordered_map>

#include "Vertex.h"
#include "viewcell.h"
#include "GLFWVulkanWindow.h"
#include "CUDAUtil.h"

class Statistics;

class RasterVisibility {
public:
    Statistics *statistics;

    RasterVisibility(
        VkPhysicalDevice physicalDevice,
        VkDevice device,
        VkCommandPool graphicsCommandPool,
        VkQueue graphicsQueue,
        uint32_t frameBufferWidth,
        uint32_t frameBufferHeight,
        VkFormat depthFormat,
        VkQueue computeQueue,
        VkCommandPool computeCommandPool,
        VkQueue transferQueue,
        VkCommandPool transferCommandPool,
        VkBuffer vertexBuffer,
        const std::vector<Vertex> &vertices,
        VkBuffer indexBuffer,
        const std::vector<uint32_t> indices,
        const int numTriangles,
        VkBuffer sampleOutputBuffer,
        VkBuffer setBuffer,
        VkBuffer triangleCounterBuffer
    );
    int run(const ViewCell &viewCell, glm::vec3 cameraForward, const std::vector<glm::vec2> &haltonPoints);

private:
    const int FRAME_BUFFER_WIDTH;
    const int FRAME_BUFFER_HEIGHT;
    const int MAX_NUM_TRIANGLES;
    const int MULTI_VIEW_LAYER_COUNT = 5;

    VkCommandPool graphicsCommandPool;
    VkCommandPool computeCommandPool;
    VkCommandPool transferCommandPool;
    VkQueue graphicsQueue;
    VkQueue computeQueue;
    VkQueue transferQueue;
    VkPhysicalDevice physicalDevice;
    VkDevice logicalDevice;

    VkRenderPass renderPass;
    VkDescriptorPool descriptorPool;
    VkDescriptorSet nirensteinDescriptorSet;
    VkDescriptorSetLayout nirensteinDescriptorSetLayout;
    VkDescriptorSet computeDescriptorSet;
    VkDescriptorSetLayout computeDescriptorSetLayout;
    VkPipeline nirensteinPipeline;
    VkPipelineLayout nirensteinPipelineLayout;
    VkPipeline computePipeline;
    VkPipelineLayout computePipelineLayout;
    VkFramebuffer framebuffer;
    VkCommandBuffer commandBufferRenderFront;
    VkCommandBuffer commandBufferRenderSides;
    VkCommandBuffer commandBufferRenderTopBottom;
    VkCommandBuffer computeCommandBuffer;
    VkFence fence;

    VkImage depthImage;
    VkDeviceMemory depthImageMemory;
    VkImageView depthImageView;
    VkImage colorImage;     // MSAA render target
    VkDeviceMemory colorImageMemory;
    VkImageView colorImageView;
    VkSampler colorImageSampler;

    VkBuffer pvsBuffer;
    VkDeviceMemory pvsBufferMemory;
    VkBuffer currentPvsBuffer;
    VkDeviceMemory currentPvsBufferMemory;
    int *currentPvsCuda;
    cudaExternalMemory_t currentPvsCudaMemory = {};
    VkBuffer uniformBuffer;
    VkDeviceMemory uniformBufferMemory;
    VkBuffer triangleIDBuffer;
    VkDeviceMemory triangleIDBufferMemory;
    VkBuffer pvsSizeUniformBuffer;
    VkDeviceMemory pvsSizeUniformBufferMemory;
    VkBuffer pvsSizeBuffer;
    VkDeviceMemory pvsSizeBufferMemory;
    VkBuffer currentPvsIndexUniformBuffer;
    VkDeviceMemory currentPvsIndexUniformBufferMemory;

    VkBuffer primitiveCounterBuffer;
    VkDeviceMemory primitiveCounterBufferMemory;

    VkBuffer sampleOutputBuffer;
    VkBuffer setBuffer;
    VkBuffer triangleCounterBuffer;

    VkBuffer cubePosUniformBuffer;
    VkDeviceMemory cubePosUniformBufferMemory;
    VkBuffer numSamplesBuffer;
    VkDeviceMemory numSamplesBufferMemory;

    void createRenderPass(VkFormat depthFormat);
    void createDescriptorPool();
    void createDescriptorSet();
    void createDescriptorSetLayout();
    void createComputeDescriptorSet();
    void createComputeDescriptorSetLayout();
    void createComputePipeline();
    void createPipeline(
        VkPipeline &pipeline, VkPipelineLayout &pipelineLayout, std::string vertShaderPath,
        std::string fragShaderPath, VkPipelineLayoutCreateInfo pipelineLayoutInfo
    );
    void createBuffers(const int numTriangles);
    void createFramebuffer(VkFormat depthFormat);
    void createCommandBuffers(
        VkBuffer vertexBuffer, const std::vector<Vertex> &vertices, VkBuffer indexBuffer,
        const std::vector<uint32_t> indices
    );
    void createCommandBuffer(
        VkCommandBuffer &commandBuffer, VkBuffer vertexBuffer, const std::vector<Vertex> &vertices,
        VkBuffer indexBuffer, const std::vector<uint32_t> indices, VkRect2D scissor
    );
    void createComputeCommandBuffer();
    void releaseRessources();
    void renderVisibilityCube(
        const std::array<glm::vec3, 5> &cameraForwards,
        const std::array<glm::vec3, 5> &cameraUps,
        const glm::vec3 &pos
    );
    void divideHaltonRandom(
        const ViewCell &viewCell, glm::vec3 cameraForward,
        const std::vector<glm::vec2> &haltonPoints
    );
    void resetPVS();
};

#endif // NIRENSTEINSAMPLER_H