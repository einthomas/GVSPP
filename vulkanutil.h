#ifndef VULKANUTIL_H
#define VULKANUTIL_H

#include <vulkan/vulkan.h>
#include <vulkan/vulkan_core.h>

#include <string>
#include <vector>
//#include <QString>
//#include <QFile>

class VulkanUtil {
public:
    static void createBuffer(
        VkPhysicalDevice physicalDevice, VkDevice logicalDevice, VkDeviceSize size,
        VkBufferUsageFlags usageFlags, VkBuffer &buffer, VkDeviceMemory &bufferMemory,
        VkMemoryPropertyFlags properties
    );
    static void copyBuffer(
        VkDevice logicalDevice, VkCommandPool commandPool, VkQueue queue, VkBuffer srcBuffer,
        VkBuffer dstBuffer, VkDeviceSize size
    );
    static VkCommandBuffer beginSingleTimeCommands(
        VkDevice logicalDevice, VkCommandPool commandPool
    );
    static void endSingleTimeCommands(
        VkDevice logicalDevice, VkCommandBuffer commandBuffer, VkCommandPool commandPool,
        VkQueue queue
    );
    static uint32_t findMemoryType(
        VkPhysicalDevice physicalDevice, uint32_t typeFilter, VkMemoryPropertyFlags properties
    );
    static uint32_t findQueueFamilies(VkPhysicalDevice device, VkQueueFlags queueFlags, int k = -1);
    static VkShaderModule createShader(VkDevice logicalDevice, const std::string &filename);
    static void executeCommandBuffer(
        VkDevice logicalDevice, VkQueue queue, VkCommandBuffer commandBuffer, VkFence fence
    );
    static void createImage(
        VkPhysicalDevice physicalDevice, VkDevice device, uint32_t width, uint32_t height,
        VkFormat format, VkImageTiling tiling, VkImageUsageFlags usage,
        VkMemoryPropertyFlags properties, VkImage& image, VkDeviceMemory& imageMemory
    );
    static VkImageView createImageView(
        VkDevice device, VkImage image, VkFormat format, VkImageAspectFlags aspectFlags
    );

private:
    static std::vector<char> readBinaryFile(const std::string &filename);
};

#endif // VULKANUTIL_H
