#ifndef VULKANUTIL_H
#define VULKANUTIL_H

#include <vulkan/vulkan.h>
#include <vulkan/vulkan_core.h>

#include <QString>
#include <QFile>

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
    static VkShaderModule createShader(VkDevice logicalDevice, const QString &name);
    static void executeCommandBuffer(
        VkDevice logicalDevice, VkQueue queue, VkCommandBuffer commandBuffer, VkFence fence
    );
};

#endif // VULKANUTIL_H
