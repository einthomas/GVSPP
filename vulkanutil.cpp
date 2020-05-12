#include "vulkanutil.h"
#include <optional>
#include <cstring>
#include <fstream>

void VulkanUtil::createBuffer(
    VkPhysicalDevice physicalDevice, VkDevice logicalDevice, VkDeviceSize size,
    VkBufferUsageFlags usageFlags, VkBuffer &buffer, VkDeviceMemory &bufferMemory,
    VkMemoryPropertyFlags properties
) {
    // Create buffer
    VkBufferCreateInfo bufferInfo = {};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = size;
    bufferInfo.usage = usageFlags;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    if (vkCreateBuffer(logicalDevice, &bufferInfo, nullptr, &buffer) != VK_SUCCESS) {
        //qWarning("failed to create buffer");
    }

    // Allocate memory
    VkMemoryRequirements memoryReq;
    vkGetBufferMemoryRequirements(logicalDevice, buffer, &memoryReq);

    VkMemoryAllocateInfo memoryAllocInfo = {};
    memoryAllocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    memoryAllocInfo.allocationSize = memoryReq.size;
    memoryAllocInfo.memoryTypeIndex = findMemoryType(
        physicalDevice, memoryReq.memoryTypeBits, properties
    );
    if (vkAllocateMemory(
            logicalDevice, &memoryAllocInfo, nullptr, &bufferMemory
        ) != VK_SUCCESS
    ) {
        //qWarning("failed to allocate vertex buffer memory");
    }

    // Assign memory
    vkBindBufferMemory(logicalDevice, buffer, bufferMemory, 0);
}

void VulkanUtil::copyBuffer(
    VkDevice logicalDevice, VkCommandPool commandPool, VkQueue queue, VkBuffer srcBuffer,
    VkBuffer dstBuffer, VkDeviceSize size
) {
    VkCommandBuffer commandBuffer = beginSingleTimeCommands(logicalDevice, commandPool);

    // Copy command
    VkBufferCopy copyRegion = {};
    copyRegion.size = size;
    vkCmdCopyBuffer(commandBuffer, srcBuffer, dstBuffer, 1, &copyRegion);

    endSingleTimeCommands(logicalDevice, commandBuffer, commandPool, queue);
}

VkCommandBuffer VulkanUtil::beginSingleTimeCommands(
    VkDevice logicalDevice, VkCommandPool commandPool
) {
    // Allocate command buffer
    VkCommandBufferAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandPool = commandPool;  // TODO: Create separate command pool for temp command buffers
    allocInfo.commandBufferCount = 1;

    VkCommandBuffer commandBuffer;
    vkAllocateCommandBuffers(logicalDevice, &allocInfo, &commandBuffer);

    // Begin recording commands
    VkCommandBufferBeginInfo beginInfo = {};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(commandBuffer, &beginInfo);

    return commandBuffer;
}

void VulkanUtil::endSingleTimeCommands(
    VkDevice logicalDevice, VkCommandBuffer commandBuffer, VkCommandPool commandPool, VkQueue queue
) {
    vkEndCommandBuffer(commandBuffer);

    // Execute command buffer
    VkSubmitInfo submitInfo = {};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;

    VkFenceCreateInfo fenceInfo;
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fenceInfo.pNext = NULL;
    fenceInfo.flags = 0;
    VkFence fence;
    vkCreateFence(logicalDevice, &fenceInfo, NULL, &fence);

    vkQueueSubmit(queue, 1, &submitInfo, fence);

    VkResult result;
    // Wait for the command buffer to complete execution in a loop in case it takes longer to
    // complete than expected
    do {
        result = vkWaitForFences(logicalDevice, 1, &fence, VK_TRUE, UINT64_MAX);
    } while(result == VK_TIMEOUT);

    vkDestroyFence(logicalDevice, fence, nullptr);
    vkFreeCommandBuffers(logicalDevice, commandPool, 1, &commandBuffer);
}

uint32_t VulkanUtil::findMemoryType(
    VkPhysicalDevice physicalDevice, uint32_t typeFilter, VkMemoryPropertyFlags properties
) {
    VkPhysicalDeviceMemoryProperties memProperties;
    vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);

    for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
        if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
        }
    }
}

uint32_t VulkanUtil::findQueueFamilies(VkPhysicalDevice device, VkQueueFlags queueFlags, int k) {
    uint32_t queueFamilyCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, nullptr);

    std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, queueFamilies.data());

    std::optional<uint32_t> family;
    int i = 0;
    for (const auto& queueFamily : queueFamilies) {
        if (queueFamily.queueFlags & queueFlags) {
            family = i;
        }

        if (family.has_value() && int(family.value()) > k) {
            break;
        }

        i++;
    }

    return family.value();
}

std::vector<char> VulkanUtil::readBinaryFile(const std::string &filename) {
    // Open a binary file with the read position at the end
    std::ifstream file(filename, std::ios::ate | std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("failed to open file " + filename);
    }

    // Get the file size from the current read position
    size_t fileSize = (size_t) file.tellg();

    std::vector<char> buffer(fileSize);
    file.seekg(0);
    file.read(buffer.data(), fileSize);

    file.close();
    return buffer;
}

VkShaderModule VulkanUtil::createShader(VkDevice logicalDevice, const std::string &filename) {
    /*
    QFile file(name);
    if (!file.open(QIODevice::ReadOnly)) {
        qWarning("Failed to read shader %s", qPrintable(name));
        return VK_NULL_HANDLE;
    }
    QByteArray blob = file.readAll();
    file.close();
    */

    auto shaderCode = readBinaryFile(filename);

    VkShaderModuleCreateInfo shaderInfo = {};
    shaderInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    shaderInfo.codeSize = shaderCode.size();
    shaderInfo.pCode = reinterpret_cast<const uint32_t *>(shaderCode.data());

    VkShaderModule shaderModule;
    VkResult err = vkCreateShaderModule(logicalDevice, &shaderInfo, nullptr, &shaderModule);
    if (err != VK_SUCCESS) {
        //qWarning("Failed to create shader module: %d", err);
        return VK_NULL_HANDLE;
    }

    return shaderModule;
}

void VulkanUtil::executeCommandBuffer(
    VkDevice logicalDevice, VkQueue queue, VkCommandBuffer commandBuffer, VkFence fence
) {
    VkSubmitInfo submitInfo = {};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;

    vkQueueSubmit(queue, 1, &submitInfo, fence);
    //vkQueueWaitIdle(window->graphicsQueue());
    VkResult result;
    // Wait for the command buffer to complete execution in a loop in case it takes longer to
    // complete than expected
    do {
        result = vkWaitForFences(logicalDevice, 1, &fence, VK_TRUE, UINT64_MAX);
    } while(result == VK_TIMEOUT);
    // Free the command buffer
    vkResetFences(logicalDevice, 1, &fence);
}
