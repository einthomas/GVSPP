#include "NirensteinSampler.h"
#include "vulkanutil.h"
#include "Vertex.h"
#include "viewcell.h"
#include "CUDAUtil.h"

#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/string_cast.hpp>
#include <glm/gtc/type_ptr.hpp>
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/hash.hpp>
#include <vector>
#include <unordered_set>

struct UniformBufferObjectMultiView {
    glm::mat4 model;
    glm::mat4 view[5];
    glm::mat4 projection;
};
struct UniformBufferObject {
    glm::mat4 model;
    glm::mat4 view;
    glm::mat4 projection;
};

NirensteinSampler::NirensteinSampler(
    GLFWVulkanWindow *window,
    VkQueue computeQueue,
    VkCommandPool computeCommandPool,
    VkQueue transferQueue,
    VkCommandPool transferCommandPool,
    VkBuffer vertexBuffer,
    const std::vector<Vertex> &vertices,
    VkBuffer indexBuffer,
    const std::vector<uint32_t> indices,
    const int numTriangles,
    const float ERROR_THRESHOLD,
    const int MAX_SUBDIVISIONS,
    const bool USE_MULTI_VIEW_RENDERING,
    const bool USE_ADAPTIVE_DIVIDE,
    VkBuffer sampleOutputBuffer,
    VkBuffer setBuffer,
    VkBuffer triangleCounterBuffer
) :
    computeQueue(computeQueue),
    computeCommandPool(computeCommandPool),
    transferQueue(transferQueue),
    transferCommandPool(transferCommandPool),
    MAX_NUM_TRIANGLES(numTriangles),
    physicalDevice(window->physicalDevice), logicalDevice(window->device),
    graphicsCommandPool(window->graphicsCommandPool), graphicsQueue(window->graphicsQueue),
    FRAME_BUFFER_WIDTH(window->swapChainImageSize.width),
    FRAME_BUFFER_HEIGHT(window->swapChainImageSize.height),
    ERROR_THRESHOLD(ERROR_THRESHOLD),
    MAX_SUBDIVISIONS(MAX_SUBDIVISIONS),
    USE_MULTI_VIEW_RENDERING(USE_MULTI_VIEW_RENDERING),
    USE_ADAPTIVE_DIVIDE(USE_ADAPTIVE_DIVIDE),
    sampleOutputBuffer(sampleOutputBuffer),
    setBuffer(setBuffer),
    triangleCounterBuffer(triangleCounterBuffer)
{
    const VkFormat depthFormat = window->findDepthFormat();
    createRenderPass(depthFormat);

    createDescriptorPool();

    createDescriptorSetLayout();
    createComputeDescriptorSetLayout();

    std::array<VkDescriptorSetLayout, 2> layout = { nirensteinDescriptorSetLayout, computeDescriptorSetLayout };
    VkDescriptorSetAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = descriptorPool;
    allocInfo.descriptorSetCount = static_cast<uint32_t>(layout.size());
    allocInfo.pSetLayouts = layout.data();

    std::array<VkDescriptorSet, 2> descriptorSets;
    if (vkAllocateDescriptorSets(
            logicalDevice, &allocInfo, descriptorSets.data()
        ) != VK_SUCCESS
    ) {
        throw std::runtime_error("failed to allocate nirenstein descriptor set");
    }
    nirensteinDescriptorSet = descriptorSets[0];
    computeDescriptorSet = descriptorSets[1];

    createBuffers(numTriangles);
    createDescriptorSet();

    VkPipelineLayoutCreateInfo pipelineLayoutInfo = {};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = 1;
    pipelineLayoutInfo.pSetLayouts = &nirensteinDescriptorSetLayout;

    createPipeline(
        nirensteinPipeline, nirensteinPipelineLayout, "shaders/nirenstein.vert.spv",
        "shaders/nirenstein.frag.spv", pipelineLayoutInfo
    );

    createFramebuffer(depthFormat);
    createCommandBuffers(vertexBuffer, vertices, indexBuffer, indices);

    createComputeDescriptorSet();
    createComputePipeline();
    createComputeCommandBuffer();

    VkFenceCreateInfo fenceInfo;
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fenceInfo.pNext = NULL;
    fenceInfo.flags = 0;
    vkCreateFence(logicalDevice, &fenceInfo, NULL, &fence);

    srand (time(NULL));
}

void NirensteinSampler::renderVisibilityCube(
    const std::array<glm::vec3, 5> &cameraForwards,
    const std::array<glm::vec3, 5> &cameraUps,
    const glm::vec3 &pos
) {
    VkSubmitInfo computeCommandBufferSubmitInfo = {};
    computeCommandBufferSubmitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    computeCommandBufferSubmitInfo.commandBufferCount = 1;
    computeCommandBufferSubmitInfo.pCommandBuffers = &computeCommandBuffer;

    // Update uniform buffer
    if (USE_MULTI_VIEW_RENDERING) {
        UniformBufferObjectMultiView ubo;
        ubo.model = glm::mat4(1.0f);
        ubo.projection = glm::perspective(
            glm::radians(90.0f),
            FRAME_BUFFER_WIDTH / (float) FRAME_BUFFER_HEIGHT,
            0.1f,
            100000.0f
        );
        ubo.projection[1][1] *= -1; // Flip y axis
        for (int k = 0; k < cameraForwards.size(); k++) {
            ubo.view[k] = glm::lookAt(pos, pos + cameraForwards[k], cameraUps[k]);
        }
        void *data;
        vkMapMemory(logicalDevice, uniformBufferMemory, 0, sizeof(ubo), 0, &data);
        memcpy(data, &ubo, sizeof(UniformBufferObjectMultiView));
        vkUnmapMemory(logicalDevice, uniformBufferMemory);

        // Submit command buffer
        VkSubmitInfo renderCommandBufferSubmitInfo = {};
        renderCommandBufferSubmitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        renderCommandBufferSubmitInfo.commandBufferCount = 1;
        renderCommandBufferSubmitInfo.pCommandBuffers = &commandBufferRenderFront;

        auto startTotal = std::chrono::steady_clock::now();
        vkQueueSubmit(graphicsQueue, 1, &renderCommandBufferSubmitInfo, fence);
        VkResult result;
        do {
            result = vkWaitForFences(logicalDevice, 1, &fence, VK_TRUE, UINT64_MAX);
        } while(result == VK_TIMEOUT);
        vkResetFences(logicalDevice, 1, &fence);
        renderTime += std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - startTotal).count();

        // Dispatch compute shader to collect triangle IDs
        startTotal = std::chrono::steady_clock::now();
        {
            vkQueueSubmit(computeQueue, 1, &computeCommandBufferSubmitInfo, fence);
            VkResult result;
            do {
                result = vkWaitForFences(logicalDevice, 1, &fence, VK_TRUE, UINT64_MAX);
            } while(result == VK_TIMEOUT);
            vkResetFences(logicalDevice, 1, &fence);
        }
        computeShaderTime += std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - startTotal).count();
    } else {
        for (int k = 0; k < cameraForwards.size(); k++) {
            // Update uniform buffer
            {
                UniformBufferObject ubo;
                ubo.model = glm::mat4(1.0f);
                ubo.view = glm::lookAt(pos, pos + cameraForwards[k], cameraUps[k]);
                ubo.projection = glm::perspective(
                    glm::radians(90.0f),
                    FRAME_BUFFER_WIDTH / (float) FRAME_BUFFER_HEIGHT,
                    0.1f,
                    100000.0f
                );
                ubo.projection[1][1] *= -1; // Flip y axis

                // Copy data in the uniform buffer object to the uniform buffer
                void *data;
                vkMapMemory(
                    logicalDevice, uniformBufferMemory, 0, sizeof(ubo), 0, &data
                );
                memcpy(data, &ubo, sizeof(UniformBufferObject));
                vkUnmapMemory(logicalDevice, uniformBufferMemory);
            }

            // Submit command buffer
            auto startTotal = std::chrono::steady_clock::now();
            VkSubmitInfo renderCommandBufferSubmitInfo = {};
            renderCommandBufferSubmitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
            renderCommandBufferSubmitInfo.commandBufferCount = 1;
            if (k == 0) {
                renderCommandBufferSubmitInfo.pCommandBuffers = &commandBufferRenderFront;
            } else if (k == 1 || k == 2) {
                renderCommandBufferSubmitInfo.pCommandBuffers = &commandBufferRenderSides;
            } else {
                renderCommandBufferSubmitInfo.pCommandBuffers = &commandBufferRenderTopBottom;
            }
            vkQueueSubmit(graphicsQueue, 1, &renderCommandBufferSubmitInfo, fence);
            VkResult result;
            do {
                result = vkWaitForFences(logicalDevice, 1, &fence, VK_TRUE, UINT64_MAX);
            } while(result == VK_TIMEOUT);
            vkResetFences(logicalDevice, 1, &fence);
            renderTime += std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - startTotal).count();

            // Dispatch compute shader to collect triangle IDs
            startTotal = std::chrono::steady_clock::now();
            {
                vkQueueSubmit(computeQueue, 1, &computeCommandBufferSubmitInfo, fence);
                VkResult result;
                do {
                    result = vkWaitForFences(logicalDevice, 1, &fence, VK_TRUE, UINT64_MAX);
                } while(result == VK_TIMEOUT);
                vkResetFences(logicalDevice, 1, &fence);
            }
            computeShaderTime += std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - startTotal).count();
        }
    }

    /*
    for (int k = 0; k < cameraForwards.size(); k++) {
        // Update uniform buffer
        {
            UniformBufferObject ubo;
            ubo.model = glm::mat4(1.0f);
            ubo.view = glm::lookAt(pos, pos + cameraForwards[k], cameraUps[k]);
            ubo.projection = glm::perspective(
                glm::radians(90.0f),
                FRAME_BUFFER_WIDTH / (float) FRAME_BUFFER_HEIGHT,
                100000.0f,
                0.1f
            );
            ubo.projection[1][1] *= -1; // Flip y axis

            // Copy data in the uniform buffer object to the uniform buffer
            void *data;
            vkMapMemory(
                logicalDevice, uniformBufferMemory, 0, sizeof(ubo), 0, &data
            );
            memcpy(data, &ubo, sizeof(UniformBufferObject));
            vkUnmapMemory(logicalDevice, uniformBufferMemory);
        }

        // Submit command buffer
        auto startTotal = std::chrono::steady_clock::now();
        VkSubmitInfo renderCommandBufferSubmitInfo = {};
        renderCommandBufferSubmitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        renderCommandBufferSubmitInfo.commandBufferCount = 1;
        if (k == 0) {
            renderCommandBufferSubmitInfo.pCommandBuffers = &commandBufferRenderFront;
        } else if (k == 1 || k == 2) {
            renderCommandBufferSubmitInfo.pCommandBuffers = &commandBufferRenderSides;
        } else {
            renderCommandBufferSubmitInfo.pCommandBuffers = &commandBufferRenderTopBottom;
        }
        vkQueueSubmit(graphicsQueue, 1, &renderCommandBufferSubmitInfo, fence);
        VkResult result;
        do {
            result = vkWaitForFences(logicalDevice, 1, &fence, VK_TRUE, UINT64_MAX);
        } while(result == VK_TIMEOUT);
        vkResetFences(logicalDevice, 1, &fence);
        renderTime += std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - startTotal).count();

        //std::cout << "render time " << std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - startTotal).count() << std::endl;

        startTotal = std::chrono::steady_clock::now();
        // Copy frame buffer to storage buffer
        {
            VkCommandBuffer commandBuffer = VulkanUtil::beginSingleTimeCommands(logicalDevice, computeCommandPool);

            VkBufferImageCopy imageCopyRegion = {};
            imageCopyRegion.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            imageCopyRegion.imageSubresource.layerCount = 1;
            imageCopyRegion.imageExtent.width = FRAME_BUFFER_WIDTH;
            imageCopyRegion.imageExtent.height = FRAME_BUFFER_HEIGHT;
            imageCopyRegion.imageExtent.depth = 1;
            imageCopyRegion.bufferRowLength = FRAME_BUFFER_WIDTH;
            imageCopyRegion.bufferImageHeight = 0;
            vkCmdCopyImageToBuffer(
                commandBuffer,
                colorImage,
                VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                triangleIDBuffer,
                1,
                &imageCopyRegion
            );

            VulkanUtil::endSingleTimeCommands(logicalDevice, commandBuffer, computeCommandPool, computeQueue);
        }
        copyTime += std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - startTotal).count();
        //std::cout << "copy time " << std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - startTotal).count() << std::endl;

        // Dispatch compute shader to collect triangle IDs
        startTotal = std::chrono::steady_clock::now();
        {
            vkQueueSubmit(computeQueue, 1, &computeCommandBufferSubmitInfo, fence);
            VkResult result;
            do {
                result = vkWaitForFences(logicalDevice, 1, &fence, VK_TRUE, UINT64_MAX);
            } while(result == VK_TIMEOUT);
            vkResetFences(logicalDevice, 1, &fence);
        }
        computeShaderTime += std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - startTotal).count();
        //std::cout << "render time 2 " << std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - startTotal).count() << std::endl;
    }
    */
}

std::vector<int> NirensteinSampler::run(const ViewCell &viewCell, glm::vec3 viewCellSize, glm::vec3 cameraForward, const std::vector<glm::vec2> &haltonPoints) {
    pvsCache.clear();
    renderTime = 0;
    computeShaderTime = 0;
    copyTime = 0;
    cudaTime = 0;
    fillCacheTime = 0;
    numSubdivisions = 0;
    resetPVS();

    {
        VkDeviceSize bufferSize = sizeof(int);

        // Create staging buffer using host-visible memory
        VkBuffer stagingBuffer;
        VkDeviceMemory stagingBufferMemory;
        VulkanUtil::createBuffer(
            physicalDevice, logicalDevice, bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            stagingBuffer, stagingBufferMemory,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT
        );

        // Copy triangles data to the staging buffer
        unsigned int numTriangles[1] = { 0 };
        void *data;
        vkMapMemory(logicalDevice, stagingBufferMemory, 0, bufferSize, 0, &data);    // Map buffer memory into CPU accessible memory
        memcpy(data, &numTriangles, (size_t) bufferSize);  // Copy vertex data to mapped memory
        vkUnmapMemory(logicalDevice, stagingBufferMemory);

        // Copy triangles data from the staging buffer to GPU-visible absWorkingBuffer
        VulkanUtil::copyBuffer(
            logicalDevice, transferCommandPool, transferQueue, stagingBuffer,
            numSamplesBuffer, bufferSize
        );

        vkDestroyBuffer(logicalDevice, stagingBuffer, nullptr);
        vkFreeMemory(logicalDevice, stagingBufferMemory, nullptr);
    }

    auto startTotal = std::chrono::steady_clock::now();

    std::array<glm::vec3, 4> cornerPositions;
    for (int i = 0; i < 4; i++) {
        glm::vec3 offset;
        offset.x = i % 2 == 0 ? -1.0f : 1.0f;
        offset.y = int(i / 2) % 2 == 0 ? -1.0f : 1.0f;
        offset.z = 0.0f;
        cornerPositions[i] = viewCell.model * glm::vec4(offset, 1.0f);
    }

    if (USE_ADAPTIVE_DIVIDE) {
        divideAdaptive(viewCell, viewCellSize, cameraForward, cornerPositions);
    } else {
        divideHaltonRandom(viewCell, cameraForward, haltonPoints);
    }

    {
        VkDeviceSize bufferSize = sizeof(int);

        VkBuffer hostBuffer;
        VkDeviceMemory hostBufferMemory;
        VulkanUtil::createBuffer(
            physicalDevice,
            logicalDevice, bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT, hostBuffer, hostBufferMemory,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT
        );

        VulkanUtil::copyBuffer(
            logicalDevice, transferCommandPool, transferQueue, numSamplesBuffer,
            hostBuffer, bufferSize
        );

        void *data;
        vkMapMemory(logicalDevice, hostBufferMemory, 0, bufferSize, 0, &data);
        int *n = (int*) data;
        numSamples = n[0];

        vkUnmapMemory(logicalDevice, hostBufferMemory);
        vkDestroyBuffer(logicalDevice, hostBuffer, nullptr);
        vkFreeMemory(logicalDevice, hostBufferMemory, nullptr);
    }

    // Copy current pvs to host
    std::vector<int> pvs;
    {
        VkDeviceSize bufferSize = sizeof(int) * MAX_NUM_TRIANGLES;

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

        int* pvsArray = (int*)data;
        for (int k = 0; k < MAX_NUM_TRIANGLES; k++) {
            if (pvsArray[k] >= 0) {
                pvs.push_back(pvsArray[k]);
            }
        }
    }

    float avgRenderTime = renderTime / 1000000.0f / pvsCache.size();
    avgRenderTimes.push_back(avgRenderTime);
    avgTotalRenderTimes.push_back(renderTime / 1000000.0f);
    pvsSizes.push_back(pvs.size());

    std::cout << "Total render time: " << renderTime / 1000000.0f << "ms" << std::endl;
    std::cout << "Avg render time per hemicube: " << avgRenderTime << "ms" << std::endl;
    std::cout << "# cubes rendered: " << pvsCache.size() << std::endl;
    std::cout << "PVS size: " << pvs.size() << "/" << MAX_NUM_TRIANGLES << "(" << pvs.size() / float(MAX_NUM_TRIANGLES) * 100.0f << "%)" << std::endl;

    std::cout << "Compute shader time: " << computeShaderTime / 1000000.0f << "ms" << std::endl;
    std::cout << "CUDA time: " << cudaTime / 1000000.0f << "ms" << std::endl;
    std::cout << "Fill cache time: " << fillCacheTime / 1000000.0f << "ms" << std::endl;
    //std::cout << "Avg render time per cube: " << renderTime / 1000000.0f / haltonPoints.size() << "ms" << std::endl;
    std::cout << "Total time: " << std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - startTotal).count() / 1000000.0f << "ms " << std::endl;
    std::cout << std::endl;

    return pvs;
}

void NirensteinSampler::divideAdaptive(
    const ViewCell &viewCell, glm::vec3 viewCellSize, glm::vec3 cameraForward, std::array<glm::vec3, 4> positions
) {
    for (int i = 0; i < positions.size(); i++) {
    }

    if (glm::length(positions[0] - positions[1]) / viewCellSize.x <= 0.01f ||
        glm::length(positions[0] - positions[2]) / viewCellSize.y <= 0.01f
    ) {
        return;
    }

    for (auto a : positions) {
        renderCubePositions.push_back(a);
    }

    const glm::vec3 cameraRight = glm::normalize(glm::cross(cameraForward, glm::vec3(0.0f, 1.0f, 0.0f)));
    const glm::vec3 cameraUp = glm::normalize(glm::cross(cameraForward, cameraRight));

    std::array<glm::vec3, 5> cameraForwards = {
        cameraForward,
        cameraRight,
        -cameraRight,
        cameraUp,
        -cameraUp
    };
    std::array<glm::vec3, 5> cameraUps = {
        cameraUp,
        -cameraUp,
        cameraUp,
        cameraForward,
        cameraForward
    };

    // Fill PVS buffer with -1
    {
        VkDeviceSize pvsSize = sizeof(int) * MAX_NUM_TRIANGLES * 5;

        VkDeviceSize bufferSize = pvsSize;

        // Create staging buffer using host-visible memory
        VkBuffer stagingBuffer;
        VkDeviceMemory stagingBufferMemory;
        VulkanUtil::createBuffer(
            physicalDevice,
            logicalDevice, bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            stagingBuffer, stagingBufferMemory,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT
        );

        std::vector<int> vec(MAX_NUM_TRIANGLES * 5);
        std::fill(vec.begin(), vec.end(), -1);
        void *data;
        vkMapMemory(logicalDevice, stagingBufferMemory, 0, bufferSize, 0, &data);    // Map buffer memory into CPU accessible memory
        memcpy(data, vec.data(), (size_t) bufferSize);  // Copy vertex data to mapped memory
        vkUnmapMemory(logicalDevice, stagingBufferMemory);

        VulkanUtil::copyBuffer(
            logicalDevice, transferCommandPool, transferQueue, stagingBuffer,
            currentPvsBuffer, bufferSize
        );

        vkDestroyBuffer(logicalDevice, stagingBuffer, nullptr);
        vkFreeMemory(logicalDevice, stagingBufferMemory, nullptr);
    }

    std::vector<size_t> hashes;
    hashes.reserve(positions.size());
    for (int i = 0; i < positions.size(); i++) {
        auto hash = std::hash<glm::vec3>()(positions[i]);
        if (pvsCache.find(hash) == pvsCache.end()) {
            //std::cout << "not found" << std::endl;
            hashes.push_back(0);

            // Copy data in the uniform buffer object to the uniform buffer
            void *data;
            vkMapMemory(logicalDevice, currentPvsIndexUniformBufferMemory, 0, sizeof(int), 0, &data);
            memcpy(data, &i, sizeof(int));
            vkUnmapMemory(logicalDevice, currentPvsIndexUniformBufferMemory);

            //pvss[i] = renderVisibilityCube(cameraForwards, cameraUps, positions[i]);
            //pvsCache[hash] = pvss[i];
            //std::cout << hash << std::endl;

            //pvsCache[hash] = renderVisibilityCube(cameraForwards, cameraUps, positions[i]);

            renderVisibilityCube(cameraForwards, cameraUps, positions[i]);
        } else {
            //std::cout << hash << std::endl;
            hashes.push_back(hash);
        }
    }

    // Copy current pvs to host to fill cache
    auto startTotal = std::chrono::steady_clock::now();
    {
         for (int i = 0; i < positions.size(); i++) {
            if (hashes[i] == 0) {
                auto hash = std::hash<glm::vec3>()(positions[i]);

                VkDeviceSize bufferSize = sizeof(int) * MAX_NUM_TRIANGLES;

                VkBuffer hostBuffer;
                VkDeviceMemory hostBufferMemory;
                VulkanUtil::createBuffer(
                    physicalDevice,
                    logicalDevice, bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT, hostBuffer, hostBufferMemory,
                    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT
                );

                VulkanUtil::copyBuffer(
                    logicalDevice, transferCommandPool, transferQueue, currentPvsBuffer,
                    hostBuffer, bufferSize, sizeof(int) * MAX_NUM_TRIANGLES * i
                );

                void *data;
                vkMapMemory(logicalDevice, hostBufferMemory, 0, bufferSize, 0, &data);

                int* pvsArray = (int*)data;
                pvsCache[hash].insert(pvsCache[hash].end(), pvsArray, pvsArray + MAX_NUM_TRIANGLES);

                vkUnmapMemory(logicalDevice, hostBufferMemory);
            }
        }
    }
    fillCacheTime += std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - startTotal).count();

    {
        VkDeviceSize bufferSize = sizeof(int) * MAX_NUM_TRIANGLES;

        // Create staging buffer using host-visible memory
        VkBuffer stagingBuffer;
        VkDeviceMemory stagingBufferMemory;
        VulkanUtil::createBuffer(
            physicalDevice, logicalDevice, bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            stagingBuffer, stagingBufferMemory,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT
        );

        for (int i = 0; i < positions.size(); i++) {
            if (hashes[i] != 0) {
                // Copy triangles data to the staging buffer
                void *data;
                vkMapMemory(logicalDevice, stagingBufferMemory, 0, bufferSize, 0, &data);    // Map buffer memory into CPU accessible memory
                memcpy(data, pvsCache[hashes[i]].data(), (size_t) bufferSize);  // Copy vertex data to mapped memory
                vkUnmapMemory(logicalDevice, stagingBufferMemory);

                VulkanUtil::copyBuffer(
                    logicalDevice, transferCommandPool, transferQueue, stagingBuffer,
                    currentPvsBuffer, bufferSize, 0, sizeof(int) * MAX_NUM_TRIANGLES * i
                );
            }
        }

        vkDestroyBuffer(logicalDevice, stagingBuffer, nullptr);
        vkFreeMemory(logicalDevice, stagingBufferMemory, nullptr);
    }

    /*
    {
        VkDeviceSize bufferSize = sizeof(int) * MAX_NUM_TRIANGLES * 5;

        VkBuffer hostBuffer;
        VkDeviceMemory hostBufferMemory;
        VulkanUtil::createBuffer(
            physicalDevice,
            logicalDevice, bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT, hostBuffer, hostBufferMemory,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT
        );

        VulkanUtil::copyBuffer(
            logicalDevice, transferCommandPool, transferQueue, currentPvsBuffer,
            hostBuffer, bufferSize
        );

        void *data;
        vkMapMemory(logicalDevice, hostBufferMemory, 0, bufferSize, 0, &data);

        int* a = (int*)data;
        std::vector<int> v0(a + MAX_NUM_TRIANGLES * 0, a + MAX_NUM_TRIANGLES * 1);
        std::vector<int> v1(a + MAX_NUM_TRIANGLES * 1, a + MAX_NUM_TRIANGLES * 2);
        std::vector<int> v2(a + MAX_NUM_TRIANGLES * 2, a + MAX_NUM_TRIANGLES * 3);
        std::vector<int> v3(a + MAX_NUM_TRIANGLES * 3, a + MAX_NUM_TRIANGLES * 4);

        int common = 0;
        for (int i = 0; i < v0.size(); i++) {
            if (v0[i] == -1) {
                continue;
            }
            if (
                std::find(v1.begin(), v1.end(), v0[i]) != v1.end() &&
                std::find(v2.begin(), v2.end(), v0[i]) != v2.end() &&
                std::find(v3.begin(), v3.end(), v0[i]) != v3.end()
            ) {
                common++;
            }
        }

        int size = 0;

        int tempSize = 0;
        for (int i = 0; i < v0.size(); i++) {
            if (v0[i] > -1) {
                tempSize++;
            }
        }
        size = std::max(size, tempSize);
        tempSize = 0;
        for (int i = 0; i < v1.size(); i++) {
            if (v1[i] > -1) {
                tempSize++;
            }
        }
        size = std::max(size, tempSize);
        tempSize = 0;
        for (int i = 0; i < v2.size(); i++) {
            if (v2[i] > -1) {
                tempSize++;
            }
        }
        size = std::max(size, tempSize);
        tempSize = 0;
        for (int i = 0; i < v3.size(); i++) {
            if (v3[i] > -1) {
                tempSize++;
            }
        }
        size = std::max(size, tempSize);

        vkUnmapMemory(logicalDevice, hostBufferMemory);

        std::cout << "common " << common << " size " << size << std::endl;
    }
    */

    startTotal = std::chrono::steady_clock::now();
    int numCommonIDs = CUDAUtil::setIntersection(currentPvsCuda, MAX_NUM_TRIANGLES);
    int largestSetSize = CUDAUtil::calculateLargestSetSize(currentPvsCuda, MAX_NUM_TRIANGLES);
    cudaTime += std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - startTotal).count();
    float nirensteinCriterion = 1.0f - float(numCommonIDs) / float(largestSetSize);
    //std::cout << "numCommonIDs " << numCommonIDs << " largestSetSize " << largestSetSize << std::endl;
    std::cout << "nirensteinCriterion " << nirensteinCriterion << " " << numCommonIDs << " " << ERROR_THRESHOLD << std::endl;

    if (nirensteinCriterion > ERROR_THRESHOLD) {
        numSubdivisions++;
        if (numSubdivisions > MAX_SUBDIVISIONS) {
            return;
        }

        std::vector<glm::vec3> newPositions = {
            positions[0] + ((positions[1] - positions[0]) / 2.0f) + ((positions[1] - positions[0]) / 2.0f) *         (rand() / float(RAND_MAX)) * 0.0f,
            positions[0] + ((positions[2] - positions[0]) / 2.0f) + ((positions[2] - positions[0]) / 2.0f) *         (rand() / float(RAND_MAX)) * 0.0f,
            positions[1] + ((positions[3] - positions[1]) / 2.0f) + ((positions[3] - positions[1]) / 2.0f) *         (rand() / float(RAND_MAX)) * 0.0f,
            positions[2] + ((positions[3] - positions[2]) / 2.0f) + ((positions[3] - positions[2]) / 2.0f) *         (rand() / float(RAND_MAX)) * 0.0f,
            positions[0] + ((positions[3] - positions[0]) / 2.0f) + ((positions[3] - positions[0]) / 2.0f) *         (rand() / float(RAND_MAX)) * 0.0f
        };

        //std::cout << glm::to_string(viewCell.model * glm::vec4(-1.0f,-1.0f,0.0f,1.0f)) << std::endl;
        //std::cout << glm::to_string(viewCell.model * glm::vec4(1.0f,1.0f,0.0f,1.0f)) << std::endl;

        glm::vec3 corner0 = viewCell.model * glm::vec4(-1.0f,-1.0f,0.0f,1.0f);
        glm::vec3 corner1 = viewCell.model * glm::vec4(1.0f,1.0f,0.0f,1.0f);
        for (int i = 0; i < newPositions.size(); i++) {
            newPositions[i] = glm::clamp(
                newPositions[i],
                glm::vec3(std::min(corner0.x, corner1.x), std::min(corner0.y, corner1.y), std::min(corner0.z, corner1.z)),
                glm::vec3(std::max(corner0.x, corner1.x), std::max(corner0.y, corner1.y), std::max(corner0.z, corner1.z))
            );
        }

        divideAdaptive(
            viewCell, viewCellSize,
            cameraForward,
            { positions[0], newPositions[0], newPositions[1], newPositions[4] }
        );
        divideAdaptive(
            viewCell, viewCellSize,
            cameraForward,
            { newPositions[0], positions[1], newPositions[4], newPositions[2] }
        );
        divideAdaptive(
            viewCell, viewCellSize,
            cameraForward,
            { newPositions[1], newPositions[4], positions[2], newPositions[3] }
        );
        divideAdaptive(
            viewCell, viewCellSize,
            cameraForward,
            { newPositions[4], newPositions[2], newPositions[3], positions[3] }
        );
    }
}

void NirensteinSampler::divideHaltonRandom(
    const ViewCell &viewCell, glm::vec3 cameraForward, const std::vector<glm::vec2> &haltonPoints
) {
    const glm::vec3 cameraRight = glm::normalize(glm::cross(cameraForward, glm::vec3(0.0f, 1.0f, 0.0f)));
    const glm::vec3 cameraUp = glm::normalize(glm::cross(cameraForward, cameraRight));

    std::array<glm::vec3, 5> cameraForwards = {
        cameraForward,
        cameraRight,
        -cameraRight,
        cameraUp,
        -cameraUp
    };
    std::array<glm::vec3, 5> cameraUps = {
        cameraUp,
        -cameraUp,
        cameraUp,
        cameraForward,
        cameraForward
    };

    std::array<glm::vec2, 4> corners = {
        glm::vec2(-1.0f, -1.0f),
        glm::vec2(-1.0f, 1.0f),
        glm::vec2(1.0f, -1.0f),
        glm::vec2(1.0f, 1.0f)
    };
    std::unordered_set<int> oldPvs;
    for (int i = 0; i < haltonPoints.size() + 4; i++) {
        glm::vec4 position;
        if (i < 4) {
            position = viewCell.model * glm::vec4(corners[i].x, corners[i].y , 0.0f, 1.0f);
        } else {
            position = viewCell.model * glm::vec4(haltonPoints[i - 4].x * 2.0f - 1.0f, haltonPoints[i - 4].y * 2.0f - 1.0f, 0.0f, 1.0f);
        }
        renderCubePositions.push_back({ position.x, position.y, position.z });

        // Update cube position uniform
        {
            VkDeviceSize bufferSize = sizeof(glm::vec3);

            // Create staging buffer using host-visible memory
            VkBuffer stagingBuffer;
            VkDeviceMemory stagingBufferMemory;
            VulkanUtil::createBuffer(
                physicalDevice, logicalDevice, bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                stagingBuffer, stagingBufferMemory,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT
            );

            // Copy triangles data to the staging buffer
            glm::vec3 cubePos[1] = { position };
            void *data;
            vkMapMemory(logicalDevice, stagingBufferMemory, 0, bufferSize, 0, &data);    // Map buffer memory into CPU accessible memory
            memcpy(data, &cubePos, (size_t) bufferSize);  // Copy vertex data to mapped memory
            vkUnmapMemory(logicalDevice, stagingBufferMemory);

            // Copy triangles data from the staging buffer to GPU-visible absWorkingBuffer
            VulkanUtil::copyBuffer(
                logicalDevice, transferCommandPool, transferQueue, stagingBuffer,
                cubePosUniformBuffer, bufferSize
            );

            vkDestroyBuffer(logicalDevice, stagingBuffer, nullptr);
            vkFreeMemory(logicalDevice, stagingBufferMemory, nullptr);
        }

        renderVisibilityCube(cameraForwards, cameraUps, position);

        /*
        {
            VkDeviceSize bufferSize = sizeof(int) * MAX_NUM_TRIANGLES;

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

            int numNew = 0;
            int* pvsArray = (int*)data;
            oldPvs.insert(pvsArray, pvsArray + MAX_NUM_TRIANGLES);

            std::cout << oldPvs.size() << std::endl;

            vkUnmapMemory(logicalDevice, hostBufferMemory);
        }
        */
    }
}

void NirensteinSampler::resetPVS() {
    // Fill PVS buffer with -1
    VkDeviceSize pvsSize = sizeof(int) * MAX_NUM_TRIANGLES;

    VkDeviceSize bufferSize = pvsSize;

    // Create staging buffer using host-visible memory
    VkBuffer stagingBuffer;
    VkDeviceMemory stagingBufferMemory;
    VulkanUtil::createBuffer(
        physicalDevice,
        logicalDevice, bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        stagingBuffer, stagingBufferMemory,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT
    );

    std::vector<int> vec(MAX_NUM_TRIANGLES);
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
}

void NirensteinSampler::printAverageStatistics() {
    float avgRenderTime = 0.0f;
    for (auto renderTime : avgRenderTimes) {
        avgRenderTime += renderTime;
    }
    avgRenderTime /= float(avgRenderTimes.size());

    float avgTotalRenderTime = 0.0f;
    for (auto totalRenderTime : avgTotalRenderTimes) {
        avgTotalRenderTime += totalRenderTime;
    }
    avgTotalRenderTime /= float(avgTotalRenderTimes.size());

    float avgPVSSize = 0.0f;
    for (auto pvsSize : pvsSizes) {
        avgPVSSize += pvsSize;
    }
    avgPVSSize /= pvsSizes.size();
    avgPVSSize /= float(MAX_NUM_TRIANGLES);

    std::cout << "Average total render time: " << avgTotalRenderTime << "ms" << std::endl;
    std::cout << "Average single hemicube render time: " << avgRenderTime << "ms" << std::endl;
    std::cout << "Average PVS size: " << avgPVSSize * 100.0f << "%" << std::endl;
}

void NirensteinSampler::createRenderPass(VkFormat depthFormat) {
    VkAttachmentDescription colorAttachment = {};
    colorAttachment.format = VK_FORMAT_R32_SINT; //VK_FORMAT_R32_SINT
    colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
    colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;   // Clear framebuffer before rendering
    colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;    // No stencil buffer is used
    colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;   // No stencil buffer is used
    colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;      // We don't care about the layout before rendering
    colorAttachment.finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL; //VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL; //VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL; //VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL; //VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    VkAttachmentReference colorAttachmentReference = {};
    colorAttachmentReference.attachment = 0;
    colorAttachmentReference.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    VkAttachmentDescription depthAttachment = {};
    depthAttachment.format = depthFormat;
    depthAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
    depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    depthAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    depthAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    depthAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    depthAttachment.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    VkAttachmentReference depthAttachmentReference = {};
    depthAttachmentReference.attachment = 1;
    depthAttachmentReference.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    VkSubpassDescription subpassDescription = {};
    subpassDescription.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpassDescription.colorAttachmentCount = 1;
    subpassDescription.pColorAttachments = &colorAttachmentReference;
    subpassDescription.pDepthStencilAttachment = &depthAttachmentReference;

    VkRenderPassMultiviewCreateInfo renderPassMultiviewCI = {};
    const uint32_t viewMask = 0b00011111;
    const uint32_t correlationMask = 0b00000000;
    if (USE_MULTI_VIEW_RENDERING) {
        renderPassMultiviewCI.sType = VK_STRUCTURE_TYPE_RENDER_PASS_MULTIVIEW_CREATE_INFO;
        renderPassMultiviewCI.subpassCount = 1;
        renderPassMultiviewCI.pViewMasks = &viewMask;
        renderPassMultiviewCI.correlationMaskCount = 1;
        renderPassMultiviewCI.pCorrelationMasks = &correlationMask;
    }

    std::array<VkAttachmentDescription, 2> attachments = {
        colorAttachment, depthAttachment
    };
    VkRenderPassCreateInfo renderPassInfo = {};
    renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    renderPassInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
    renderPassInfo.pAttachments = attachments.data();
    renderPassInfo.subpassCount = 1;
    renderPassInfo.pSubpasses = &subpassDescription;
    if (USE_MULTI_VIEW_RENDERING) {
        renderPassInfo.pNext = &renderPassMultiviewCI;
    }

    if (vkCreateRenderPass(logicalDevice, &renderPassInfo, nullptr, &renderPass) != VK_SUCCESS) {
        throw std::runtime_error("failed to create nirenstein render pass");
    }
}

void NirensteinSampler::createDescriptorPool() {
    std::array<VkDescriptorPoolSize, 3> poolSizes = {};
    poolSizes[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    poolSizes[0].descriptorCount = 1;
    poolSizes[1].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    poolSizes[1].descriptorCount = 3;
    poolSizes[2].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    poolSizes[2].descriptorCount = 1;

    VkDescriptorPoolCreateInfo poolInfo = {};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
    poolInfo.pPoolSizes = poolSizes.data();
    poolInfo.maxSets = 2;

    if (vkCreateDescriptorPool(
            logicalDevice, &poolInfo, nullptr, &descriptorPool
        ) != VK_SUCCESS
    ) {
        throw std::runtime_error("failed to create nirenstein descriptor pool");
    }
}

void NirensteinSampler::createDescriptorSet() {
    std::array<VkWriteDescriptorSet, 1> descriptorWrites = {};

    VkDescriptorBufferInfo bufferInfo = {};
    bufferInfo.buffer = uniformBuffer;
    bufferInfo.offset = 0;
    if (USE_MULTI_VIEW_RENDERING) {
        bufferInfo.range = sizeof(UniformBufferObjectMultiView);
    } else {
        bufferInfo.range = sizeof(UniformBufferObject);
    }
    descriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptorWrites[0].dstSet = nirensteinDescriptorSet;
    descriptorWrites[0].dstBinding = 0;
    descriptorWrites[0].dstArrayElement = 0;
    descriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    descriptorWrites[0].descriptorCount = 1;
    descriptorWrites[0].pBufferInfo = &bufferInfo;

    vkUpdateDescriptorSets(
        logicalDevice,
        static_cast<uint32_t>(descriptorWrites.size()),
        descriptorWrites.data(),
        0,
        nullptr
    );
}

void NirensteinSampler::createDescriptorSetLayout() {
    VkDescriptorSetLayoutBinding uboLayoutBinding = {};
    uboLayoutBinding.binding = 0;
    uboLayoutBinding.descriptorCount = 1;
    uboLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    uboLayoutBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_COMPUTE_BIT;

    std::array<VkDescriptorSetLayoutBinding, 1> bindings = {
        uboLayoutBinding
    };
    VkDescriptorSetLayoutCreateInfo layoutInfo = {};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
    layoutInfo.pBindings = bindings.data();

    if (vkCreateDescriptorSetLayout(
            logicalDevice, &layoutInfo, nullptr, &nirensteinDescriptorSetLayout
        ) != VK_SUCCESS
    ) {
        throw std::runtime_error("failed to create descriptor set layout");
    }
}

void NirensteinSampler::createComputeDescriptorSet() {
    std::array<VkWriteDescriptorSet, 11> descriptorWrites = {};

    VkDescriptorBufferInfo pvsBufferInfo = {};
    pvsBufferInfo.buffer = pvsBuffer;
    pvsBufferInfo.offset = 0;
    pvsBufferInfo.range = VK_WHOLE_SIZE;
    descriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptorWrites[0].dstSet = computeDescriptorSet;
    descriptorWrites[0].dstBinding = 0;
    descriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    descriptorWrites[0].descriptorCount = 1;
    descriptorWrites[0].pBufferInfo = &pvsBufferInfo;

    VkDescriptorBufferInfo currentPvsBufferInfo = {};
    currentPvsBufferInfo.buffer = currentPvsBuffer;
    currentPvsBufferInfo.offset = 0;
    currentPvsBufferInfo.range = VK_WHOLE_SIZE;
    descriptorWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptorWrites[1].dstSet = computeDescriptorSet;
    descriptorWrites[1].dstBinding = 1;
    descriptorWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    descriptorWrites[1].descriptorCount = 1;
    descriptorWrites[1].pBufferInfo = &currentPvsBufferInfo;

    VkDescriptorBufferInfo triangleIDFramebufferInfo = {};
    triangleIDFramebufferInfo.buffer = triangleIDBuffer;
    triangleIDFramebufferInfo.offset = 0;
    triangleIDFramebufferInfo.range = VK_WHOLE_SIZE;
    descriptorWrites[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptorWrites[2].dstSet = computeDescriptorSet;
    descriptorWrites[2].dstBinding = 2;
    descriptorWrites[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    descriptorWrites[2].descriptorCount = 1;
    descriptorWrites[2].pBufferInfo = &triangleIDFramebufferInfo;

    VkDescriptorBufferInfo pvsSizeUniformBufferInfo = {};
    pvsSizeUniformBufferInfo.buffer = pvsSizeUniformBuffer;
    pvsSizeUniformBufferInfo.offset = 0;
    pvsSizeUniformBufferInfo.range = sizeof(int);
    descriptorWrites[3].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptorWrites[3].dstSet = computeDescriptorSet;
    descriptorWrites[3].dstBinding = 3;
    descriptorWrites[3].dstArrayElement = 0;
    descriptorWrites[3].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    descriptorWrites[3].descriptorCount = 1;
    descriptorWrites[3].pBufferInfo = &pvsSizeUniformBufferInfo;

    VkDescriptorBufferInfo currentPvsIndexUniformBufferInfo = {};
    currentPvsIndexUniformBufferInfo.buffer = currentPvsIndexUniformBuffer;
    currentPvsIndexUniformBufferInfo.offset = 0;
    currentPvsIndexUniformBufferInfo.range = sizeof(int);
    descriptorWrites[4].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptorWrites[4].dstSet = computeDescriptorSet;
    descriptorWrites[4].dstBinding = 4;
    descriptorWrites[4].dstArrayElement = 0;
    descriptorWrites[4].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    descriptorWrites[4].descriptorCount = 1;
    descriptorWrites[4].pBufferInfo = &currentPvsIndexUniformBufferInfo;

    VkDescriptorImageInfo framebufferInfo = {};
    framebufferInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    framebufferInfo.imageView = colorImageView;
    framebufferInfo.sampler = colorImageSampler;
    descriptorWrites[5].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptorWrites[5].dstSet = computeDescriptorSet;
    descriptorWrites[5].dstBinding = 5;
    descriptorWrites[5].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    descriptorWrites[5].descriptorCount = 1;
    descriptorWrites[5].pImageInfo = &framebufferInfo;

    VkDescriptorBufferInfo sampleOutputBufferInfo = {};
    sampleOutputBufferInfo.buffer = sampleOutputBuffer;
    sampleOutputBufferInfo.offset = 0;
    sampleOutputBufferInfo.range = VK_WHOLE_SIZE;
    descriptorWrites[6].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptorWrites[6].dstSet = computeDescriptorSet;
    descriptorWrites[6].dstBinding = 6;
    descriptorWrites[6].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    descriptorWrites[6].descriptorCount = 1;
    descriptorWrites[6].pBufferInfo = &sampleOutputBufferInfo;

    VkDescriptorBufferInfo cubePosUniformBufferInfo = {};
    cubePosUniformBufferInfo.buffer = cubePosUniformBuffer;
    cubePosUniformBufferInfo.offset = 0;
    cubePosUniformBufferInfo.range = VK_WHOLE_SIZE;
    descriptorWrites[7].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptorWrites[7].dstSet = computeDescriptorSet;
    descriptorWrites[7].dstBinding = 7;
    descriptorWrites[7].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    descriptorWrites[7].descriptorCount = 1;
    descriptorWrites[7].pBufferInfo = &cubePosUniformBufferInfo;

    VkDescriptorBufferInfo sampleCounterBufferInfo = {};
    sampleCounterBufferInfo.buffer = numSamplesBuffer;
    sampleCounterBufferInfo.offset = 0;
    sampleCounterBufferInfo.range = VK_WHOLE_SIZE;
    descriptorWrites[8].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptorWrites[8].dstSet = computeDescriptorSet;
    descriptorWrites[8].dstBinding = 8;
    descriptorWrites[8].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    descriptorWrites[8].descriptorCount = 1;
    descriptorWrites[8].pBufferInfo = &sampleCounterBufferInfo;

    VkDescriptorBufferInfo setBufferInfo = {};
    setBufferInfo.buffer = setBuffer;
    setBufferInfo.offset = 0;
    setBufferInfo.range = VK_WHOLE_SIZE;
    descriptorWrites[9].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptorWrites[9].dstSet = computeDescriptorSet;
    descriptorWrites[9].dstBinding = 9;
    descriptorWrites[9].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    descriptorWrites[9].descriptorCount = 1;
    descriptorWrites[9].pBufferInfo = &setBufferInfo;

    VkDescriptorBufferInfo triangleCounterBufferInfo = {};
    triangleCounterBufferInfo.buffer = triangleCounterBuffer;
    triangleCounterBufferInfo.offset = 0;
    triangleCounterBufferInfo.range = VK_WHOLE_SIZE;
    descriptorWrites[10].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptorWrites[10].dstSet = computeDescriptorSet;
    descriptorWrites[10].dstBinding = 10;
    descriptorWrites[10].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    descriptorWrites[10].descriptorCount = 1;
    descriptorWrites[10].pBufferInfo = &triangleCounterBufferInfo;

    vkUpdateDescriptorSets(
        logicalDevice,
        static_cast<uint32_t>(descriptorWrites.size()),
        descriptorWrites.data(),
        0,
        VK_NULL_HANDLE
    );
}

void NirensteinSampler::createComputeDescriptorSetLayout() {
    VkDescriptorSetLayoutBinding pvsBufferBinding = {};
    pvsBufferBinding.binding = 0;
    pvsBufferBinding.descriptorCount = 1;
    pvsBufferBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    pvsBufferBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    VkDescriptorSetLayoutBinding currentPvsBufferBinding = {};
    currentPvsBufferBinding.binding = 1;
    currentPvsBufferBinding.descriptorCount = 1;
    currentPvsBufferBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    currentPvsBufferBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    VkDescriptorSetLayoutBinding triangleIDFramebufferBinding = {};
    triangleIDFramebufferBinding.binding = 2;
    triangleIDFramebufferBinding.descriptorCount = 1;
    triangleIDFramebufferBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    triangleIDFramebufferBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    VkDescriptorSetLayoutBinding pvsSizeUniformBufferBinding = {};
    pvsSizeUniformBufferBinding.binding = 3;
    pvsSizeUniformBufferBinding.descriptorCount = 1;
    pvsSizeUniformBufferBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    pvsSizeUniformBufferBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    VkDescriptorSetLayoutBinding currentPvsIndexUniformBuffer = {};
    currentPvsIndexUniformBuffer.binding = 4;
    currentPvsIndexUniformBuffer.descriptorCount = 1;
    currentPvsIndexUniformBuffer.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    currentPvsIndexUniformBuffer.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    VkDescriptorSetLayoutBinding framebufferBinding = {};
    framebufferBinding.binding = 5;
    framebufferBinding.descriptorCount = 1;
    framebufferBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    framebufferBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    VkDescriptorSetLayoutBinding sampleOutputBinding = {};
    sampleOutputBinding.binding = 6;
    sampleOutputBinding.descriptorCount = 1;
    sampleOutputBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    sampleOutputBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    VkDescriptorSetLayoutBinding cubePosBinding = {};
    cubePosBinding.binding = 7;
    cubePosBinding.descriptorCount = 1;
    cubePosBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    cubePosBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    VkDescriptorSetLayoutBinding numSamplesBufferBinding = {};
    numSamplesBufferBinding.binding = 8;
    numSamplesBufferBinding.descriptorCount = 1;
    numSamplesBufferBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    numSamplesBufferBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    VkDescriptorSetLayoutBinding setBufferBinding = {};
    setBufferBinding.binding = 9;
    setBufferBinding.descriptorCount = 1;
    setBufferBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    setBufferBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    VkDescriptorSetLayoutBinding triangleCounterBufferBinding = {};
    triangleCounterBufferBinding.binding = 10;
    triangleCounterBufferBinding.descriptorCount = 1;
    triangleCounterBufferBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    triangleCounterBufferBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    std::array<VkDescriptorSetLayoutBinding, 11> bindings = {
        pvsBufferBinding,
        currentPvsBufferBinding,
        triangleIDFramebufferBinding,
        pvsSizeUniformBufferBinding,
        currentPvsIndexUniformBuffer,
        framebufferBinding,
        sampleOutputBinding,
        cubePosBinding,
        numSamplesBufferBinding,
        setBufferBinding,
        triangleCounterBufferBinding
    };
    VkDescriptorSetLayoutCreateInfo layoutInfo = {};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
    layoutInfo.pBindings = bindings.data();

    if (vkCreateDescriptorSetLayout(
            logicalDevice, &layoutInfo, nullptr, &computeDescriptorSetLayout
        ) != VK_SUCCESS
    ) {
        throw std::runtime_error("failed to create nirenstein compute descriptor set layout");
    }
}

void NirensteinSampler::createComputePipeline() {
    VkPipelineShaderStageCreateInfo pipelineShaderStageCreateInfo = {};
    pipelineShaderStageCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    pipelineShaderStageCreateInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    pipelineShaderStageCreateInfo.module = VulkanUtil::createShader(logicalDevice, "shaders/nirenstein.comp.spv");
    pipelineShaderStageCreateInfo.pName = "main";

    VkPipelineLayoutCreateInfo pipelineLayoutInfo = {};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = 1;
    pipelineLayoutInfo.pSetLayouts = &computeDescriptorSetLayout;
    if (vkCreatePipelineLayout(
            logicalDevice, &pipelineLayoutInfo, nullptr, &computePipelineLayout
        ) != VK_SUCCESS
    ) {
        throw std::runtime_error("failed to create halton compute pipeline layout");
    }

    VkComputePipelineCreateInfo computePipelineCreateInfo = {};
    computePipelineCreateInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    computePipelineCreateInfo.stage = pipelineShaderStageCreateInfo;
    computePipelineCreateInfo.layout = computePipelineLayout;

    if (vkCreateComputePipelines(
            logicalDevice, 0, 1, &computePipelineCreateInfo, nullptr, &computePipeline
        ) != VK_SUCCESS
    ) {
        throw std::runtime_error("failed to create nirenstein compute pipeline");
    }
}

void NirensteinSampler::createBuffers(const int numTriangles) {
    if (USE_MULTI_VIEW_RENDERING) {
        VulkanUtil::createBuffer(
            physicalDevice,
            logicalDevice,
            sizeof(UniformBufferObjectMultiView),
            VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
            uniformBuffer,
            uniformBufferMemory,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
        );
    } else {
        VulkanUtil::createBuffer(
            physicalDevice,
            logicalDevice,
            sizeof(UniformBufferObject),
            VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
            uniformBuffer,
            uniformBufferMemory,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
        );
    }

    VulkanUtil::createBuffer(
        physicalDevice,
        logicalDevice,
        sizeof(int),
        VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
        pvsSizeUniformBuffer,
        pvsSizeUniformBufferMemory,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
    );

    VulkanUtil::createBuffer(
        physicalDevice,
        logicalDevice,
        sizeof(int),
        VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
        currentPvsIndexUniformBuffer,
        currentPvsIndexUniformBufferMemory,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
    );

    VulkanUtil::createBuffer(
        physicalDevice,
        logicalDevice, sizeof(int) * FRAME_BUFFER_WIDTH * FRAME_BUFFER_HEIGHT,
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        triangleIDBuffer, triangleIDBufferMemory, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
    );

    VulkanUtil::createBuffer(
        physicalDevice,
        logicalDevice, sizeof(int) * numTriangles,
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        pvsBuffer, pvsBufferMemory, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
    );



    VulkanUtil::createBuffer(
        physicalDevice,
        logicalDevice,
        sizeof(glm::vec3),
        VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
        cubePosUniformBuffer,
        cubePosUniformBufferMemory,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
    );
    VulkanUtil::createBuffer(
        physicalDevice,
        logicalDevice,
        sizeof(int),
        VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
        numSamplesBuffer,
        numSamplesBufferMemory,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
    );


    /*
    VulkanUtil::createBuffer(
        physicalDevice,
        logicalDevice, sizeof(int) * numTriangles * 4,
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        currentPvsBuffer, currentPvsBufferMemory, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
    );
    */
    CUDAUtil::createExternalBuffer(
        sizeof(int) * numTriangles * 5,
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, currentPvsBuffer,
        currentPvsBufferMemory, logicalDevice, physicalDevice
    );
    CUDAUtil::importCudaExternalMemory(
        (void**)&currentPvsCuda, currentPvsCudaMemory,
        currentPvsBufferMemory, sizeof(int) * numTriangles * 5, logicalDevice
    );

    resetPVS();



    {
        VkDeviceSize bufferSize = sizeof(int);

        // Create staging buffer using host-visible memory
        VkBuffer stagingBuffer;
        VkDeviceMemory stagingBufferMemory;
        VulkanUtil::createBuffer(
            physicalDevice, logicalDevice, bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            stagingBuffer, stagingBufferMemory,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT
        );

        // Copy triangles data to the staging buffer
        unsigned int numTriangles[1] = { 0 };
        void *data;
        vkMapMemory(logicalDevice, stagingBufferMemory, 0, bufferSize, 0, &data);    // Map buffer memory into CPU accessible memory
        memcpy(data, &numTriangles, (size_t) bufferSize);  // Copy vertex data to mapped memory
        vkUnmapMemory(logicalDevice, stagingBufferMemory);

        // Copy triangles data from the staging buffer to GPU-visible absWorkingBuffer
        VulkanUtil::copyBuffer(
            logicalDevice, transferCommandPool, transferQueue, stagingBuffer,
            numSamplesBuffer, bufferSize
        );

        vkDestroyBuffer(logicalDevice, stagingBuffer, nullptr);
        vkFreeMemory(logicalDevice, stagingBufferMemory, nullptr);
    }
}

void NirensteinSampler::createFramebuffer(VkFormat depthFormat) {
    VulkanUtil::createImage(
        physicalDevice, logicalDevice, FRAME_BUFFER_WIDTH, FRAME_BUFFER_HEIGHT, VK_SAMPLE_COUNT_1_BIT, VK_FORMAT_R32_SINT,
        VK_IMAGE_TILING_OPTIMAL,
        VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, //VK_IMAGE_USAGE_TRANSFER_SRC_BIT
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, colorImage, colorImageMemory, USE_MULTI_VIEW_RENDERING ? MULTI_VIEW_LAYER_COUNT : 1
    );
    colorImageView = VulkanUtil::createImageView(
        logicalDevice, colorImage, VK_FORMAT_R32_SINT, VK_IMAGE_ASPECT_COLOR_BIT, USE_MULTI_VIEW_RENDERING ? MULTI_VIEW_LAYER_COUNT : 1
    );

    VulkanUtil::createImage(
        physicalDevice, logicalDevice, FRAME_BUFFER_WIDTH, FRAME_BUFFER_HEIGHT, VK_SAMPLE_COUNT_1_BIT, depthFormat,
        VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, depthImage, depthImageMemory, USE_MULTI_VIEW_RENDERING ? MULTI_VIEW_LAYER_COUNT : 1
    );
    depthImageView = VulkanUtil::createImageView(
        logicalDevice, depthImage, depthFormat, VK_IMAGE_ASPECT_DEPTH_BIT, USE_MULTI_VIEW_RENDERING ? MULTI_VIEW_LAYER_COUNT : 1
    );

    VkSamplerCreateInfo samplerInfo = {};
    samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    samplerInfo.magFilter = VK_FILTER_NEAREST;
    samplerInfo.minFilter = VK_FILTER_NEAREST;
    samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.anisotropyEnable = VK_FALSE;
    samplerInfo.maxAnisotropy = 1.0f;
    samplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_WHITE;
    samplerInfo.unnormalizedCoordinates = VK_FALSE;
    samplerInfo.compareEnable = VK_FALSE;
    samplerInfo.compareOp = VK_COMPARE_OP_ALWAYS;
    samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    if (vkCreateSampler(logicalDevice, &samplerInfo, nullptr, &colorImageSampler) != VK_SUCCESS) {
        throw std::runtime_error("failed to create nirenstein color image sampler!");
    }

    std::array<VkImageView, 2> attachments = {
        colorImageView,
        depthImageView
    };

    VkFramebufferCreateInfo framebufferInfo = {};
    framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
    framebufferInfo.renderPass = renderPass;
    framebufferInfo.attachmentCount = static_cast<uint32_t>(attachments.size());;
    framebufferInfo.pAttachments = attachments.data();
    framebufferInfo.width = FRAME_BUFFER_WIDTH;
    framebufferInfo.height = FRAME_BUFFER_HEIGHT;
    framebufferInfo.layers = 1;

    if (vkCreateFramebuffer(logicalDevice, &framebufferInfo, nullptr, &framebuffer) != VK_SUCCESS) {
        throw std::runtime_error("failed to create nirenstein framebuffer");
    }
}

void NirensteinSampler::createCommandBuffers(
    VkBuffer vertexBuffer, const std::vector<Vertex> &vertices, VkBuffer indexBuffer,
    const std::vector<uint32_t> indices
) {
    createCommandBuffer(
        commandBufferRenderFront, vertexBuffer, vertices, indexBuffer, indices,
        { { 0, 0 }, { (uint32_t)FRAME_BUFFER_WIDTH, (uint32_t)FRAME_BUFFER_HEIGHT } }
    );
    if (!USE_MULTI_VIEW_RENDERING) {
        createCommandBuffer(
            commandBufferRenderSides, vertexBuffer, vertices, indexBuffer, indices,
            { { 0, 0 }, { (uint32_t)(FRAME_BUFFER_WIDTH * 0.5f), (uint32_t)FRAME_BUFFER_HEIGHT } }
        );
        createCommandBuffer(
            commandBufferRenderTopBottom, vertexBuffer, vertices, indexBuffer, indices,
            { { 0, 0 }, { (uint32_t)FRAME_BUFFER_WIDTH, (uint32_t)(FRAME_BUFFER_HEIGHT * 0.5f) } }
        );
    }
}

void NirensteinSampler::createCommandBuffer(
    VkCommandBuffer &commandBuffer, VkBuffer vertexBuffer, const std::vector<Vertex> &vertices,
    VkBuffer indexBuffer, const std::vector<uint32_t> indices, VkRect2D scissor
) {
    // Allocate command buffer
    VkCommandBufferAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandPool = graphicsCommandPool;
    allocInfo.commandBufferCount = 1;
    vkAllocateCommandBuffers(logicalDevice, &allocInfo, &commandBuffer);

    // Begin recording commands
    VkCommandBufferBeginInfo beginInfo = {};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    vkBeginCommandBuffer(commandBuffer, &beginInfo);

    // Rasterization
    VkRenderPassBeginInfo renderPassInfo = {};
    renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    renderPassInfo.renderPass = renderPass;
    renderPassInfo.framebuffer = framebuffer;
    renderPassInfo.renderArea.extent.width = FRAME_BUFFER_WIDTH;
    renderPassInfo.renderArea.extent.height = FRAME_BUFFER_HEIGHT;

    std::array<VkClearValue, 2> clearValues = {};
    clearValues[0].color = {{ 0 }}; //clearValues[0].color = {{ 0.0f, 0.0f, 0.0f, 1.0f }};
    clearValues[1].depthStencil = { 1.0f, 0 };
    renderPassInfo.clearValueCount = static_cast<uint32_t>(clearValues.size());
    renderPassInfo.pClearValues = clearValues.data();

    vkCmdBeginRenderPass(commandBuffer, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, nirensteinPipeline);

    // Define viewport
    VkViewport viewport = {};
    viewport.x = 0.0f;
    viewport.y = 0.0f;
    viewport.width = (float) FRAME_BUFFER_WIDTH;
    viewport.height = (float) FRAME_BUFFER_HEIGHT;
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;
    vkCmdSetViewport(commandBuffer, 0, 1, &viewport);

    vkCmdSetScissor(commandBuffer, 0, 1, &scissor);

    vkCmdBindDescriptorSets(
        commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, nirensteinPipelineLayout, 0, 1,
        &nirensteinDescriptorSet, 0, nullptr
    );

    // Draw scene
    VkDeviceSize offsets[] = { 0 };
    vkCmdBindVertexBuffers(commandBuffer, 0, 1, &vertexBuffer, offsets);
    vkCmdBindIndexBuffer(commandBuffer, indexBuffer, 0, VK_INDEX_TYPE_UINT32);
    vkCmdDrawIndexed(commandBuffer, static_cast<uint32_t>(indices.size()), 1, 0, 0, 0);
    vkCmdEndRenderPass(commandBuffer);
    vkEndCommandBuffer(commandBuffer);

    void *data;
    vkMapMemory(logicalDevice, pvsSizeUniformBufferMemory, 0, sizeof(int), 0, &data);
    if (USE_MULTI_VIEW_RENDERING) {
        memcpy(data, &MAX_NUM_TRIANGLES, sizeof(UniformBufferObjectMultiView));
    } else {
        memcpy(data, &MAX_NUM_TRIANGLES, sizeof(UniformBufferObject));
    }
    vkUnmapMemory(logicalDevice, pvsSizeUniformBufferMemory);
}

void NirensteinSampler::createComputeCommandBuffer() {
    // Allocate command buffer
    VkCommandBufferAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandPool = computeCommandPool;
    allocInfo.commandBufferCount = 1;
    vkAllocateCommandBuffers(logicalDevice, &allocInfo, &computeCommandBuffer);

    // Begin recording commands
    VkCommandBufferBeginInfo beginInfo = {};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    vkBeginCommandBuffer(computeCommandBuffer, &beginInfo);

    vkCmdBindPipeline(computeCommandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, computePipeline);
    vkCmdBindDescriptorSets(
        computeCommandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, computePipelineLayout,
        0, 1, &computeDescriptorSet, 0, nullptr
    );
    vkCmdDispatch(computeCommandBuffer, std::ceil(FRAME_BUFFER_WIDTH / 32.0f), std::ceil(FRAME_BUFFER_HEIGHT / 32.0f), 1);
    vkEndCommandBuffer(computeCommandBuffer);
}

void NirensteinSampler::releaseRessources() {
    vkDestroyFence(logicalDevice, fence, nullptr);
    vkFreeCommandBuffers(logicalDevice, graphicsCommandPool, 1, &commandBufferRenderFront);
    vkFreeCommandBuffers(logicalDevice, graphicsCommandPool, 1, &commandBufferRenderSides);
    vkFreeCommandBuffers(logicalDevice, graphicsCommandPool, 1, &commandBufferRenderTopBottom);
}

void NirensteinSampler::createPipeline(
    VkPipeline &pipeline, VkPipelineLayout &pipelineLayout, std::string vertShaderPath,
    std::string fragShaderPath, VkPipelineLayoutCreateInfo pipelineLayoutInfo
) {
    VkShaderModule vertShaderModule = VulkanUtil::createShader(logicalDevice, vertShaderPath);
    VkShaderModule fragShaderModule = VulkanUtil::createShader(logicalDevice, fragShaderPath);

    // Assign vertex shader module to the vertex shader stage
    VkPipelineShaderStageCreateInfo vertShaderStageInfo = {};
    vertShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    vertShaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
    vertShaderStageInfo.module = vertShaderModule;
    vertShaderStageInfo.pName = "main";

    // Assign fragment shader module to the fragment shader stage
    VkPipelineShaderStageCreateInfo fragShaderStageInfo = {};
    fragShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    fragShaderStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    fragShaderStageInfo.module = fragShaderModule;
    fragShaderStageInfo.pName = "main";

    VkPipelineShaderStageCreateInfo shaderStages[] = {
        vertShaderStageInfo,
        fragShaderStageInfo
    };

    // Describe format of the vertex data
    auto vertexBindingDesc = Vertex::getBindingDescription();
    auto vertexAttributeDesc = Vertex::getAttributeDescriptions();

    VkPipelineVertexInputStateCreateInfo vertexInputInfo = {};
    vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vertexInputInfo.vertexBindingDescriptionCount = 1;
    vertexInputInfo.vertexAttributeDescriptionCount = static_cast<uint32_t>(vertexAttributeDesc.size());
    vertexInputInfo.pVertexBindingDescriptions = &vertexBindingDesc;
    vertexInputInfo.pVertexAttributeDescriptions = vertexAttributeDesc.data();

    // Describe input assembler
    VkPipelineInputAssemblyStateCreateInfo inputAssemblyInfo = {};
    inputAssemblyInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    inputAssemblyInfo.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    inputAssemblyInfo.primitiveRestartEnable = VK_FALSE;

    // Set viewport
    VkPipelineViewportStateCreateInfo viewportStateInfo = {};
    viewportStateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewportStateInfo.viewportCount = 1;
    viewportStateInfo.scissorCount = 1;

    // Describe rasterizer
    VkPipelineRasterizationStateCreateInfo rasterizerInfo = {};
    rasterizerInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    //rasterizerInfo.polygonMode = VK_POLYGON_MODE_LINE;        // Wireframe
    rasterizerInfo.polygonMode = VK_POLYGON_MODE_FILL;
    rasterizerInfo.lineWidth = 10.0f;
    rasterizerInfo.cullMode = VK_CULL_MODE_BACK_BIT;
    rasterizerInfo.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rasterizerInfo.rasterizerDiscardEnable = VK_FALSE;
    rasterizerInfo.depthClampEnable = VK_FALSE;
    rasterizerInfo.depthBiasEnable = VK_FALSE;

    // Enable multisampling
    VkPipelineMultisampleStateCreateInfo multisamplingInfo = {};
    multisamplingInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    //multisamplingInfo.rasterizationSamples = window->sampleCountFlagBits();
    multisamplingInfo.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    // Describe color blending
    VkPipelineColorBlendAttachmentState colorBlendAttachmentState = {};
    colorBlendAttachmentState.colorWriteMask = 0xF;
    colorBlendAttachmentState.blendEnable = VK_FALSE;
    VkPipelineColorBlendStateCreateInfo colorBlendingInfo = {};
    colorBlendingInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    colorBlendingInfo.attachmentCount = 1;
    colorBlendingInfo.pAttachments = &colorBlendAttachmentState;

    if (vkCreatePipelineLayout(logicalDevice, &pipelineLayoutInfo, nullptr, &pipelineLayout) != VK_SUCCESS) {
        throw std::runtime_error("failed to create pipeline layout");
    }

    VkDynamicState dynEnable[] = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
    VkPipelineDynamicStateCreateInfo dyndynamicStateInfo = {};
    memset(&dyndynamicStateInfo, 0, sizeof(dyndynamicStateInfo));
    dyndynamicStateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dyndynamicStateInfo.dynamicStateCount = 2; //sizeof(dynEnable) / sizeof(VkDynamicState);
    dyndynamicStateInfo.pDynamicStates = dynEnable;

    VkPipelineDepthStencilStateCreateInfo depthStencilStateInfo = {};
    memset(&depthStencilStateInfo, 0, sizeof(depthStencilStateInfo));
    depthStencilStateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    depthStencilStateInfo.depthTestEnable = VK_TRUE;
    depthStencilStateInfo.depthWriteEnable = VK_TRUE;
    depthStencilStateInfo.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL; //VK_COMPARE_OP_LESS;//VK_COMPARE_OP_LESS_OR_EQUAL;
    depthStencilStateInfo.stencilTestEnable = VK_FALSE;
    depthStencilStateInfo.depthBoundsTestEnable = VK_FALSE;
    //depthStencil.minDepthBounds = 0.0f; // Optional
    //depthStencil.maxDepthBounds = 1.0f; // Optional

    // Create graphics pipeline
    VkGraphicsPipelineCreateInfo graphicsPipelineInfo = {};
    graphicsPipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    graphicsPipelineInfo.stageCount = 2;
    graphicsPipelineInfo.pStages = shaderStages;
    graphicsPipelineInfo.pVertexInputState = &vertexInputInfo;
    graphicsPipelineInfo.pInputAssemblyState = &inputAssemblyInfo;
    graphicsPipelineInfo.pViewportState = &viewportStateInfo;
    graphicsPipelineInfo.pRasterizationState = &rasterizerInfo;
    graphicsPipelineInfo.pMultisampleState = &multisamplingInfo;
    graphicsPipelineInfo.pColorBlendState = &colorBlendingInfo;
    graphicsPipelineInfo.pDynamicState = &dyndynamicStateInfo;
    graphicsPipelineInfo.pDepthStencilState = &depthStencilStateInfo;
    graphicsPipelineInfo.layout = pipelineLayout;
    graphicsPipelineInfo.renderPass = renderPass;

    if (vkCreateGraphicsPipelines(
            logicalDevice, VK_NULL_HANDLE, 1, &graphicsPipelineInfo, nullptr, &pipeline
        ) != VK_SUCCESS
    ) {
        throw std::runtime_error("failed to create graphics pipeline");
    }

    vkDestroyShaderModule(logicalDevice, vertShaderModule, nullptr);
    vkDestroyShaderModule(logicalDevice, fragShaderModule, nullptr);
}
