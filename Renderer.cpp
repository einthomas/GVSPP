#include <vulkan/vulkan.h>
#include <vulkan/vulkan_core.h>

#define GLM_FORCE_RADIANS
#define STB_IMAGE_IMPLEMENTATION

#include <chrono>

//#define GLM_FORCE_DEFAULT_ALIGNED_GENTYPES
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/string_cast.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <stb_image.h>
#define TINYOBJLOADER_IMPLEMENTATION
#include <tiny_obj_loader.h>

#include "vulkanutil.h"
#include "Renderer.h"
#include "sample.h"

#include "viewcell.h"

#include "NirensteinSampler.h"

struct UniformBufferObjectMultiView {
    alignas(64) glm::mat4 model;
    alignas(64) glm::mat4 view;
    alignas(64) glm::mat4 projection;
};

/*
VulkanRenderer::VulkanRenderer(QVulkanWindow *w)
    : window(w), visibilityManager(RAYS_PER_ITERATION)
{
    const QVector<int> counts = w->supportedSampleCounts();
    qDebug() << "Supported sample counts:" << counts;
    for (int s = 16; s >= 4; s /= 2) {
        if (counts.contains(s)) {
            qDebug("Requesting sample count %d", s);
            window->setSampleCount(s);
            break;
        }
    }
}
*/

VulkanRenderer::VulkanRenderer(GLFWVulkanWindow *w)
    : window(w), visibilityManager()
{
    Settings settings = loadSettingsFile();

    std::cout << "compiling shaders..." << std::endl;
    system(se.at("SHADER_COMPILE_SCRIPT").c_str());

    std::cout << std::endl << "========================================" << std::endl;
    std::cout << "Settings loaded: " << std::endl;
    for (const auto &pair : se) {
        std::cout << "    " << pair.first << " " << pair.second << std::endl;
    }
    std::cout << "========================================" << std::endl << std::endl;

    std::vector<glm::mat4> viewCellMatrices = loadSceneFile(settings);

    createDescriptorSetLayout();
    {
        // Specify shader uniforms
        VkPushConstantRange pushConstantRangeModelMatrix = {};
        pushConstantRangeModelMatrix.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
        pushConstantRangeModelMatrix.size = sizeof(glm::mat4);
        pushConstantRangeModelMatrix.offset = 0;
        VkPushConstantRange pushConstantRangeShadedRendering = {};
        pushConstantRangeShadedRendering.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        pushConstantRangeShadedRendering.size = sizeof(VkBool32);
        pushConstantRangeShadedRendering.offset = sizeof(glm::mat4);

        VkPipelineLayoutCreateInfo pipelineLayoutInfo = {};
        pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipelineLayoutInfo.setLayoutCount = 1;
        pipelineLayoutInfo.pSetLayouts = &descriptorSetLayout;
        std::array<VkPushConstantRange, 2> pushConstantRanges = { pushConstantRangeShadedRendering, pushConstantRangeModelMatrix };
        pipelineLayoutInfo.pushConstantRangeCount = static_cast<uint32_t>(pushConstantRanges.size());
        pipelineLayoutInfo.pPushConstantRanges = pushConstantRanges.data();

        createGraphicsPipeline(
            pipeline, pipelineLayout, "shaders/shader.vert.spv", "shaders/shader.frag.spv",
            pipelineLayoutInfo
        );
    }
    {
        VkPipelineLayoutCreateInfo pipelineLayoutInfo = {};
        pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipelineLayoutInfo.setLayoutCount = 1;
        pipelineLayoutInfo.pSetLayouts = &descriptorSetLayout;

        createGraphicsPipeline(
            rayVisualizationPipeline, rayVisualizationPipelineLayout,
            "shaders/rayVisualizationShader.vert.spv", "shaders/rayVisualizationShader.frag.spv",
            pipelineLayoutInfo, VK_PRIMITIVE_TOPOLOGY_LINE_LIST
        );
    }

    createVertexBuffer(vertices, vertexBuffer, vertexBufferMemory);
    updateVertexBuffer(vertices, vertexBuffer, vertexBufferMemory);

    // Create index buffer using GPU memory
    VulkanUtil::createBuffer(
        window->physicalDevice,
        window->device, sizeof(indices[0]) * indices.size(), VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        indexBuffer, indexBufferMemory, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
    );
    createIndexBuffer();
    createUniformBuffers();
    createDescriptorPool();
    createDescriptorSets();

    cameraPos = glm::vec3(0.0f, 0.0f, 12.0f);
    glm::vec3 cameraTarget = glm::vec3(0.0f);
    cameraForward = glm::normalize(cameraTarget - cameraPos);
    cameraRight = glm::normalize(glm::cross(cameraForward, glm::vec3(0.0f, 1.0f, 0.0f)));
    cameraUp = glm::normalize(glm::cross(cameraForward, cameraRight));

    updateUniformBuffer(0);
    updateUniformBuffer(1);

    int reverseSamplingMethod = std::stoi(se.at("REVERSE_SAMPLING_METHOD"));
    int numReverseSamplingSamples = 2;
    if (reverseSamplingMethod == 1) {
        numReverseSamplingSamples = 15;
    } else if (reverseSamplingMethod == 2) {
        numReverseSamplingSamples = std::stoi(se.at("REVERSE_SAMPLING_NUM_SAMPLES_ALONG_EDGE")) * 4;
    }
    visibilityManager = new VisibilityManager(
        se.at("USE_TERMINATION_CRITERION") == "true",
        se.at("USE_RECURSIVE_EDGE_SUBDIVISION") == "true",
        std::stoi(se.at("RAY_COUNT_TERMINATION_THRESHOLD")),
        std::stoi(se.at("NEW_TRIANGLE_TERMINATION_THRESHOLD")),
        std::stoi(se.at("RANDOM_RAYS_PER_ITERATION")),
        std::stoi(se.at("ABS_MAX_SUBDIVISION_STEPS")),
        std::stoi(se.at("ABS_NUM_SAMPLES_PER_EDGE")) * 3,
        numReverseSamplingSamples,
        std::stoi(se.at("MAX_BULK_INSERT_BUFFER_SIZE")),
        std::stoi(se.at("SET_TYPE")),
        std::stoi(se.at("INITIAL_HASH_SET_SIZE")),
        window->physicalDevice,
        window->device,
        indexBuffer,
        indices,
        vertexBuffer,
        vertices,
        uniformBuffers,
        NUM_THREADS,
        window->deviceUUID,
        viewCellMatrices
    );

    nextCorner();
    alignCameraWithViewCellNormal();

    nirensteinSampler = new NirensteinSampler(
        window, visibilityManager->computeQueue, visibilityManager->commandPool[0], vertexBuffer,
        vertices, indexBuffer, indices, indices.size() / 3.0f,
        std::stof(se.at("NIRENSTEIN_ERROR_THRESHOLD")),
        std::stoi(se.at("NIRENSTEIN_MAX_SUBDIVISIONS")),
        USE_NIRENSTEIN_MULTI_VIEW_RENDERING
    );
}

void VulkanRenderer::initResources() {
}

void VulkanRenderer::initSwapChainResources() {
}

void VulkanRenderer::releaseSwapChainResources() {
}

void VulkanRenderer::releaseResources() {
    VkDevice device = window->device;

    vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);

    vkDestroyPipeline(device, pipeline, nullptr);
    vkDestroyPipelineLayout(device, pipelineLayout, nullptr);

    vkDestroyBuffer(device, vertexBuffer, nullptr);
    vkFreeMemory(device, vertexBufferMemory, nullptr);
    vkDestroyBuffer(device, shadedVertexBuffer, nullptr);
    vkFreeMemory(device, shadedVertexBufferMemory, nullptr);
    vkDestroyBuffer(device, indexBuffer, nullptr);
    vkFreeMemory(device, indexBufferMemory, nullptr);
    vkDestroyBuffer(device, rayVertexBuffer, nullptr);
    vkFreeMemory(device, rayVertexBufferMemory, nullptr);

    vkDestroyImageView(device, textureImageView, nullptr);
    vkDestroyImage(device, textureImage, nullptr);
    vkFreeMemory(device, textureImageMemory, nullptr);

    for (int i = 0; i < window->imageCount; i++) {
        vkDestroyBuffer(device, uniformBuffers[i], nullptr);
        vkFreeMemory(device, uniformBuffersMemory[i], nullptr);
    }

    vkDestroyDescriptorPool(device, descriptorPool, nullptr);

    visibilityManager->releaseResources();
}

void VulkanRenderer::createGraphicsPipeline(
    VkPipeline &pipeline, VkPipelineLayout &pipelineLayout, std::string vertShaderPath,
    std::string fragShaderPath, VkPipelineLayoutCreateInfo pipelineLayoutInfo,
    VkPrimitiveTopology primitiveTopology
) {
    VkShaderModule vertShaderModule = VulkanUtil::createShader(window->device, vertShaderPath);
    VkShaderModule fragShaderModule = VulkanUtil::createShader(window->device, fragShaderPath);

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
    inputAssemblyInfo.topology = primitiveTopology;
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
    //rasterizerInfo.cullMode = VK_CULL_MODE_BACK_BIT;
    rasterizerInfo.cullMode = VK_CULL_MODE_NONE;        // TODO: Activate back face culling
    rasterizerInfo.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rasterizerInfo.rasterizerDiscardEnable = VK_FALSE;
    rasterizerInfo.depthClampEnable = VK_FALSE;
    rasterizerInfo.depthBiasEnable = VK_FALSE;

    // Enable multisampling
    VkPipelineMultisampleStateCreateInfo multisamplingInfo = {};
    multisamplingInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    //multisamplingInfo.rasterizationSamples = window->sampleCountFlagBits();
    multisamplingInfo.rasterizationSamples = window->msaaSamples;

    // Describe color blending
    VkPipelineColorBlendAttachmentState colorBlendAttachmentState = {};
    colorBlendAttachmentState.colorWriteMask = 0xF;
    colorBlendAttachmentState.blendEnable = VK_FALSE;
    VkPipelineColorBlendStateCreateInfo colorBlendingInfo = {};
    colorBlendingInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    colorBlendingInfo.attachmentCount = 1;
    colorBlendingInfo.pAttachments = &colorBlendAttachmentState;

    if (vkCreatePipelineLayout(window->device, &pipelineLayoutInfo, nullptr, &pipelineLayout) != VK_SUCCESS) {
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
    depthStencilStateInfo.depthCompareOp = VK_COMPARE_OP_LESS;//VK_COMPARE_OP_LESS_OR_EQUAL;
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
    graphicsPipelineInfo.renderPass = window->renderPass;

    if (vkCreateGraphicsPipelines(
            window->device, VK_NULL_HANDLE, 1, &graphicsPipelineInfo, nullptr, &pipeline
        ) != VK_SUCCESS
    ) {
        throw std::runtime_error("failed to create graphics pipeline");
    }

    vkDestroyShaderModule(window->device, vertShaderModule, nullptr);
    vkDestroyShaderModule(window->device, fragShaderModule, nullptr);
}

void VulkanRenderer::createVertexBuffer(std::vector<Vertex> &vertices, VkBuffer &vertexBuffer, VkDeviceMemory &vertexBufferMemory) {
    VkDeviceSize bufferSize = sizeof(vertices[0]) * vertices.size();

    // Create vertex buffer using GPU memory
    VulkanUtil::createBuffer(
        window->physicalDevice,
        window->device, bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        vertexBuffer, vertexBufferMemory, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
    );
}

void VulkanRenderer::updateVertexBuffer(std::vector<Vertex> &vertices, VkBuffer &vertexBuffer, VkDeviceMemory &vertexBufferMemory) {
    VkDeviceSize bufferSize = sizeof(vertices[0]) * vertices.size();

    // Create staging buffer using host-visible memory
    VkBuffer stagingBuffer;
    VkDeviceMemory stagingBufferMemory;
    VulkanUtil::createBuffer(
        window->physicalDevice,
        window->device, bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, stagingBuffer, stagingBufferMemory,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
    );

    // Copy vertex data to the staging buffer
    void *data;
    vkMapMemory(window->device, stagingBufferMemory, 0, bufferSize, 0, &data);    // Map buffer memory into CPU accessible memory
    memcpy(data, vertices.data(), (size_t) bufferSize);  // Copy vertex data to mapped memory
    vkUnmapMemory(window->device, stagingBufferMemory);

    // Copy vertex data from the staging buffer to the vertex buffer
    VulkanUtil::copyBuffer(window->device, window->graphicsCommandPool, window->graphicsQueue, stagingBuffer, vertexBuffer, bufferSize);

    vkDestroyBuffer(window->device, stagingBuffer, nullptr);
    vkFreeMemory(window->device, stagingBufferMemory, nullptr);
}

void VulkanRenderer::createIndexBuffer() {
    VkDeviceSize bufferSize = sizeof(indices[0]) * indices.size();

    // Create staging buffer using host-visible memory
    VkBuffer stagingBuffer;
    VkDeviceMemory stagingBufferMemory;
    VulkanUtil::createBuffer(
        window->physicalDevice,
        window->device, bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, stagingBuffer, stagingBufferMemory,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
    );

    // Copy index data to the staging buffer
    void *data;
    vkMapMemory(window->device, stagingBufferMemory, 0, bufferSize, 0, &data);    // Map buffer memory into CPU accessible memory
    memcpy(data, indices.data(), (size_t) bufferSize);  // Copy vertex data to mapped memory
    vkUnmapMemory(window->device, stagingBufferMemory);

    // Copy index data from the staging buffer to the index buffer
    VulkanUtil::copyBuffer(
        window->device, window->graphicsCommandPool, window->graphicsQueue,stagingBuffer,
        indexBuffer, bufferSize
    );

    vkDestroyBuffer(window->device, stagingBuffer, nullptr);
    vkFreeMemory(window->device, stagingBufferMemory, nullptr);
}

void VulkanRenderer::loadModel(std::string modelPath) {
    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;
    std::string warn, err;

    if (!tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, modelPath.c_str())) {
        throw std::runtime_error((warn + err).c_str());
    }

    float scale = 1.0f;

    uint32_t i = 0;
    for (const auto &shape : shapes) {
        for (const auto &index : shape.mesh.indices) {
            Vertex vertex = {};

            vertex.pos = {
                attrib.vertices[3 * index.vertex_index + 0] * scale,
                attrib.vertices[3 * index.vertex_index + 1] * scale,
                attrib.vertices[3 * index.vertex_index + 2] * scale
            };

            vertex.normal = {
                attrib.normals[3 * index.normal_index + 0],
                attrib.normals[3 * index.normal_index + 1],
                attrib.normals[3 * index.normal_index + 2]
            };

            vertex.color = { 1.0f, 1.0f, 1.0f };

            if (index.texcoord_index >= 0) {
                vertex.texCoord = {
                    attrib.texcoords[2 * index.texcoord_index + 0],
                    1.0 - attrib.texcoords[2 * index.texcoord_index + 1],
                    0.0f
                };
            } else {
                vertex.texCoord = { 0.0f, 0.0f, 0.0f };
            }

            if (uniqueVertices.count(vertex) == 0) {
                uniqueVertices[vertex] = i;     // Store index of the vertex
                vertices.push_back(vertex);     // Store vertex
                i++;
            }

            indices.push_back(uniqueVertices[vertex]);
        }
    }
}

void VulkanRenderer::createTextureImage() {
    // Load texture from file
    int texWidth, texHeight, texChannels;
    auto pixels = stbi_load(
        "models/chalet.jpg", &texWidth, &texHeight, &texChannels, STBI_rgb_alpha
    );
    VkDeviceSize imageSize = texWidth * texHeight * 4;

    if (!pixels) {
        throw std::runtime_error("failed to load texture image");
    }

    // Create staging buffer using host-visible memory
    VkBuffer stagingBuffer;
    VkDeviceMemory stagingBufferMemory;
    VulkanUtil::createBuffer(
        window->physicalDevice,
        window->device, imageSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, stagingBuffer, stagingBufferMemory,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
    );

    // Copy image data to the staging buffer
    void *data;
    vkMapMemory(window->device, stagingBufferMemory, 0, imageSize, 0, &data);    // Map buffer memory into CPU accessible memory
    memcpy(data, pixels, static_cast<size_t>(imageSize));   // Copy vertex data to mapped memory
    vkUnmapMemory(window->device, stagingBufferMemory);

    stbi_image_free(pixels);

    // Create image object
    /*      // TODO
    createImage(
        static_cast<uint32_t>(texWidth), static_cast<uint32_t>(texHeight), VK_FORMAT_R8G8B8A8_SRGB,
        VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
        window->deviceLocalMemoryIndex(), textureImage, textureImageMemory
    );
    */

    // Change image layout to a layout optimal as destination in a transfer operation
    VkCommandBuffer commandBuffer = VulkanUtil::beginSingleTimeCommands(window->device, window->graphicsCommandPool);
    transitionImageLayout(
        commandBuffer, textureImage, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_LAYOUT_UNDEFINED,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT
    );
    VulkanUtil::endSingleTimeCommands(
        window->device, commandBuffer, window->graphicsCommandPool, window->graphicsQueue
    );

    // Copy the staging buffer to the texture image
    copyBufferToImage(
        stagingBuffer, textureImage, static_cast<uint32_t>(texWidth),
        static_cast<uint32_t>(texHeight)
    );

    // Change image layout to a layout optimal for sampling from a shader
    commandBuffer = VulkanUtil::beginSingleTimeCommands(window->device, window->graphicsCommandPool);
    transitionImageLayout(
        commandBuffer, textureImage, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT
    );
    VulkanUtil::endSingleTimeCommands(
        window->device, commandBuffer, window->graphicsCommandPool, window->graphicsQueue
    );
}

void VulkanRenderer::createTextureImageView() {
    VkImageViewCreateInfo viewInfo = {};
    viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewInfo.image = textureImage;
    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.format = VK_FORMAT_R8G8B8A8_SRGB;
    viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    viewInfo.subresourceRange.baseMipLevel = 0;
    viewInfo.subresourceRange.levelCount = 1;
    viewInfo.subresourceRange.baseArrayLayer = 0;
    viewInfo.subresourceRange.layerCount = 1;
    if (vkCreateImageView(window->device, &viewInfo, nullptr, &textureImageView) != VK_SUCCESS) {
        throw std::runtime_error("failed to create texture image view");
    }
}

void VulkanRenderer::createTextureSampler() {
    VkSamplerCreateInfo samplerInfo = {};
    samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;

    samplerInfo.magFilter = VK_FILTER_LINEAR;
    samplerInfo.minFilter = VK_FILTER_LINEAR;

    samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;

    samplerInfo.anisotropyEnable = VK_FALSE;
    //samplerInfo.maxAnisotropy = 16;

    samplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;

    samplerInfo.unnormalizedCoordinates = VK_FALSE;

    samplerInfo.compareEnable = VK_FALSE;
    samplerInfo.compareOp = VK_COMPARE_OP_ALWAYS;

    samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    samplerInfo.mipLodBias = 0.0f;
    samplerInfo.minLod = 0.0f;
    samplerInfo.maxLod = 0.0f;

    if (vkCreateSampler(window->device, &samplerInfo, nullptr, &textureSampler) != VK_SUCCESS) {
        throw std::runtime_error("failed to create texture sampler");
    }
}

void VulkanRenderer::transitionImageLayout(
    VkCommandBuffer commandBuffer, VkImage image, VkFormat format, VkImageLayout oldLayout,
    VkImageLayout newLayout, VkPipelineStageFlags sourceStage, VkPipelineStageFlags destinationStage
) {
    VkImageMemoryBarrier barrier = {};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.oldLayout = oldLayout;
    barrier.newLayout = newLayout;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = image;
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.baseMipLevel = 0;
    barrier.subresourceRange.levelCount = 1;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount = 1;
    barrier.srcAccessMask = sourceStage;
    barrier.dstAccessMask = destinationStage;

    switch (oldLayout) {
        case VK_IMAGE_LAYOUT_UNDEFINED:
            barrier.srcAccessMask = 0;
            break;
        case VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL:
            barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            break;
        case VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL:
            barrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
            break;
    }

    switch (newLayout) {
        case VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL:
            barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            break;
        case VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL:
            barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
            break;
        case VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL:
            if (barrier.srcAccessMask == 0) {
                barrier.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
            }
            barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            break;
    }

    vkCmdPipelineBarrier(
        commandBuffer,
        sourceStage,
        destinationStage,
        0,
        0,
        nullptr,
        0,
        nullptr,
        1,
        &barrier
    );
}

void VulkanRenderer::copyBufferToImage(
    VkBuffer buffer, VkImage image, uint32_t width, uint32_t height
) {
    VkCommandBuffer commandBuffer = VulkanUtil::beginSingleTimeCommands(window->device, window->graphicsCommandPool);

    // Specify which part of the buffer is going to be copied to which part of the image
    VkBufferImageCopy region = {};
    region.bufferOffset = 0;
    region.bufferRowLength = 0;
    region.bufferImageHeight = 0;
    region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.imageSubresource.mipLevel = 0;
    region.imageSubresource.baseArrayLayer = 0;
    region.imageSubresource.layerCount = 1;
    region.imageOffset = {0, 0, 0};
    region.imageExtent = {
        width,
        height,
        1
    };

    // Copy buffer to image
    vkCmdCopyBufferToImage(
        commandBuffer, buffer, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region
    );

    VulkanUtil::endSingleTimeCommands(
        window->device, commandBuffer, window->graphicsCommandPool, window->graphicsQueue
    );
}

void VulkanRenderer::createDescriptorSetLayout() {
    // A descriptor set layout describes the content of a (list of) descriptor set(s)

    // Describe each binding (used in a shader) via a VkDescriptorSetLayoutBinding
    // Uniform buffer descriptor
    VkDescriptorSetLayoutBinding uboLayoutBinding = {};
    uboLayoutBinding.binding = 0;
    uboLayoutBinding.descriptorCount = 1;
    uboLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    uboLayoutBinding.pImmutableSamplers = nullptr;
    uboLayoutBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_CLOSEST_HIT_BIT_NV | VK_SHADER_STAGE_RAYGEN_BIT_NV;

    // "Combined image sampler" descriptor. Allows shaders to access an image resource through a
    // sampler object
    VkDescriptorSetLayoutBinding samplerLayoutBinding = {};
    samplerLayoutBinding.binding = 1;
    samplerLayoutBinding.descriptorCount = 1;
    samplerLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    samplerLayoutBinding.pImmutableSamplers = nullptr;
    samplerLayoutBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_CLOSEST_HIT_BIT_NV;

    std::array<VkDescriptorSetLayoutBinding, 2> bindings = {
        uboLayoutBinding,
        samplerLayoutBinding
    };
    VkDescriptorSetLayoutCreateInfo layoutInfo = {};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
    layoutInfo.pBindings = bindings.data();

    if (vkCreateDescriptorSetLayout(
            window->device, &layoutInfo, nullptr, &descriptorSetLayout
        ) != VK_SUCCESS
    ) {
        throw std::runtime_error("failed to create descriptor set layout");
    }
}

void VulkanRenderer::createDescriptorPool() {
    std::array<VkDescriptorPoolSize, 2> poolSizes = {};
    poolSizes[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    poolSizes[0].descriptorCount = static_cast<uint32_t>(window->imageCount);
    poolSizes[1].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    poolSizes[1].descriptorCount = static_cast<uint32_t>(window->imageCount);

    VkDescriptorPoolCreateInfo poolInfo = {};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
    poolInfo.pPoolSizes = poolSizes.data();
    poolInfo.maxSets = static_cast<uint32_t>(window->imageCount);

    if (vkCreateDescriptorPool(
            window->device, &poolInfo, nullptr, &descriptorPool
        ) != VK_SUCCESS
    ) {
        throw std::runtime_error("failed to create descriptor pool");
    }
}

void VulkanRenderer::createDescriptorSets() {
    // A descriptor set specifies the actual buffer or image resource (just like a framebuffer
    // specifies the actual image view). Descriptor sets are allocated from a descriptor pool

    std::vector<VkDescriptorSetLayout> layouts(window->imageCount, descriptorSetLayout);
    //layouts.push_back(nirensteinDescriptorSetLayout);
    VkDescriptorSetAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = descriptorPool;
    allocInfo.descriptorSetCount = static_cast<uint32_t>(layouts.size());
    allocInfo.pSetLayouts = layouts.data();

    // Allocate descriptor sets (one descriptor set for each swap chain image)
    std::vector<VkDescriptorSet> descriptorSets;
    descriptorSets.resize(layouts.size());
    if (vkAllocateDescriptorSets(
            window->device, &allocInfo, descriptorSets.data()
        ) != VK_SUCCESS
    ) {
        throw std::runtime_error("failed to allocate descriptor sets");
    }
    for (int i = 0; i < window->imageCount; i++) {
        this->descriptorSets.push_back(descriptorSets[i]);
    }
    //nirensteinDescriptorSet = descriptorSets[descriptorSets.size() - 1];

    // Populate every descriptor
    for (int i = 0; i < window->imageCount; i++) {
        VkDescriptorBufferInfo bufferInfo = {};
        bufferInfo.buffer = uniformBuffers[i];
        bufferInfo.offset = 0;
        bufferInfo.range = sizeof(UniformBufferObjectMultiView);

        /*
        VkDescriptorImageInfo imageInfo = {};
        imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        imageInfo.imageView = textureImageView;
        imageInfo.sampler = textureSampler;
        */

        // Define write descriptor sets to copy data to the descriptors (i.e. the device memory)
        std::array<VkWriteDescriptorSet, 1> descriptorWrites = {};

        descriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWrites[0].dstSet = this->descriptorSets[i];
        descriptorWrites[0].dstBinding = 0;     // layout location in the shader
        descriptorWrites[0].dstArrayElement = 0;
        descriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        descriptorWrites[0].descriptorCount = 1;
        descriptorWrites[0].pBufferInfo = &bufferInfo;

        /*
        descriptorWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWrites[1].dstSet = descriptorSets[i];
        descriptorWrites[1].dstBinding = 1;     // layout location in the shader
        descriptorWrites[1].dstArrayElement = 0;
        descriptorWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        descriptorWrites[1].descriptorCount = 1;
        descriptorWrites[1].pImageInfo = &imageInfo;
        */

        vkUpdateDescriptorSets(
            window->device,
            static_cast<uint32_t>(descriptorWrites.size()),
            descriptorWrites.data(),
            0,
            nullptr
        );
    }
}

void VulkanRenderer::createUniformBuffers() {
    // Create one uniform buffer per swap chain image because multiple images might be rendered in
    // parallel
    uniformBuffers.resize(window->imageCount);
    uniformBuffersMemory.resize(window->imageCount);

    VkDeviceSize bufferSize = sizeof(UniformBufferObjectMultiView);
    for (int i = 0; i < window->imageCount; i++) {
        VulkanUtil::createBuffer(
            window->physicalDevice,
            window->device,
            bufferSize,
            VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
            uniformBuffers[i],
            uniformBuffersMemory[i],
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
        );
    }
}

void VulkanRenderer::updateUniformBuffer(uint32_t swapChainImageIndex) {
    UniformBufferObjectMultiView ubo;

    ubo.model = glm::mat4(1.0f);

    ubo.view = glm::lookAt(cameraPos, cameraPos + cameraForward, glm::vec3(0.0f, 1.0f, 0.0f));

    ubo.projection = glm::perspective(
        glm::radians(45.0f),
        window->swapChainImageSize.width / (float) window->swapChainImageSize.height,
        0.1f,
        100000.0f
    );
    ubo.projection[1][1] *= -1; // Flip y axis

    // Copy data in the uniform buffer object to the uniform buffer
    void *data;
    vkMapMemory(
        window->device, uniformBuffersMemory[swapChainImageIndex], 0, sizeof(ubo), 0, &data
    );
    memcpy(data, &ubo, sizeof(ubo));
    vkUnmapMemory(window->device, uniformBuffersMemory[swapChainImageIndex]);
}

void VulkanRenderer::startNextFrame(
    uint32_t swapChainImageIndex, VkFramebuffer framebuffer, VkCommandBuffer commandBuffer
) {
    cameraRight = glm::normalize(glm::cross(cameraForward, glm::vec3(0.0f, 1.0f, 0.0f)));
    cameraUp = glm::normalize(glm::cross(cameraForward, cameraRight));

    updateUniformBuffer(swapChainImageIndex);

    // Rasterization
    VkRenderPassBeginInfo renderPassInfo = {};
    renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    renderPassInfo.renderPass = window->renderPass;
    renderPassInfo.framebuffer = framebuffer;
    renderPassInfo.renderArea.extent.width = window->swapChainImageSize.width;
    renderPassInfo.renderArea.extent.height = window->swapChainImageSize.height;

    /*
    VkClearColorValue clearColor = {{ 0.0f, 0.0f, 0.0f, 1.0f }};
    VkClearDepthStencilValue clearDS = { 1, 0 };
    VkClearValue clearValues[3] = {};
    clearValues[0].color = clearValues[2].color = clearColor;
    clearValues[1].depthStencil = clearDS;
    renderPassInfo.clearValueCount = VK_SAMPLE_COUNT_1_BIT > VK_SAMPLE_COUNT_1_BIT ? 3 : 2;
    renderPassInfo.pClearValues = clearValues;
    */

    std::array<VkClearValue, 2> clearValues{};
    clearValues[0].color = {{ 0.0f, 0.0f, 0.0f, 1.0f }};
    clearValues[1].depthStencil = { 1.0f, 0 };
    renderPassInfo.clearValueCount = static_cast<uint32_t>(clearValues.size());
    renderPassInfo.pClearValues = clearValues.data();

    vkCmdBeginRenderPass(commandBuffer, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);

    // Define viewport
    VkViewport viewport = {};
    viewport.x = 0.0f;
    viewport.y = 0.0f;
    viewport.width = (float) window->swapChainImageSize.width;
    viewport.height = (float) window->swapChainImageSize.height;
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;
    vkCmdSetViewport(commandBuffer, 0, 1, &viewport);

    VkRect2D scissor = {};
    scissor.offset = { 0, 0 };
    scissor.extent.width = viewport.width;
    scissor.extent.height = viewport.height;
    vkCmdSetScissor(commandBuffer, 0, 1, &scissor);

    vkCmdBindDescriptorSets(
        commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 1,
        &descriptorSets[swapChainImageIndex], 0, nullptr
    );

    // Draw scene
    VkDeviceSize offsets[] = { 0 };
    vkCmdBindVertexBuffers(commandBuffer, 0, 1, &shadedVertexBuffer, offsets);
    vkCmdPushConstants(
        commandBuffer, pipelineLayout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(glm::mat4),
        (std::array<glm::mat4, 1> { glm::mat4(1.0f) }).data()
    );
    vkCmdPushConstants(
        commandBuffer, pipelineLayout, VK_SHADER_STAGE_FRAGMENT_BIT, sizeof(glm::mat4), sizeof(VkBool32),
        (std::array<VkBool32, 1> { shadedRendering }).data()
    );
    vkCmdDraw(commandBuffer, static_cast<uint32_t>(shadedPVS[currentViewCellIndex].size()), 1, 0, 0);

    // Draw view cell
    if (viewCellRendering) {
        vkCmdPushConstants(
            commandBuffer, pipelineLayout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(glm::mat4),
            (std::array<glm::mat4, 1> { visibilityManager->viewCells[currentViewCellIndex].model }).data()
        );
        vkCmdBindVertexBuffers(commandBuffer, 0, 1, &viewCellGeometry[currentViewCellIndex].vertexBuffer, offsets);
        vkCmdDraw(commandBuffer, 36, 1, 0, 0);
    }

    // Draw visibility cubes
    if (USE_NIRENSTEIN_VISIBILITY_SAMPLING && viewCellRendering) {
        for (auto pos : nirensteinSampler->renderCubePositions) {
            glm::mat4 model = glm::mat4(1.0f);
            model = glm::translate(model, pos);
            model = glm::scale(model, glm::vec3(2.0f));

            vkCmdPushConstants(
                commandBuffer, pipelineLayout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(glm::mat4),
                (std::array<glm::mat4, 1> { model }).data()
            );
            vkCmdBindVertexBuffers(commandBuffer, 0, 1, &viewCellGeometry[currentViewCellIndex].vertexBuffer, offsets);
            vkCmdDraw(commandBuffer, 36, 1, 0, 0);
        }
    }

    // Draw ray visualizations
    if (
        visibilityManager->visualizeRandomRays || visibilityManager->visualizeABSRays
        || visibilityManager->visualizeEdgeSubdivRays
    ) {
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, rayVisualizationPipeline);
        vkCmdBindDescriptorSets(
            commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, rayVisualizationPipelineLayout, 0, 1,
            &descriptorSets[swapChainImageIndex], 0, nullptr
        );
        vkCmdBindVertexBuffers(commandBuffer, 0, 1, &rayVertexBuffer, offsets);
        /*
        vkCmdPushConstants(
            commandBuffer, rayVisualizationPipelineLayout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(glm::mat4),
            (std::array<glm::mat4, 1> { glm::mat4(1.0f) }).data()
        );
        vkCmdPushConstants(
            commandBuffer, rayVisualizationPipelineLayout, VK_SHADER_STAGE_FRAGMENT_BIT, sizeof(glm::mat4), sizeof(VkBool32),
            (std::array<VkBool32, 1> { shadedRendering }).data()
        );
        */
        vkCmdDraw(commandBuffer, static_cast<uint32_t>(visibilityManager->rayVertices[currentViewCellIndex].size()), 1, 0, 0);
    }

    vkCmdEndRenderPass(commandBuffer);
}

void VulkanRenderer::toggleShadedRendering() {
    shadedRendering = !shadedRendering;
}

void VulkanRenderer::toggleViewCellRendering() {
    viewCellRendering = !viewCellRendering;
}

void VulkanRenderer::toggleRayVisualization() {
    visibilityManager->visualizeRandomRays = !visibilityManager->visualizeRandomRays;
}

void VulkanRenderer::nextCorner() {
    glm::vec3 offset;
    offset.x = currentViewCellCornerView % 2 == 0 ? -1.0f : 1.0f;
    offset.y = int(currentViewCellCornerView / 2) % 2 == 0 ? -1.0f : 1.0f;
    offset.z = int(currentViewCellCornerView / 4) % 4 == 0 ? -1.0f : 1.0f;

    cameraPos = visibilityManager->viewCells[currentViewCellIndex].model * glm::vec4(offset, 1.0f);
    //std::cout << "camera position: " << glm::to_string(cameraPos) << std::endl;
    currentViewCellCornerView = (currentViewCellCornerView + 1) % 8;

}

void VulkanRenderer::nextViewCell() {
    currentViewCellIndex++;
    currentViewCellIndex %= visibilityManager->viewCells.size();
    currentViewCellCornerView = 0;
    updateVertexBuffer(shadedPVS[currentViewCellIndex], shadedVertexBuffer, shadedVertexBufferMemory);
    if (
        visibilityManager->visualizeRandomRays || visibilityManager->visualizeABSRays
        || visibilityManager->visualizeEdgeSubdivRays
    ) {
        updateVertexBuffer(visibilityManager->rayVertices[currentViewCellIndex], rayVertexBuffer, shadedVertexBufferMemory);
    }
    std::cout
        << "View cell " << currentViewCellIndex << ": "
        << pvsTriangleIDs[currentViewCellIndex].size() << "/" << int(indices.size() / 3.0f)
        << " triangles (" << (pvsTriangleIDs[currentViewCellIndex].size() / (indices.size() / 3.0f)) * 100.0f << "%)" << std::endl;
}

void VulkanRenderer::alignCameraWithViewCellNormal() {
    cameraForward = glm::cross(
        glm::normalize(glm::vec3(visibilityManager->viewCells[currentViewCellIndex].model[0])),
        glm::normalize(glm::vec3(visibilityManager->viewCells[currentViewCellIndex].model[1]))
    );
}

void VulkanRenderer::initVisibilityManager() {

}

std::vector<glm::mat4> VulkanRenderer::loadSceneFile(Settings settings) {
    std::vector<glm::mat4> viewCellMatrices;

    std::string scene = settings.modelName;
    int viewCellIndex = settings.viewCellIndex;

    int i = 0;
    int currentViewCell = 0;
    bool found = false;

    float x, y, z;
    glm::vec3 v[3];

    std::cout << std::endl << "========================================" << std::endl;

    int viewCellCounter = 0;
    std::string modelPath;
    std::ifstream file("scenes.txt");
    std::string line;
    while (std::getline(file, line)) {
        if (i == 3) {
            glm::vec3 pos = v[0];
            glm::vec3 size = v[1] * 0.5f;
            glm::vec3 rotation = glm::radians(v[2]);

            glm::mat4 model = glm::mat4(1.0f);
            model = glm::translate(model, pos);
            model = glm::rotate(model, rotation.y, glm::vec3(0.0f, 1.0f, 0.0f));
            model = glm::rotate(model, rotation.x, glm::vec3(1.0f, 0.0f, 0.0f));
            model = glm::translate(model, size * glm::vec3(1.0f, 1.0f, 1.0f)); // translate such that the bottom left corner is at the position read from scenes.txt
            model = glm::scale(model, size);

            std::cout << "View cell " << viewCellCounter++ << ":" << std::endl;
            std::cout << "    Position (bottom left corner): " << glm::to_string(pos) << std::endl;
            std::cout << "    Rotation around axes (radiant): " << glm::to_string(rotation) << std::endl;
            std::cout << "    Size: " << glm::to_string(size * 2.0f) << std::endl;
            std::cout << "    Model matrix: " << glm::to_string(model) << std::endl;

            viewCellMatrices.push_back(model);

            if (line.length() == 0) {
                break;
            } else {
                currentViewCell++;
                viewCellIndex++;
                i = 0;
            }
        }

        if (line.find(scene) != std::string::npos) {
            found = true;

            std::istringstream iss(line);
            iss >> modelPath >> modelPath;
        } else if (found) {
            if (line.length() == 0) {
                currentViewCell++;
            } else if (currentViewCell == viewCellIndex) {
                std::istringstream iss(line);
                iss >> x >> y >> z;
                v[i] = { x, y, z };

                i++;
            }
        }
    }

    loadModel(modelPath);

    std::cout << "Model: " << modelPath << " (" << int(indices.size() / 3.0f) << " triangles)" << std::endl;
    std::cout << "========================================" << std::endl << std::endl;

    return viewCellMatrices;
}

Settings VulkanRenderer::loadSettingsFile() {
    Settings settings;

    std::ifstream file("settings.txt");
    std::string line;

    bool readSettings = false;
    bool readSceneDefinition = false;
    bool readComment = false;
    while (std::getline(file, line)) {
        if (line.length() == 0) {
            continue;
        }

        if (line.rfind("/*", 0) == 0) {
            readComment = true;
        } else if (line.rfind("*/") == 0) {
            readComment = false;
        } else if (!readComment) {
            if (line.rfind("--- SCENE ---", 0) == 0) {
                readSettings = false;
                readSceneDefinition = true;
            } else if (line.rfind("--- SETTINGS ---", 0) == 0) {
                readSettings = true;
                readSceneDefinition = false;
            } else if (readSettings) {
                se[line.substr(0, line.find(" "))] = line.substr(line.find(" ") + 1, line.length());
            } else if (readSceneDefinition) {
                if (line.rfind("CALCPVS", 0) == 0) {
                    loadPVS = false;
                } else if (line.rfind("LOADPVS", 0) == 0) {
                    loadPVS = true;
                }
                pvsStorageFile = line.substr(line.find(" "));

                std::getline(file, line);
                settings.modelName = line;

                std::getline(file, line);
                settings.viewCellIndex = std::stoi(line);
            }
        }
    }

    std::ofstream shaderDefinesFile;
    shaderDefinesFile.open("shaders/rt/defines.glsl");
    shaderDefinesFile << "const float ABS_DELTA = " << se.at("ABS_DELTA") << ";\n";
    shaderDefinesFile << "const int ABS_NUM_SAMPLES_PER_EDGE = " << se.at("ABS_NUM_SAMPLES_PER_EDGE") << ";\n";
    shaderDefinesFile << "const int ABS_MAX_SUBDIVISION_STEPS = " << se.at("ABS_MAX_SUBDIVISION_STEPS") << ";\n";
    shaderDefinesFile << "const int REVERSE_SAMPLING_NUM_SAMPLES_ALONG_EDGE = " << se.at("REVERSE_SAMPLING_NUM_SAMPLES_ALONG_EDGE") << ";\n";
    shaderDefinesFile << "#define REVERSE_SAMPLING_METHOD " << se.at("REVERSE_SAMPLING_METHOD") << "\n";
    shaderDefinesFile << "#define SET_TYPE " << se.at("SET_TYPE") << "\n";
    if (se.at("USE_3D_VIEW_CELL") == "true") {
        shaderDefinesFile << "#define USE_3D_VIEW_CELL\n";
    }
    if (se.at("USE_RECURSIVE_EDGE_SUBDIVISION") == "true") {
        shaderDefinesFile << "#define USE_RECURSIVE_EDGE_SUBDIVISION\n";
    }
    if (se.at("NIRENSTEIN_USE_MULTI_VIEW_RENDERING") == "true") {
        shaderDefinesFile << "#define NIRENSTEIN_USE_MULTI_VIEW_RENDERING\n";
    }
    shaderDefinesFile.close();

    USE_NIRENSTEIN_VISIBILITY_SAMPLING = se.at("USE_NIRENSTEIN_VISIBILITY_SAMPLING") == "true";
    USE_NIRENSTEIN_MULTI_VIEW_RENDERING = se.at("NIRENSTEIN_USE_MULTI_VIEW_RENDERING") == "true";

    return settings;
}

void VulkanRenderer::startVisibilityThread() {
    // Calculate the PVS
    if (!loadPVS) {
        std::ofstream pvsFile;
        pvsFile.open(pvsStorageFile);
        for (int k = 0; k < visibilityManager->viewCells.size(); k++) {
            std::cout << "View cell " << k << ":" << std::endl;
            std::vector<int> pvs;
            if (USE_NIRENSTEIN_VISIBILITY_SAMPLING) {
                pvs = nirensteinSampler->run(
                    visibilityManager->viewCells[k], cameraForward,
                    visibilityManager->generateHaltonPoints2d<2>({2, 3}, 20)
                );
            } else {
                visibilityManager->rayTrace(indices, 0, k);
                // Fetch the PVS from the GPU
                visibilityManager->fetchPVS();
            }

            // Write view cell model matrix to the PVS file
            for (int x = 0; x < 4; x++) {
                for (int y = 0; y < 4; y++) {
                    pvsFile << visibilityManager->viewCells[k].model[x][y] << ";";
                }
            }
            pvsFile << "|";

            // Write pvs to the PVS file
            std::ostringstream oss;
            if (USE_NIRENSTEIN_VISIBILITY_SAMPLING) {
                std::copy(pvs.begin(), pvs.end(), std::ostream_iterator<int>(oss, ";"));
            } else {
                std::copy(visibilityManager->pvs.pvsVector.begin(), visibilityManager->pvs.pvsVector.end(), std::ostream_iterator<int>(oss, ";"));
            }
            pvsFile << oss.str() << "\n";
        }
        pvsFile.close();
    }

    // Color all vertices red
    for (int i = 0; i < indices.size(); i++) {
        vertices[indices[i]].color = glm::vec3(1.0f, 0.0f, 0.0f);
    }

    // Parse PVS file
    std::ifstream pvsFile(pvsStorageFile);
    std::string line;
    int viewCellIndex = 0;
    while (std::getline(pvsFile, line)) {
        // Read view cell data from the PVS file
        std::string viewCellString = line.substr(0, line.find("|"));
        std::stringstream stringStream(viewCellString);

        std::array<float, 16> viewCellData;
        int i = 0;
        for (float f; stringStream >> f; i++) {
            viewCellData[i] = f;
            if (stringStream.peek() == ';') {
                stringStream.ignore();
            }
        }

        ViewCell viewCell(glm::make_mat4(viewCellData.data()));

        shadedPVS.push_back({});
        for (int i = 0; i < indices.size(); i++) {
            shadedPVS[viewCellIndex].push_back(vertices[indices[i]]);
        }

        // Read the triangle IDs (PVS) from the PVS file. These triangles are colored green
        pvsTriangleIDs.push_back({});
        std::string pvsString = line.substr(line.find("|") + 1);
        std::stringstream ss(pvsString);
        for (int triangleID; ss >> triangleID;) {
            pvsTriangleIDs[viewCellIndex].push_back(triangleID);

            shadedPVS[viewCellIndex][3 * triangleID].color = glm::vec3(0.0f, 1.0f, 0.0f);
            shadedPVS[viewCellIndex][3 * triangleID + 1].color = glm::vec3(0.0f, 1.0f, 0.0f);
            shadedPVS[viewCellIndex][3 * triangleID + 2].color = glm::vec3(0.0f, 1.0f, 0.0f);

            if (ss.peek() == ';') {
                ss.ignore();
            }
        }

        {
            // Load a box model for the current view cell
            std::vector<Vertex> viewCellGeomtryVertices;

            tinyobj::attrib_t attrib;
            std::vector<tinyobj::shape_t> shapes;
            std::vector<tinyobj::material_t> materials;
            std::string warn, err;

            std::string viewCellModelPath = "models/box/box.obj";
            if (!tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, viewCellModelPath.c_str())) {
                throw std::runtime_error((warn + err).c_str());
            }

            for (const auto &shape : shapes) {
                for (const auto &index : shape.mesh.indices) {
                    Vertex vertex = {};
                    vertex.pos = {
                        attrib.vertices[3 * index.vertex_index + 0],
                        attrib.vertices[3 * index.vertex_index + 1],
                        attrib.vertices[3 * index.vertex_index + 2]
                    };
                    vertex.normal = {
                        attrib.normals[3 * index.normal_index + 0],
                        attrib.normals[3 * index.normal_index + 1],
                        attrib.normals[3 * index.normal_index + 2]
                    };
                    vertex.color = { 1.0f, 1.0f, 1.0f };

                    viewCellGeomtryVertices.push_back(vertex);
                }
            }

            VkBuffer vertexBuffer;
            VkDeviceMemory vertexBufferMemory;
            createVertexBuffer(viewCellGeomtryVertices, vertexBuffer, vertexBufferMemory);
            updateVertexBuffer(viewCellGeomtryVertices, vertexBuffer, vertexBufferMemory);
            viewCellGeometry.emplace_back(vertexBuffer, vertexBufferMemory);
        }

        viewCellIndex++;
    }
    pvsFile.close();

    currentViewCellIndex = 0;
    cameraPos = visibilityManager->viewCells[currentViewCellIndex].model[3];
    createVertexBuffer(shadedPVS[currentViewCellIndex], shadedVertexBuffer, shadedVertexBufferMemory);
    updateVertexBuffer(shadedPVS[currentViewCellIndex], shadedVertexBuffer, shadedVertexBufferMemory);

    // Create and fill ray visualization buffers
    if (
        visibilityManager->visualizeRandomRays || visibilityManager->visualizeABSRays
        || visibilityManager->visualizeEdgeSubdivRays
    ) {
        createVertexBuffer(visibilityManager->rayVertices[currentViewCellIndex], rayVertexBuffer, shadedVertexBufferMemory);
        updateVertexBuffer(visibilityManager->rayVertices[currentViewCellIndex], rayVertexBuffer, shadedVertexBufferMemory);
    }
}
