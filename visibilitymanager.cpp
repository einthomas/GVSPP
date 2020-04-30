#include <cstring>
#include <glm/gtx/string_cast.hpp>

#include "vulkanutil.h"
#include "visibilitymanager.h"
#include "viewcell.h"
#include "sample.h"
#include "Vertex.h"

struct UniformBufferObject {
    alignas(16) glm::mat4 model;
    alignas(16) glm::mat4 view;
    alignas(16) glm::mat4 projection;
};

VisibilityManager::VisibilityManager(int raysPerIteration)
    : MAX_ABS_TRIANGLES_PER_ITERATION(raysPerIteration), MAX_EDGE_SUBDIV_RAYS(raysPerIteration)
{
}

void VisibilityManager::init(
    int raysPerIteration, VkPhysicalDevice physicalDevice, VkDevice logicalDevice,
    VkCommandPool graphicsCommandPool, VkQueue graphicsQueue, VkBuffer indexBuffer,
    const std::vector<uint32_t> &indices, VkBuffer vertexBuffer,
    const std::vector<Vertex> &vertices, const std::vector<VkBuffer> &uniformBuffers,
    uint32_t deviceLocalMemoryIndex
) {
    this->logicalDevice = logicalDevice;
    this->graphicsCommandPool = graphicsCommandPool;
    this->graphicsQueue = graphicsQueue;
    this->physicalDevice = physicalDevice;

    this->raysPerIteration = raysPerIteration;

    generateHaltonPoints(raysPerIteration);
    createHaltonPointsBuffer();
    createViewCellBuffer();
    createBuffers(indices);
    initRayTracing(
        indexBuffer, vertexBuffer, indices, vertices, uniformBuffers, deviceLocalMemoryIndex
    );
}

void VisibilityManager::addViewCell(glm::vec3 pos, glm::vec2 size, glm::vec3 normal) {
    viewCells.push_back(ViewCell(pos, size, normal));
}

/*
 * From "Sampling with Hammersley and Halton Points" (Wong et al. 1997)
 */
void VisibilityManager::generateHaltonPoints(int n, int p2) {
    haltonPoints.resize(n);

    float p, u, v, ip;
    int k, kk, pos, a;
    for (k = 0, pos = 0; k < n; k++) {
        u = 0;
        for (p = 0.5, kk = k; kk; p *= 0.5, kk >>= 1) {
            if (kk & 1) {
                u += p;
            }
        }

        v = 0;
        ip = 1.0 / p2;
        for (p = ip, kk = k; kk; p *= ip, kk /= p2) {
            if ((a = kk % p2)) {
                v += a * p;
            }
        }

        haltonPoints[pos].x = u;
        haltonPoints[pos].y = v;
        pos++;
    }
}

void VisibilityManager::createHaltonPointsBuffer() {
    VkDeviceSize bufferSize = sizeof(haltonPoints[0]) * haltonPoints.size();

    // Create staging buffer using host-visible memory
    VkBuffer stagingBuffer;
    VkDeviceMemory stagingBufferMemory;
    VulkanUtil::createBuffer(
        physicalDevice, logicalDevice, bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, stagingBuffer, stagingBufferMemory,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
    );

    // Copy halton points to the staging buffer
    void *data;
    vkMapMemory(logicalDevice, stagingBufferMemory, 0, bufferSize, 0, &data);    // Map buffer memory into CPU accessible memory
    memcpy(data, haltonPoints.data(), (size_t) bufferSize);  // Copy vertex data to mapped memory
    vkUnmapMemory(logicalDevice, stagingBufferMemory);

    // Create halton points buffer using GPU memory
    VulkanUtil::createBuffer(
        physicalDevice, logicalDevice, bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_RAY_TRACING_BIT_NV | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        haltonPointsBuffer, haltonPointsBufferMemory, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
    );

    // Copy halton points from the staging buffer to the halton points buffer
    VulkanUtil::copyBuffer(logicalDevice, graphicsCommandPool, graphicsQueue, stagingBuffer, haltonPointsBuffer, bufferSize);

    vkDestroyBuffer(logicalDevice, stagingBuffer, nullptr);
    vkFreeMemory(logicalDevice, stagingBufferMemory, nullptr);
}

void VisibilityManager::createViewCellBuffer() {
    VkDeviceSize bufferSize = sizeof(viewCells[0]) * viewCells.size();

    // Create staging buffer using host-visible memory
    VkBuffer stagingBuffer;
    VkDeviceMemory stagingBufferMemory;
    VulkanUtil::createBuffer(
        physicalDevice,
        logicalDevice, bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, stagingBuffer, stagingBufferMemory,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
    );

    // Copy halton points to the staging buffer
    void *data;
    vkMapMemory(logicalDevice, stagingBufferMemory, 0, bufferSize, 0, &data);    // Map buffer memory into CPU accessible memory
    memcpy(data, viewCells.data(), (size_t) bufferSize);  // Copy vertex data to mapped memory
    vkUnmapMemory(logicalDevice, stagingBufferMemory);

    // Create halton points buffer using GPU memory
    VulkanUtil::createBuffer(
        physicalDevice,
        logicalDevice, bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_RAY_TRACING_BIT_NV | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
        viewCellBuffer, viewCellBufferMemory, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
    );

    // Copy halton points from the staging buffer to the halton points buffer
    VulkanUtil::copyBuffer(
        logicalDevice, graphicsCommandPool, graphicsQueue, stagingBuffer, viewCellBuffer, bufferSize
    );

    vkDestroyBuffer(logicalDevice, stagingBuffer, nullptr);
    vkFreeMemory(logicalDevice, stagingBufferMemory, nullptr);
}

void VisibilityManager::createBuffers(const std::vector<uint32_t> &indices) {
    VulkanUtil::createBuffer(
        physicalDevice,
        logicalDevice, sizeof(unsigned int) * raysPerIteration * 3,
        VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_RAY_TRACING_BIT_NV | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        intersectedTrianglesBuffer, intersectedTrianglesBufferMemory, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
    );

    VulkanUtil::createBuffer(
        physicalDevice,
        logicalDevice, sizeof(Sample) * raysPerIteration,
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_RAY_TRACING_BIT_NV | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        rayOriginBuffer, rayOriginBufferMemory, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
    );

    VulkanUtil::createBuffer(
        physicalDevice,
        logicalDevice, sizeof(Sample) * MAX_ABS_TRIANGLES_PER_ITERATION * 9 * 2,
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_RAY_TRACING_BIT_NV | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        absOutputBuffer, absOutputBufferMemory, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
    );

    VulkanUtil::createBuffer(
        physicalDevice,
        logicalDevice, sizeof(indices[0]) * indices.size(),
        VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
        pvsVisualizationBuffer, pvsVisualizationBufferMemory, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
    );

    VulkanUtil::createBuffer(
        physicalDevice,
        logicalDevice, sizeof(Sample) * MAX_ABS_TRIANGLES_PER_ITERATION,
        VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_RAY_TRACING_BIT_NV | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        absWorkingBuffer, absWorkingBufferMemory, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
    );

    VulkanUtil::createBuffer(
        physicalDevice,
        logicalDevice, sizeof(Sample) * MAX_EDGE_SUBDIV_RAYS,
        VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_RAY_TRACING_BIT_NV | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        edgeSubdivWorkingBuffer, edgeSubdivWorkingBufferMemory, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
    );
}

void VisibilityManager::createDescriptorSets(
    VkBuffer indexBuffer, VkBuffer vertexBuffer, const std::vector<VkBuffer> &uniformBuffers
) {
    /*
    VkDescriptorSetAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = rtDescriptorPool;
    allocInfo.descriptorSetCount = 1;
    allocInfo.pSetLayouts = &rtDescriptorSetLayout;

    // Allocate descriptor sets
    if (vkAllocateDescriptorSets(
            logicalDevice, &allocInfo, &rtDescriptorSets
        ) != VK_SUCCESS
    ) {
        throw std::runtime_error("failed to allocate rt descriptor sets");
    }
    */

    std::array<VkWriteDescriptorSet, 8> descriptorWrites = {};

    VkWriteDescriptorSetAccelerationStructureNV asWriteInfo = {};
    asWriteInfo.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_NV;
    asWriteInfo.accelerationStructureCount = 1;
    asWriteInfo.pAccelerationStructures = &topLevelAS.as;
    descriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptorWrites[0].pNext = &asWriteInfo;
    descriptorWrites[0].dstSet = descriptorSet;
    descriptorWrites[0].dstBinding = 0;
    descriptorWrites[0].descriptorCount = 1;
    descriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_NV;

    VkDescriptorBufferInfo uniformBufferInfo = {};
    uniformBufferInfo.buffer = uniformBuffers[0];
    uniformBufferInfo.offset = 0;
    uniformBufferInfo.range = sizeof(UniformBufferObject);
    descriptorWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptorWrites[1].dstSet = descriptorSet;
    descriptorWrites[1].dstBinding = 2;
    descriptorWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    descriptorWrites[1].descriptorCount = 1;
    descriptorWrites[1].pBufferInfo = &uniformBufferInfo;

    VkDescriptorBufferInfo indexBufferInfo = {};
    indexBufferInfo.buffer = indexBuffer;
    indexBufferInfo.offset = 0;
    indexBufferInfo.range = VK_WHOLE_SIZE;
    descriptorWrites[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptorWrites[2].dstSet = descriptorSet;
    descriptorWrites[2].dstBinding = 4;
    descriptorWrites[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    descriptorWrites[2].descriptorCount = 1;
    descriptorWrites[2].pBufferInfo = &indexBufferInfo;

    VkDescriptorBufferInfo haltonPointsBufferInfo = {};
    haltonPointsBufferInfo.buffer = haltonPointsBuffer;
    haltonPointsBufferInfo.offset = 0;
    haltonPointsBufferInfo.range = VK_WHOLE_SIZE;
    descriptorWrites[3].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptorWrites[3].dstSet = descriptorSet;
    descriptorWrites[3].dstBinding = 5;
    descriptorWrites[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    descriptorWrites[3].descriptorCount = 1;
    descriptorWrites[3].pBufferInfo = &haltonPointsBufferInfo;

    VkDescriptorBufferInfo viewCellBufferInfo = {};
    viewCellBufferInfo.buffer = viewCellBuffer;
    viewCellBufferInfo.offset = 0;
    viewCellBufferInfo.range = VK_WHOLE_SIZE;
    descriptorWrites[4].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptorWrites[4].dstSet = descriptorSet;
    descriptorWrites[4].dstBinding = 6;
    descriptorWrites[4].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    descriptorWrites[4].descriptorCount = 1;
    descriptorWrites[4].pBufferInfo = &viewCellBufferInfo;

    VkDescriptorBufferInfo pvsBufferInfo = {};
    pvsBufferInfo.buffer = intersectedTrianglesBuffer;
    pvsBufferInfo.offset = 0;
    pvsBufferInfo.range = VK_WHOLE_SIZE;
    descriptorWrites[5].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptorWrites[5].dstSet = descriptorSet;
    descriptorWrites[5].dstBinding = 7;
    descriptorWrites[5].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    descriptorWrites[5].descriptorCount = 1;
    descriptorWrites[5].pBufferInfo = &pvsBufferInfo;

    VkDescriptorBufferInfo rayOriginBufferInfo = {};
    rayOriginBufferInfo.buffer = rayOriginBuffer;
    rayOriginBufferInfo.offset = 0;
    rayOriginBufferInfo.range = VK_WHOLE_SIZE;
    descriptorWrites[6].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptorWrites[6].dstSet = descriptorSet;
    descriptorWrites[6].dstBinding = 8;
    descriptorWrites[6].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    descriptorWrites[6].descriptorCount = 1;
    descriptorWrites[6].pBufferInfo = &rayOriginBufferInfo;

    VkDescriptorBufferInfo vertexBufferInfo = {};
    vertexBufferInfo.buffer = vertexBuffer;
    vertexBufferInfo.offset = 0;
    vertexBufferInfo.range = VK_WHOLE_SIZE;
    descriptorWrites[7].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptorWrites[7].dstSet = descriptorSet;
    descriptorWrites[7].dstBinding = 3;
    descriptorWrites[7].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    descriptorWrites[7].descriptorCount = 1;
    descriptorWrites[7].pBufferInfo = &vertexBufferInfo;

    vkUpdateDescriptorSets(
        logicalDevice,
        static_cast<uint32_t>(descriptorWrites.size()),
        descriptorWrites.data(),
        0,
        VK_NULL_HANDLE
    );
}

void VisibilityManager::initRayTracing(
    VkBuffer indexBuffer, VkBuffer vertexBuffer, const std::vector<uint32_t> &indices,
    const std::vector<Vertex> &vertices, const std::vector<VkBuffer> &uniformBuffers,
    uint32_t deviceLocalMemoryIndex
) {
    rayTracingProperties = {};
    rayTracingProperties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PROPERTIES_NV;

    VkPhysicalDeviceProperties2 deviceProperties = {};
    deviceProperties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    deviceProperties.pNext = &rayTracingProperties;
    vkGetPhysicalDeviceProperties2(physicalDevice, &deviceProperties);
    //qDebug() << deviceProperties.properties.limits.maxComputeWorkGroupCount[2];

    // Get function pointers
    vkCreateAccelerationStructureNV = reinterpret_cast<PFN_vkCreateAccelerationStructureNV>(vkGetDeviceProcAddr(logicalDevice, "vkCreateAccelerationStructureNV"));
    vkDestroyAccelerationStructureNV = reinterpret_cast<PFN_vkDestroyAccelerationStructureNV>(vkGetDeviceProcAddr(logicalDevice, "vkDestroyAccelerationStructureNV"));
    vkBindAccelerationStructureMemoryNV = reinterpret_cast<PFN_vkBindAccelerationStructureMemoryNV>(vkGetDeviceProcAddr(logicalDevice, "vkBindAccelerationStructureMemoryNV"));
    vkGetAccelerationStructureHandleNV = reinterpret_cast<PFN_vkGetAccelerationStructureHandleNV>(vkGetDeviceProcAddr(logicalDevice, "vkGetAccelerationStructureHandleNV"));
    vkGetAccelerationStructureMemoryRequirementsNV = reinterpret_cast<PFN_vkGetAccelerationStructureMemoryRequirementsNV>(vkGetDeviceProcAddr(logicalDevice, "vkGetAccelerationStructureMemoryRequirementsNV"));
    vkCmdBuildAccelerationStructureNV = reinterpret_cast<PFN_vkCmdBuildAccelerationStructureNV>(vkGetDeviceProcAddr(logicalDevice, "vkCmdBuildAccelerationStructureNV"));
    vkCreateRayTracingPipelinesNV = reinterpret_cast<PFN_vkCreateRayTracingPipelinesNV>(vkGetDeviceProcAddr(logicalDevice, "vkCreateRayTracingPipelinesNV"));
    vkGetRayTracingShaderGroupHandlesNV = reinterpret_cast<PFN_vkGetRayTracingShaderGroupHandlesNV>(vkGetDeviceProcAddr(logicalDevice, "vkGetRayTracingShaderGroupHandlesNV"));
    vkCmdTraceRaysNV = reinterpret_cast<PFN_vkCmdTraceRaysNV>(vkGetDeviceProcAddr(logicalDevice, "vkCmdTraceRaysNV"));

    VkGeometryNV geometry = {};
    geometry.sType = VK_STRUCTURE_TYPE_GEOMETRY_NV;
    geometry.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_NV;
    geometry.geometry.triangles.sType = VK_STRUCTURE_TYPE_GEOMETRY_TRIANGLES_NV;
    geometry.geometry.triangles.vertexData = vertexBuffer;
    geometry.geometry.triangles.vertexOffset = 0;
    geometry.geometry.triangles.vertexCount = vertices.size();
    geometry.geometry.triangles.vertexStride = sizeof(Vertex);
    geometry.geometry.triangles.vertexFormat =VK_FORMAT_R32G32B32_SFLOAT;
    geometry.geometry.triangles.indexData = indexBuffer;
    geometry.geometry.triangles.indexOffset = 0;
    geometry.geometry.triangles.indexCount = indices.size();
    geometry.geometry.triangles.indexType = VK_INDEX_TYPE_UINT32;
    geometry.geometry.triangles.transformData = VK_NULL_HANDLE;
    geometry.geometry.triangles.transformOffset = 0;
    geometry.geometry.aabbs = {};
    geometry.geometry.aabbs.sType = { VK_STRUCTURE_TYPE_GEOMETRY_AABB_NV };
    geometry.flags = VK_GEOMETRY_OPAQUE_BIT_NV;

    createBottomLevelAS(&geometry, deviceLocalMemoryIndex);

    VkBuffer instanceBuffer;
    VkDeviceMemory instanceBufferMemory;

    /*
    glm::mat4x4 m = glm::rotate(
        glm::mat4(1.0f),
        glm::radians(0.0f),
        glm::vec3(0.0f, 1.0f, 0.0f)
    );
    */

    glm::mat4x4 model = glm::translate(
        glm::mat4(1.0f),
        glm::vec3(0.0f, 0.0f, 0.0f) * 0.5f
    );
    model = glm::transpose(model);

    glm::mat3x4 transform = {
        1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f, 0.0f,
    };
    memcpy(&transform, &model, sizeof(transform));

    GeometryInstance geometryInstance = {};
    geometryInstance.transform = transform;
    geometryInstance.instanceId = 0;
    geometryInstance.mask = 0xff;
    geometryInstance.instanceOffset = 0;
    geometryInstance.flags = VK_GEOMETRY_INSTANCE_TRIANGLE_CULL_DISABLE_BIT_NV;
    geometryInstance.accelerationStructureHandle = bottomLevelAS.handle;

    // Upload instance descriptions to the device
    VkDeviceSize instanceBufferSize = sizeof(GeometryInstance);
    VulkanUtil::createBuffer(
        physicalDevice,
        logicalDevice, instanceBufferSize,
        VK_BUFFER_USAGE_RAY_TRACING_BIT_NV,
        instanceBuffer,
        instanceBufferMemory,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
    );
    void *data;
    vkMapMemory(logicalDevice, instanceBufferMemory, 0, instanceBufferSize, 0, &data);
    memcpy(data, &geometryInstance, instanceBufferSize);
    vkUnmapMemory(logicalDevice, instanceBufferMemory);

    createTopLevelAS(deviceLocalMemoryIndex);

    // Build acceleration structures
    buildAS(instanceBuffer, &geometry);

    vkDestroyBuffer(logicalDevice, instanceBuffer, nullptr);
    vkFreeMemory(logicalDevice, instanceBufferMemory, nullptr);

    createCommandBuffers();
    createDescriptorPool();
    createDescriptorSetLayout();
    createABSDescriptorSetLayout();
    createEdgeSubdivDescriptorSetLayout();

    {   // TODO: Cleanup
        std::array<VkDescriptorSetLayout, 3> d = {
            descriptorSetLayout, descriptorSetLayoutABS, descriptorSetLayoutEdgeSubdiv
        };
        VkDescriptorSetAllocateInfo allocInfo = {};
        allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocInfo.descriptorPool = descriptorPool;
        allocInfo.descriptorSetCount = 3;
        allocInfo.pSetLayouts = d.data();

        // Allocate descriptor sets
        std::array<VkDescriptorSet, 3> dd = {
            descriptorSet, descriptorSetABS, descriptorSetEdgeSubdiv
        };
        if (vkAllocateDescriptorSets(
                logicalDevice, &allocInfo, dd.data()
            ) != VK_SUCCESS
        ) {
            throw std::runtime_error("failed to allocate rt descriptor sets ABS");
        }
        descriptorSet = dd[0];
        descriptorSetABS = dd[1];
        descriptorSetEdgeSubdiv = dd[2];
    }

    createDescriptorSets(indexBuffer, vertexBuffer, uniformBuffers);
    createRandomSamplingPipeline();
    createShaderBindingTable(shaderBindingTable, shaderBindingTableMemory, pipeline);

    createABSDescriptorSets();
    createABSPipeline();
    createShaderBindingTable(shaderBindingTableABS, shaderBindingTableMemoryABS, pipelineABS);

    createEdgeSubdivDescriptorSets();
    createEdgeSubdivPipeline();
    createShaderBindingTable(shaderBindingTableEdgeSubdiv, shaderBindingTableMemoryEdgeSubdiv, pipelineEdgeSubdiv);

    // Calculate shader binding offsets
    bindingOffsetRayGenShader = rayTracingProperties.shaderGroupHandleSize * RT_SHADER_INDEX_RAYGEN;
    bindingOffsetMissShader = rayTracingProperties.shaderGroupHandleSize * RT_SHADER_INDEX_MISS;
    bindingOffsetHitShader = rayTracingProperties.shaderGroupHandleSize * RT_SHADER_INDEX_CLOSEST_HIT;
    bindingStride = rayTracingProperties.shaderGroupHandleSize;
}

void VisibilityManager::createBottomLevelAS(
    const VkGeometryNV *geometry, uint32_t deviceLocalMemoryIndex
) {
    // The bottom level acceleration structure contains the scene's geometry

    VkAccelerationStructureInfoNV accelerationStructureInfo = {};
    accelerationStructureInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_INFO_NV;
    accelerationStructureInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_NV;
    accelerationStructureInfo.instanceCount = 0;
    accelerationStructureInfo.geometryCount = 1;
    accelerationStructureInfo.pGeometries = geometry;
    //accelerationStructureInfo.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_NV;

    VkAccelerationStructureCreateInfoNV accelerationStructureCI = {};
    accelerationStructureCI.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_NV;
    accelerationStructureCI.info = accelerationStructureInfo;
    vkCreateAccelerationStructureNV(logicalDevice, &accelerationStructureCI, nullptr, &bottomLevelAS.as);

    VkAccelerationStructureMemoryRequirementsInfoNV memoryRequirementsInfo = {};
    memoryRequirementsInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_MEMORY_REQUIREMENTS_INFO_NV;
    memoryRequirementsInfo.type = VK_ACCELERATION_STRUCTURE_MEMORY_REQUIREMENTS_TYPE_OBJECT_NV;
    memoryRequirementsInfo.accelerationStructure = bottomLevelAS.as;

    VkMemoryRequirements2 memoryRequirements2 = {};
    vkGetAccelerationStructureMemoryRequirementsNV(logicalDevice, &memoryRequirementsInfo, &memoryRequirements2);

    VkMemoryAllocateInfo memoryAllocateInfo = {};
    memoryAllocateInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    memoryAllocateInfo.allocationSize = memoryRequirements2.memoryRequirements.size;
    memoryAllocateInfo.memoryTypeIndex = deviceLocalMemoryIndex;
    vkAllocateMemory(logicalDevice, &memoryAllocateInfo, nullptr, &bottomLevelAS.deviceMemory);

    VkBindAccelerationStructureMemoryInfoNV accelerationStructureMemoryInfo = {};
    accelerationStructureMemoryInfo.sType = VK_STRUCTURE_TYPE_BIND_ACCELERATION_STRUCTURE_MEMORY_INFO_NV;
    accelerationStructureMemoryInfo.accelerationStructure = bottomLevelAS.as;
    accelerationStructureMemoryInfo.memory = bottomLevelAS.deviceMemory;
    vkBindAccelerationStructureMemoryNV(logicalDevice, 1, &accelerationStructureMemoryInfo);

    vkGetAccelerationStructureHandleNV(logicalDevice, bottomLevelAS.as, sizeof(uint64_t), &bottomLevelAS.handle);
}

void VisibilityManager::createTopLevelAS(uint32_t deviceLocalMemoryIndex) {
    VkAccelerationStructureInfoNV accelerationStructureInfo = {};
    accelerationStructureInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_INFO_NV;
    accelerationStructureInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_NV;
    accelerationStructureInfo.instanceCount = 1;
    accelerationStructureInfo.geometryCount = 0;
    //accelerationStructureInfo.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_NV;

    VkAccelerationStructureCreateInfoNV accelerationStructureCI = {};
    accelerationStructureCI.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_NV;
    accelerationStructureCI.info = accelerationStructureInfo;
    vkCreateAccelerationStructureNV(logicalDevice, &accelerationStructureCI, nullptr, &topLevelAS.as);

    VkAccelerationStructureMemoryRequirementsInfoNV memoryRequirementsInfo = {};
    memoryRequirementsInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_MEMORY_REQUIREMENTS_INFO_NV;
    memoryRequirementsInfo.type = VK_ACCELERATION_STRUCTURE_MEMORY_REQUIREMENTS_TYPE_OBJECT_NV;
    memoryRequirementsInfo.accelerationStructure = topLevelAS.as;

    VkMemoryRequirements2 memoryRequirements2 = {};
    vkGetAccelerationStructureMemoryRequirementsNV(logicalDevice, &memoryRequirementsInfo, &memoryRequirements2);

    VkMemoryAllocateInfo memoryAllocateInfo = {};
    memoryAllocateInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    memoryAllocateInfo.allocationSize = memoryRequirements2.memoryRequirements.size;
    memoryAllocateInfo.memoryTypeIndex = deviceLocalMemoryIndex;
    vkAllocateMemory(logicalDevice, &memoryAllocateInfo, nullptr, &topLevelAS.deviceMemory);

    VkBindAccelerationStructureMemoryInfoNV accelerationStructureMemoryInfo = {};
    accelerationStructureMemoryInfo.sType = VK_STRUCTURE_TYPE_BIND_ACCELERATION_STRUCTURE_MEMORY_INFO_NV;
    accelerationStructureMemoryInfo.accelerationStructure = topLevelAS.as;
    accelerationStructureMemoryInfo.memory = topLevelAS.deviceMemory;
    vkBindAccelerationStructureMemoryNV(logicalDevice, 1, &accelerationStructureMemoryInfo);

    vkGetAccelerationStructureHandleNV(logicalDevice, topLevelAS.as, sizeof(uint64_t), &topLevelAS.handle);
}

void VisibilityManager::buildAS(const VkBuffer instanceBuffer, const VkGeometryNV *geometry) {
    // Acceleration structure build requires some scratch space to store temporary information
    VkAccelerationStructureMemoryRequirementsInfoNV memoryRequirementsInfo = {};
    memoryRequirementsInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_MEMORY_REQUIREMENTS_INFO_NV;
    memoryRequirementsInfo.type = VK_ACCELERATION_STRUCTURE_MEMORY_REQUIREMENTS_TYPE_BUILD_SCRATCH_NV;

    // Memory requirement for the bottom level acceleration structure
    VkMemoryRequirements2 memReqBottomLevelAS;
    memoryRequirementsInfo.accelerationStructure = bottomLevelAS.as;
    vkGetAccelerationStructureMemoryRequirementsNV(logicalDevice, &memoryRequirementsInfo, &memReqBottomLevelAS);

    // Memory requirement for the top level acceleration structure
    VkMemoryRequirements2 memReqTopLevelAS;
    memoryRequirementsInfo.accelerationStructure = topLevelAS.as;
    vkGetAccelerationStructureMemoryRequirementsNV(logicalDevice, &memoryRequirementsInfo, &memReqTopLevelAS);

    // Create temporary buffer
    const VkDeviceSize tempBufferSize = std::max(memReqBottomLevelAS.memoryRequirements.size, memReqTopLevelAS.memoryRequirements.size);
    VkBuffer tempBuffer = {};
    VkDeviceMemory tempBufferMemory = {};
    VulkanUtil::createBuffer(
        physicalDevice,
        logicalDevice, tempBufferSize,
        VK_BUFFER_USAGE_RAY_TRACING_BIT_NV,
        tempBuffer,
        tempBufferMemory,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
    );

    VkCommandBuffer commandBuffer = VulkanUtil::beginSingleTimeCommands(logicalDevice, graphicsCommandPool);

    // Build bottom level acceleration structure
    VkAccelerationStructureInfoNV buildInfo = {};
    buildInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_INFO_NV;
    buildInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_NV;
    buildInfo.geometryCount = 1;
    buildInfo.pGeometries = geometry;
    vkCmdBuildAccelerationStructureNV(
        commandBuffer,
        &buildInfo,
        VK_NULL_HANDLE,
        0,
        VK_FALSE,
        bottomLevelAS.as,
        VK_NULL_HANDLE,
        tempBuffer,
        0
    );
    VkMemoryBarrier memoryBarrier = {};
    memoryBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    memoryBarrier.srcAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_NV | VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_NV;
    memoryBarrier.dstAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_NV | VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_NV;
    vkCmdPipelineBarrier(
        commandBuffer,
        VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_NV,
        VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_NV,
        0,
        1,
        &memoryBarrier,
        0,
        0,
        0,
        0
    );

    // Build top level acceleration structure
    buildInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_NV;
    buildInfo.pGeometries = 0;
    buildInfo.geometryCount = 0;
    buildInfo.instanceCount = 1;
    vkCmdBuildAccelerationStructureNV(
        commandBuffer,
        &buildInfo,
        instanceBuffer,
        0,
        VK_FALSE,
        topLevelAS.as,
        VK_NULL_HANDLE,
        tempBuffer,
        0
    );
    vkCmdPipelineBarrier(
        commandBuffer,
        VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_NV,
        VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_NV,
        0,
        1,
        &memoryBarrier,
        0,
        0,
        0,
        0
    );

    VulkanUtil::endSingleTimeCommands(
        logicalDevice, commandBuffer, graphicsCommandPool, graphicsQueue
    );

    vkDestroyBuffer(logicalDevice, tempBuffer, nullptr);
    vkFreeMemory(logicalDevice, tempBufferMemory, nullptr);
}

void VisibilityManager::createDescriptorSetLayout() {
    // Top level acceleration structure binding
    VkDescriptorSetLayoutBinding aslayoutBinding = {};
    aslayoutBinding.binding = 0;
    aslayoutBinding.descriptorCount = 1;
    aslayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_NV;
    aslayoutBinding.stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_NV;

    // Uniform binding
    VkDescriptorSetLayoutBinding uniformLayoutBinding = {};
    uniformLayoutBinding.binding = 2;
    uniformLayoutBinding.descriptorCount = 1;
    uniformLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    uniformLayoutBinding.stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_NV | VK_SHADER_STAGE_CLOSEST_HIT_BIT_NV;

    // Vertex array binding
    VkDescriptorSetLayoutBinding vertexLayoutBinding = {};
    vertexLayoutBinding.binding = 3;
    vertexLayoutBinding.descriptorCount = 1;
    vertexLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    vertexLayoutBinding.stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_NV | VK_SHADER_STAGE_CLOSEST_HIT_BIT_NV;

    // Index array binding
    VkDescriptorSetLayoutBinding indexLayoutBinding = {};
    indexLayoutBinding.binding = 4;
    indexLayoutBinding.descriptorCount = 1;
    indexLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    indexLayoutBinding.stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_NV | VK_SHADER_STAGE_CLOSEST_HIT_BIT_NV;

    // Halton points binding
    VkDescriptorSetLayoutBinding haltonPointsBinding = {};
    haltonPointsBinding.binding = 5;
    haltonPointsBinding.descriptorCount = 1;
    haltonPointsBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    haltonPointsBinding.stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_NV;

    // View cell uniform binding
    VkDescriptorSetLayoutBinding viewCellBinding = {};
    viewCellBinding.binding = 6;
    viewCellBinding.descriptorCount = 1;
    viewCellBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    viewCellBinding.stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_NV;

    // Triangle output buffer binding
    VkDescriptorSetLayoutBinding pvsBinding = {};
    pvsBinding.binding = 7;
    pvsBinding.descriptorCount = 1;
    pvsBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    pvsBinding.stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_NV;

    // Ray origin output buffer binding
    VkDescriptorSetLayoutBinding rayOriginBinding = {};
    rayOriginBinding.binding = 8;
    rayOriginBinding.descriptorCount = 1;
    rayOriginBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    rayOriginBinding.stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_NV;

    std::array<VkDescriptorSetLayoutBinding, 8> bindings = {
        aslayoutBinding,
        uniformLayoutBinding,
        vertexLayoutBinding,
        indexLayoutBinding,
        haltonPointsBinding,
        viewCellBinding,
        pvsBinding,
        rayOriginBinding
    };
    VkDescriptorSetLayoutCreateInfo layoutInfo = {};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
    layoutInfo.pBindings = bindings.data();

    if (vkCreateDescriptorSetLayout(
            logicalDevice, &layoutInfo, nullptr, &descriptorSetLayout
        ) != VK_SUCCESS
    ) {
        throw std::runtime_error("failed to create rt descriptor set layout");
    }
}

void VisibilityManager::createDescriptorPool() {
    std::array<VkDescriptorPoolSize, 4> poolSizes = {};
    poolSizes[0].type = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_NV;
    poolSizes[0].descriptorCount = 1;
    poolSizes[1].type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    poolSizes[1].descriptorCount = 1;
    poolSizes[2].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    poolSizes[2].descriptorCount = 2;
    poolSizes[3].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    poolSizes[3].descriptorCount = 8;

    VkDescriptorPoolCreateInfo poolInfo = {};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
    poolInfo.pPoolSizes = poolSizes.data();
    poolInfo.maxSets = 3;

    if (vkCreateDescriptorPool(
            logicalDevice, &poolInfo, nullptr, &descriptorPool
        ) != VK_SUCCESS
    ) {
        throw std::runtime_error("failed to create rt descriptor pool");
    }
}

void VisibilityManager::createPipeline(
    std::array<VkPipelineShaderStageCreateInfo, 3> shaderStages, VkPipelineLayout *pipelineLayout,
    VkPipeline *pipeline, std::vector<VkDescriptorSetLayout> descriptorSetLayouts
) {
    VkPipelineLayoutCreateInfo pipelineLayoutInfo = {};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = static_cast<uint32_t>(descriptorSetLayouts.size());
    pipelineLayoutInfo.pSetLayouts = descriptorSetLayouts.data();
    if (vkCreatePipelineLayout(
            logicalDevice, &pipelineLayoutInfo, nullptr, pipelineLayout
        ) != VK_SUCCESS
    ) {
        throw std::runtime_error("failed to create rt pipeline layout");
    }

    // Setup shader groups
    std::array<VkRayTracingShaderGroupCreateInfoNV, 3> groups = {};
    for (auto &group : groups) {
        group.sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_NV;
        group.generalShader = VK_SHADER_UNUSED_NV;
        group.closestHitShader = VK_SHADER_UNUSED_NV;
        group.anyHitShader = VK_SHADER_UNUSED_NV;
        group.intersectionShader = VK_SHADER_UNUSED_NV;
    }
    groups[RT_SHADER_INDEX_RAYGEN].type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_NV;
    groups[RT_SHADER_INDEX_RAYGEN].generalShader = RT_SHADER_INDEX_RAYGEN;
    groups[RT_SHADER_INDEX_MISS].type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_NV;
    groups[RT_SHADER_INDEX_MISS].generalShader = RT_SHADER_INDEX_MISS;
    groups[RT_SHADER_INDEX_CLOSEST_HIT].type = VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_NV;
    groups[RT_SHADER_INDEX_CLOSEST_HIT].generalShader = VK_SHADER_UNUSED_NV;
    groups[RT_SHADER_INDEX_CLOSEST_HIT].closestHitShader = RT_SHADER_INDEX_CLOSEST_HIT;

    // Setup ray tracing pipeline
    VkRayTracingPipelineCreateInfoNV rtPipelineInfo = {};
    rtPipelineInfo.sType = VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_NV;
    rtPipelineInfo.stageCount = static_cast<uint32_t>(shaderStages.size());
    rtPipelineInfo.pStages = shaderStages.data();
    rtPipelineInfo.groupCount = static_cast<uint32_t>(groups.size());
    rtPipelineInfo.pGroups = groups.data();
    rtPipelineInfo.maxRecursionDepth = 1;
    rtPipelineInfo.layout = *pipelineLayout;
    if (vkCreateRayTracingPipelinesNV(
            logicalDevice, VK_NULL_HANDLE, 1, &rtPipelineInfo, nullptr, pipeline
        ) != VK_SUCCESS
    ) {
        throw std::runtime_error("failed to create rt pipeline");
    }
}

void VisibilityManager::createRandomSamplingPipeline() {
    // Load shaders
    VkPipelineShaderStageCreateInfo rayGenShaderStageInfo = {};
    rayGenShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    rayGenShaderStageInfo.stage = VK_SHADER_STAGE_RAYGEN_BIT_NV;
    rayGenShaderStageInfo.module = VulkanUtil::createShader(logicalDevice, "shaders/rt/raytrace.rgen.spv");
    rayGenShaderStageInfo.pName = "main";

    VkPipelineShaderStageCreateInfo rayClosestHitShaderStageInfo = {};
    rayClosestHitShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    rayClosestHitShaderStageInfo.stage = VK_SHADER_STAGE_CLOSEST_HIT_BIT_NV;
    rayClosestHitShaderStageInfo.module = VulkanUtil::createShader(logicalDevice, "shaders/rt/raytrace_abs.rchit.spv");
    rayClosestHitShaderStageInfo.pName = "main";

    VkPipelineShaderStageCreateInfo rayMissShaderStageInfo = {};
    rayMissShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    rayMissShaderStageInfo.stage = VK_SHADER_STAGE_MISS_BIT_NV;
    rayMissShaderStageInfo.module = VulkanUtil::createShader(logicalDevice, "shaders/rt/raytrace.rmiss.spv");
    rayMissShaderStageInfo.pName = "main";

    std::array<VkPipelineShaderStageCreateInfo, 3> shaderStages = {};
    shaderStages[RT_SHADER_INDEX_RAYGEN] = rayGenShaderStageInfo;
    shaderStages[RT_SHADER_INDEX_CLOSEST_HIT] = rayClosestHitShaderStageInfo;
    shaderStages[RT_SHADER_INDEX_MISS] = rayMissShaderStageInfo;

    createPipeline(shaderStages, &pipelineLayout, &pipeline, { descriptorSetLayout });

    vkDestroyShaderModule(logicalDevice, rayGenShaderStageInfo.module, nullptr);
    vkDestroyShaderModule(logicalDevice, rayClosestHitShaderStageInfo.module, nullptr);
    vkDestroyShaderModule(logicalDevice, rayMissShaderStageInfo.module, nullptr);
}

void VisibilityManager::createABSPipeline() {
    // Load shaders
    VkPipelineShaderStageCreateInfo rayGenShaderStageInfo = {};
    rayGenShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    rayGenShaderStageInfo.stage = VK_SHADER_STAGE_RAYGEN_BIT_NV;
    rayGenShaderStageInfo.module = VulkanUtil::createShader(logicalDevice, "shaders/rt/raytrace_abs.rgen.spv");
    rayGenShaderStageInfo.pName = "main";

    VkPipelineShaderStageCreateInfo rayClosestHitShaderStageInfo = {};
    rayClosestHitShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    rayClosestHitShaderStageInfo.stage = VK_SHADER_STAGE_CLOSEST_HIT_BIT_NV;
    rayClosestHitShaderStageInfo.module = VulkanUtil::createShader(logicalDevice, "shaders/rt/raytrace_abs.rchit.spv");
    rayClosestHitShaderStageInfo.pName = "main";

    VkPipelineShaderStageCreateInfo rayMissShaderStageInfo = {};
    rayMissShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    rayMissShaderStageInfo.stage = VK_SHADER_STAGE_MISS_BIT_NV;
    rayMissShaderStageInfo.module = VulkanUtil::createShader(logicalDevice, "shaders/rt/raytrace_abs.rmiss.spv");
    rayMissShaderStageInfo.pName = "main";

    std::array<VkPipelineShaderStageCreateInfo, 3> shaderStages = {};
    shaderStages[RT_SHADER_INDEX_RAYGEN] = rayGenShaderStageInfo;
    shaderStages[RT_SHADER_INDEX_CLOSEST_HIT] = rayClosestHitShaderStageInfo;
    shaderStages[RT_SHADER_INDEX_MISS] = rayMissShaderStageInfo;

    createPipeline(
        shaderStages, &pipelineABSLayout, &pipelineABS,
        { descriptorSetLayout, descriptorSetLayoutABS }
    );

    vkDestroyShaderModule(logicalDevice, rayGenShaderStageInfo.module, nullptr);
    vkDestroyShaderModule(logicalDevice, rayClosestHitShaderStageInfo.module, nullptr);
    vkDestroyShaderModule(logicalDevice, rayMissShaderStageInfo.module, nullptr);
}

void VisibilityManager::createEdgeSubdivPipeline() {
    // Load shaders
    VkPipelineShaderStageCreateInfo rayGenShaderStageInfo = {};
    rayGenShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    rayGenShaderStageInfo.stage = VK_SHADER_STAGE_RAYGEN_BIT_NV;
    rayGenShaderStageInfo.module = VulkanUtil::createShader(logicalDevice, "shaders/rt/raytrace_subdiv.rgen.spv");
    rayGenShaderStageInfo.pName = "main";

    VkPipelineShaderStageCreateInfo rayClosestHitShaderStageInfo = {};
    rayClosestHitShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    rayClosestHitShaderStageInfo.stage = VK_SHADER_STAGE_CLOSEST_HIT_BIT_NV;
    rayClosestHitShaderStageInfo.module = VulkanUtil::createShader(logicalDevice, "shaders/rt/raytrace_abs.rchit.spv");
    rayClosestHitShaderStageInfo.pName = "main";

    VkPipelineShaderStageCreateInfo rayMissShaderStageInfo = {};
    rayMissShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    rayMissShaderStageInfo.stage = VK_SHADER_STAGE_MISS_BIT_NV;
    rayMissShaderStageInfo.module = VulkanUtil::createShader(logicalDevice, "shaders/rt/raytrace_abs.rmiss.spv");
    rayMissShaderStageInfo.pName = "main";

    std::array<VkPipelineShaderStageCreateInfo, 3> shaderStages = {};
    shaderStages[RT_SHADER_INDEX_RAYGEN] = rayGenShaderStageInfo;
    shaderStages[RT_SHADER_INDEX_CLOSEST_HIT] = rayClosestHitShaderStageInfo;
    shaderStages[RT_SHADER_INDEX_MISS] = rayMissShaderStageInfo;

    createPipeline(
        shaderStages, &pipelineEdgeSubdivLayout, &pipelineEdgeSubdiv,
        { descriptorSetLayout, descriptorSetLayoutABS, descriptorSetLayoutEdgeSubdiv }
    );

    vkDestroyShaderModule(logicalDevice, rayGenShaderStageInfo.module, nullptr);
    vkDestroyShaderModule(logicalDevice, rayClosestHitShaderStageInfo.module, nullptr);
    vkDestroyShaderModule(logicalDevice, rayMissShaderStageInfo.module, nullptr);
}

void VisibilityManager::createEdgeSubdivDescriptorSetLayout() {
    VkDescriptorSetLayoutBinding edgeSubdivWorkingBufferBinding = {};
    edgeSubdivWorkingBufferBinding.binding = 0;
    edgeSubdivWorkingBufferBinding.descriptorCount = 1;
    edgeSubdivWorkingBufferBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    edgeSubdivWorkingBufferBinding.stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_NV;

    std::array<VkDescriptorSetLayoutBinding, 1> bindings = {
        edgeSubdivWorkingBufferBinding
    };
    VkDescriptorSetLayoutCreateInfo layoutInfo = {};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
    layoutInfo.pBindings = bindings.data();

    if (vkCreateDescriptorSetLayout(
            logicalDevice, &layoutInfo, nullptr, &descriptorSetLayoutEdgeSubdiv
        ) != VK_SUCCESS
    ) {
        throw std::runtime_error("failed to create rt descriptor set layout edge subdivision");
    }
}

void VisibilityManager::createABSDescriptorSetLayout() {
    VkDescriptorSetLayoutBinding triangleOutputBinding = {};
    triangleOutputBinding.binding = 0;
    triangleOutputBinding.descriptorCount = 1;
    triangleOutputBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    triangleOutputBinding.stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_NV;

    /*
    // Vertex array binding
    VkDescriptorSetLayoutBinding vertexLayoutBinding = {};
    vertexLayoutBinding.binding = 1;
    vertexLayoutBinding.descriptorCount = 1;
    vertexLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    vertexLayoutBinding.stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_NV | VK_SHADER_STAGE_CLOSEST_HIT_BIT_NV;
    */

    VkDescriptorSetLayoutBinding absWorkingBufferBinding = {};
    absWorkingBufferBinding.binding = 2;
    absWorkingBufferBinding.descriptorCount = 1;
    absWorkingBufferBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    absWorkingBufferBinding.stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_NV;

    std::array<VkDescriptorSetLayoutBinding, 2> bindings = {
        triangleOutputBinding,
        //vertexLayoutBinding,
        absWorkingBufferBinding
    };
    VkDescriptorSetLayoutCreateInfo layoutInfo = {};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
    layoutInfo.pBindings = bindings.data();

    if (vkCreateDescriptorSetLayout(
            logicalDevice, &layoutInfo, nullptr, &descriptorSetLayoutABS
        ) != VK_SUCCESS
    ) {
        throw std::runtime_error("failed to create rt descriptor set layout ABS");
    }
}

void VisibilityManager::createEdgeSubdivDescriptorSets() {
    std::array<VkWriteDescriptorSet, 1> descriptorWrites = {};

    VkDescriptorBufferInfo edgeSubdivWorkingBufferInfo = {};
    edgeSubdivWorkingBufferInfo.buffer = edgeSubdivWorkingBuffer;
    edgeSubdivWorkingBufferInfo.offset = 0;
    edgeSubdivWorkingBufferInfo.range = VK_WHOLE_SIZE;
    descriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptorWrites[0].dstSet = descriptorSetEdgeSubdiv;
    descriptorWrites[0].dstBinding = 0;
    descriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    descriptorWrites[0].descriptorCount = 1;
    descriptorWrites[0].pBufferInfo = &edgeSubdivWorkingBufferInfo;

    vkUpdateDescriptorSets(
        logicalDevice,
        static_cast<uint32_t>(descriptorWrites.size()),
        descriptorWrites.data(),
        0,
        VK_NULL_HANDLE
    );
}

void VisibilityManager::createABSDescriptorSets() {
    /*
    VkDescriptorSetAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = rtDescriptorPool;
    allocInfo.descriptorSetCount = 1;
    allocInfo.pSetLayouts = &rtDescriptorSetLayoutABS;

    // Allocate descriptor sets
    if (vkAllocateDescriptorSets(
            logicalDevice, &allocInfo, &rtDescriptorSetsABS
        ) != VK_SUCCESS
    ) {
        throw std::runtime_error("failed to allocate rt descriptor sets ABS");
    }
    */

    std::array<VkWriteDescriptorSet, 2> descriptorWrites = {};

    VkDescriptorBufferInfo absOutputBufferInfo = {};        // TODO: Move descriptor set creation to method
    absOutputBufferInfo.buffer = absOutputBuffer;
    absOutputBufferInfo.offset = 0;
    absOutputBufferInfo.range = VK_WHOLE_SIZE;
    descriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptorWrites[0].dstSet = descriptorSetABS;
    descriptorWrites[0].dstBinding = 0;
    descriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    descriptorWrites[0].descriptorCount = 1;
    descriptorWrites[0].pBufferInfo = &absOutputBufferInfo;

    /*
    VkDescriptorBufferInfo vertexBufferInfo = {};
    vertexBufferInfo.buffer = vertexBuffer;
    vertexBufferInfo.offset = 0;
    vertexBufferInfo.range = VK_WHOLE_SIZE;
    descriptorWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptorWrites[1].dstSet = descriptorSetABS;
    descriptorWrites[1].dstBinding = 1;
    descriptorWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    descriptorWrites[1].descriptorCount = 1;
    descriptorWrites[1].pBufferInfo = &vertexBufferInfo;
    */

    VkDescriptorBufferInfo absWorkingBufferInfo = {};
    absWorkingBufferInfo.buffer = absWorkingBuffer;
    absWorkingBufferInfo.offset = 0;
    absWorkingBufferInfo.range = VK_WHOLE_SIZE;
    descriptorWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptorWrites[1].dstSet = descriptorSetABS;
    descriptorWrites[1].dstBinding = 2;
    descriptorWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    descriptorWrites[1].descriptorCount = 1;
    descriptorWrites[1].pBufferInfo = &absWorkingBufferInfo;

    vkUpdateDescriptorSets(
        logicalDevice,
        static_cast<uint32_t>(descriptorWrites.size()),
        descriptorWrites.data(),
        0,
        VK_NULL_HANDLE
    );
}

std::vector<Sample> VisibilityManager::randomSample(int numRays) {
    VkCommandBufferBeginInfo beginInfo = {};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    // This command buffer does not have to be re-recorded every frame(?)
    // Record and execute a command buffer for running the actual random sampling on the GPU
    vkBeginCommandBuffer(commandBuffer, &beginInfo);
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_RAY_TRACING_NV, pipeline);
    vkCmdBindDescriptorSets(
        commandBuffer, VK_PIPELINE_BIND_POINT_RAY_TRACING_NV, pipelineLayout, 0, 1,
        &descriptorSet, 0, nullptr
    );
    vkCmdTraceRaysNV(
        commandBuffer,
        shaderBindingTable, bindingOffsetRayGenShader,
        shaderBindingTable, bindingOffsetMissShader, bindingStride,
        shaderBindingTable, bindingOffsetHitShader, bindingStride,
        VK_NULL_HANDLE, 0, 0,
        numRays, 1, 1
    );
    vkEndCommandBuffer(commandBuffer);
    VulkanUtil::executeCommandBuffer(
        logicalDevice, graphicsQueue, commandBuffer, commandBufferFence
    );

    // Copy intersected triangles from VRAM to CPU accessible memory
    std::vector<Sample> intersectedTriangles(numRays);
    {
        VkDeviceSize bufferSize = sizeof(intersectedTriangles[0]) * intersectedTriangles.size();

        // Create host buffer
        VkBuffer hostBuffer;
        VkDeviceMemory hostBufferMemory;
        VulkanUtil::createBuffer(
            physicalDevice,
            logicalDevice, bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT, hostBuffer, hostBufferMemory,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
        );

        // Copy the intersected triangles GPU buffer to the host buffer
        VulkanUtil::copyBuffer(
            logicalDevice, graphicsCommandPool, graphicsQueue, rayOriginBuffer, hostBuffer,
            bufferSize
        );

        // Map host buffer memory into CPU accessible memory
        void *data;
        vkMapMemory(logicalDevice, hostBufferMemory, 0, bufferSize, 0, &data);
        memcpy(intersectedTriangles.data(), data, bufferSize);
        vkUnmapMemory(logicalDevice, hostBufferMemory);
        vkDestroyBuffer(logicalDevice, hostBuffer, nullptr);
        vkFreeMemory(logicalDevice, hostBufferMemory, nullptr);
    }

    return intersectedTriangles;
}

std::vector<Sample> VisibilityManager::adaptiveBorderSample(const std::vector<Sample> &triangles) {
    // Copy triangles vector to GPU accessible buffer
    {
        VkDeviceSize bufferSize = sizeof(triangles[0]) * triangles.size();

        // Create staging buffer using host-visible memory
        VkBuffer stagingBuffer;
        VkDeviceMemory stagingBufferMemory;
        VulkanUtil::createBuffer(
            physicalDevice, logicalDevice, bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            stagingBuffer, stagingBufferMemory,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
        );

        // Copy triangles data to the staging buffer
        void *data;
        vkMapMemory(logicalDevice, stagingBufferMemory, 0, bufferSize, 0, &data);    // Map buffer memory into CPU accessible memory
        memcpy(data, triangles.data(), (size_t) bufferSize);  // Copy vertex data to mapped memory
        vkUnmapMemory(logicalDevice, stagingBufferMemory);

        // Copy triangles data from the staging buffer to GPU-visible absWorkingBuffer
        VulkanUtil::copyBuffer(
            logicalDevice, graphicsCommandPool, graphicsQueue, stagingBuffer, absWorkingBuffer,
            bufferSize
        );

        vkDestroyBuffer(logicalDevice, stagingBuffer, nullptr);
        vkFreeMemory(logicalDevice, stagingBufferMemory, nullptr);
    }

    // Record and execute a command buffer for running the actual ABS on the GPU
    VkCommandBufferBeginInfo beginInfo = {};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(commandBufferABS, &beginInfo);
    vkCmdBindPipeline(commandBufferABS, VK_PIPELINE_BIND_POINT_RAY_TRACING_NV, pipelineABS);
    vkCmdBindDescriptorSets(
        commandBufferABS, VK_PIPELINE_BIND_POINT_RAY_TRACING_NV, pipelineABSLayout, 0, 1,
        &descriptorSet, 0, nullptr
    );
    vkCmdBindDescriptorSets(    // descriptor set 0 does not have to be bound again (not right)
        commandBufferABS, VK_PIPELINE_BIND_POINT_RAY_TRACING_NV, pipelineABSLayout, 1, 1,
        &descriptorSetABS, 0, nullptr
    );
    vkCmdTraceRaysNV(
        commandBufferABS,
        shaderBindingTableABS, bindingOffsetRayGenShader,
        shaderBindingTableABS, bindingOffsetMissShader, bindingStride,
        shaderBindingTableABS, bindingOffsetHitShader, bindingStride,
        VK_NULL_HANDLE, 0, 0,
        triangles.size() * 9, 1, 1
    );
    vkEndCommandBuffer(commandBufferABS);
    VulkanUtil::executeCommandBuffer(
        logicalDevice, graphicsQueue, commandBufferABS, commandBufferFence
    );

    // Copy intersected triangles from VRAM to CPU accessible memory
    std::vector<Sample> intersectedTriangles(triangles.size() * 9 * 2);
    {
        VkDeviceSize bufferSize = sizeof(intersectedTriangles[0]) * intersectedTriangles.size();

        // Create host buffer
        VkBuffer hostBuffer;
        VkDeviceMemory hostBufferMemory;
        VulkanUtil::createBuffer(
            physicalDevice, logicalDevice, bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT, hostBuffer,
            hostBufferMemory,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
        );

        // Copy the intersected triangles GPU buffer to the host buffer
        VulkanUtil::copyBuffer(
            logicalDevice, graphicsCommandPool, graphicsQueue, absOutputBuffer, hostBuffer,
            bufferSize
        );

        // Map host buffer memory into CPU accessible memory
        void *data;
        vkMapMemory(logicalDevice, hostBufferMemory, 0, bufferSize, 0, &data);
        memcpy(intersectedTriangles.data(), data, bufferSize);
        vkUnmapMemory(logicalDevice, hostBufferMemory);
        vkDestroyBuffer(logicalDevice, hostBuffer, nullptr);
        vkFreeMemory(logicalDevice, hostBufferMemory, nullptr);
    }

    return intersectedTriangles;
}

std::vector<Sample> VisibilityManager::edgeSubdivide(
    const std::vector<Sample> &samples
) {
    // Copy triangles vector to GPU accessible buffer
    {
        VkDeviceSize bufferSize = sizeof(samples[0]) * samples.size();

        // Create staging buffer using host-visible memory
        VkBuffer stagingBuffer;
        VkDeviceMemory stagingBufferMemory;
        VulkanUtil::createBuffer(
            physicalDevice, logicalDevice, bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            stagingBuffer, stagingBufferMemory,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
        );

        // Copy triangles data to the staging buffer
        void *data;
        vkMapMemory(logicalDevice, stagingBufferMemory, 0, bufferSize, 0, &data);    // Map buffer memory into CPU accessible memory
        memcpy(data, samples.data(), (size_t) bufferSize);  // Copy vertex data to mapped memory
        vkUnmapMemory(logicalDevice, stagingBufferMemory);

        // Copy triangles data from the staging buffer to GPU-visible absWorkingBuffer
        VulkanUtil::copyBuffer(
            logicalDevice, graphicsCommandPool, graphicsQueue, stagingBuffer,
            edgeSubdivWorkingBuffer, bufferSize
        );

        vkDestroyBuffer(logicalDevice, stagingBuffer, nullptr);
        vkFreeMemory(logicalDevice, stagingBufferMemory, nullptr);
    }

    // Record and execute a command buffer for running the actual ABS on the GPU
    VkCommandBufferBeginInfo beginInfo = {};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(commandBufferEdgeSubdiv, &beginInfo);
    vkCmdBindPipeline(commandBufferEdgeSubdiv, VK_PIPELINE_BIND_POINT_RAY_TRACING_NV, pipelineEdgeSubdiv);
    vkCmdBindDescriptorSets(
        commandBufferEdgeSubdiv, VK_PIPELINE_BIND_POINT_RAY_TRACING_NV, pipelineEdgeSubdivLayout, 0, 1,
        &descriptorSet, 0, nullptr
    );
    vkCmdBindDescriptorSets(    // descriptor set 0 does not have to be bound again (not right)
        commandBufferEdgeSubdiv, VK_PIPELINE_BIND_POINT_RAY_TRACING_NV, pipelineEdgeSubdivLayout, 1, 1,
        &descriptorSetABS, 0, nullptr
    );
    vkCmdBindDescriptorSets(    // descriptor set 0 does not have to be bound again (not right)
        commandBufferEdgeSubdiv, VK_PIPELINE_BIND_POINT_RAY_TRACING_NV, pipelineEdgeSubdivLayout, 2, 1,
        &descriptorSetEdgeSubdiv, 0, nullptr
    );
    vkCmdTraceRaysNV(
        commandBufferEdgeSubdiv,
        shaderBindingTableEdgeSubdiv, bindingOffsetRayGenShader,
        shaderBindingTableEdgeSubdiv, bindingOffsetMissShader, bindingStride,
        shaderBindingTableEdgeSubdiv, bindingOffsetHitShader, bindingStride,
        VK_NULL_HANDLE, 0, 0,
        samples.size(), 1, 1
    );
    vkEndCommandBuffer(commandBufferEdgeSubdiv);
    VulkanUtil::executeCommandBuffer(
        logicalDevice, graphicsQueue, commandBufferEdgeSubdiv, commandBufferFence
    );

    // Copy intersected triangles from VRAM to CPU accessible memory
    int numSamples = int(pow(2, MAX_SUBDIVISION_STEPS + 1) + 1);
    std::vector<Sample> intersectedTriangles(samples.size() * numSamples);
    {
        VkDeviceSize bufferSize = sizeof(Sample) * intersectedTriangles.size();

        // Create host buffer
        VkBuffer hostBuffer;
        VkDeviceMemory hostBufferMemory;
        VulkanUtil::createBuffer(
            physicalDevice, logicalDevice, bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT, hostBuffer,
            hostBufferMemory,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
        );

        // Copy the intersected triangles GPU buffer to the host buffer
        VulkanUtil::copyBuffer(
            logicalDevice, graphicsCommandPool, graphicsQueue, absOutputBuffer, hostBuffer,
            bufferSize
        );

        // Map host buffer memory into CPU accessible memory
        void *data;
        vkMapMemory(logicalDevice, hostBufferMemory, 0, bufferSize, 0, &data);
        memcpy(intersectedTriangles.data(), data, bufferSize);
        vkUnmapMemory(logicalDevice, hostBufferMemory);
        vkDestroyBuffer(logicalDevice, hostBuffer, nullptr);
        vkFreeMemory(logicalDevice, hostBufferMemory, nullptr);
    }

    return intersectedTriangles;
}

void VisibilityManager::createShaderBindingTable(
    VkBuffer &shaderBindingTable, VkDeviceMemory &shaderBindingTableMemory, VkPipeline &pipeline
) {
    const uint32_t bindingTableSize = rayTracingProperties.shaderGroupHandleSize * 3;
    VulkanUtil::createBuffer(
        physicalDevice,
        logicalDevice, bindingTableSize,
        VK_BUFFER_USAGE_RAY_TRACING_BIT_NV,
        shaderBindingTable,
        shaderBindingTableMemory,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
    );

    // Fetch all the shader handles used in the pipeline, so that they can be written in the SBT
    auto shaderHandleStorage = new uint8_t[bindingTableSize];
    vkGetRayTracingShaderGroupHandlesNV(
        logicalDevice, pipeline, 0, 3, bindingTableSize, shaderHandleStorage
    );

    void *vData;
    vkMapMemory(logicalDevice, shaderBindingTableMemory, 0, bindingTableSize, 0, &vData);

    auto* data = static_cast<uint8_t*>(vData);
    data += copyShaderIdentifier(data, shaderHandleStorage, RT_SHADER_INDEX_RAYGEN);
    data += copyShaderIdentifier(data, shaderHandleStorage, RT_SHADER_INDEX_MISS);
    data += copyShaderIdentifier(data, shaderHandleStorage, RT_SHADER_INDEX_CLOSEST_HIT);

    vkUnmapMemory(logicalDevice, shaderBindingTableMemory);
}

VkDeviceSize VisibilityManager::copyShaderIdentifier(
    uint8_t *data, const uint8_t *shaderHandleStorage, uint32_t groupIndex
) {
    // Copy shader identifier to "data"
    const uint32_t shaderGroupHandleSize = rayTracingProperties.shaderGroupHandleSize;
    memcpy(data, shaderHandleStorage + groupIndex * shaderGroupHandleSize, shaderGroupHandleSize);
    data += shaderGroupHandleSize;

    return shaderGroupHandleSize;
}

void VisibilityManager::rayTrace(const std::vector<uint32_t> &indices) {
    // Execute random sampling
    std::vector<Sample> intersectedTriangles = randomSample(raysPerIteration);

    // Insert the newly found triangles into the PVS
    std::vector<Sample> newSamples;
    for (auto sample : intersectedTriangles) {
        auto result = pvs.insert(sample.triangleID);
        if (result.second) {
            // If the current triangle ID was inserted into the PVS, insert it into the new samples
            // vector as well
            newSamples.push_back(sample);
        }
    }

    // Adaptive Border Sampling. ABS is executed for a maximum of MAX_ABS_RAYS rays at a time as
    // long as there are a number of MIN_ABS_RAYS unprocessed triangles left
    while (newSamples.size() >= MIN_ABS_TRIANGLES_PER_ITERATION) {
        qDebug() << pvs.size();

        // Get a maximum of MAX_ABS_RAYS triangles for which ABS will be run at a time
        int numbAbsRays = std::min(MAX_ABS_TRIANGLES_PER_ITERATION, newSamples.size());
        std::vector<Sample> absWorkingVector(numbAbsRays);
        int num = 0;
        for (auto it = newSamples.begin(); num < numbAbsRays;) {   // TODO: Replace for loop?
            absWorkingVector[num] = *it;
            it = newSamples.erase(it);
            num++;
        }

        // Execute ABS
        std::vector<Sample> intersectedTriangles = adaptiveBorderSample(absWorkingVector);
        //for (auto sample : intersectedTriangles) {
            //qDebug() << sample.triangleID << " " << glm::to_string(sample.rayOrigin).c_str() << " " << glm::to_string(sample.hitPos).c_str();
        //}

        // Insert the newly found triangles into the PVS
        for (auto sample : intersectedTriangles) {
            //qDebug() << sample.triangleID << glm::to_string(sample.rayOrigin).c_str() << glm::to_string(sample.hitPos).c_str();
            if (sample.triangleID != -1) {
                auto result = pvs.insert(sample.triangleID);
                if (result.second) {
                    newSamples.push_back(sample);
                }
            }
        }

        while (intersectedTriangles.size() > 0) {
            std::vector<Sample> a;
            for (int i = 0; i < 9; i++) {
                //qDebug() << glm::to_string(intersectedTriangles[i].hitPos).c_str();
                a.push_back(intersectedTriangles[i]);
                intersectedTriangles.erase(intersectedTriangles.begin() + i);
            }
            std::vector<Sample> n = edgeSubdivide(a);
            for (auto sample : n) {
                if (sample.triangleID != -1) {
                    auto result = pvs.insert(sample.triangleID);
                    if (result.second) {
                        newSamples.push_back(sample);
                    }
                }
            }
        }
    }

    // Copy PVS data to GPU accessible pvs visualization buffer (has the same size as the index vector)
    VkDeviceSize bufferSize = sizeof(indices[0]) * indices.size();

    // Create staging buffer using host-visible memory
    VkBuffer stagingBuffer;
    VkDeviceMemory stagingBufferMemory;
    VulkanUtil::createBuffer(
        physicalDevice,
        logicalDevice, bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, stagingBuffer, stagingBufferMemory,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
    );

    // Collect the vertex indices of the triangles in the PVS
    std::vector<glm::uvec3> pvsIndices(indices.size());
    int i = 0;
    for (auto triangleID : pvs) {
        if (triangleID != -1 && triangleID != 2147483647 && triangleID != -2147483648) {
            pvsIndices[i] = {
                indices[3 * triangleID],
                indices[3 * triangleID + 1],
                indices[3 * triangleID + 2]
            };
            i++;
        }
    }

    // Copy PVS data to the staging buffer
    void *data;
    vkMapMemory(logicalDevice, stagingBufferMemory, 0, bufferSize, 0, &data);    // Map buffer memory into CPU accessible memory
    memcpy(data, pvsIndices.data(), (size_t) bufferSize);  // Copy vertex data to mapped memory
    vkUnmapMemory(logicalDevice, stagingBufferMemory);

    // Copy PVS data from the staging buffer to the GPU-visible PVS visualization buffer (used as an index buffer when drawing)
    VulkanUtil::copyBuffer(
        logicalDevice, graphicsCommandPool, graphicsQueue,stagingBuffer, pvsVisualizationBuffer,
         bufferSize
    );

    vkDestroyBuffer(logicalDevice, stagingBuffer, nullptr);
    vkFreeMemory(logicalDevice, stagingBufferMemory, nullptr);
}

void VisibilityManager::releaseResources() {
    vkDestroyBuffer(logicalDevice, haltonPointsBuffer, nullptr);
    vkFreeMemory(logicalDevice, haltonPointsBufferMemory, nullptr);
    vkDestroyBuffer(logicalDevice, viewCellBuffer, nullptr);
    vkFreeMemory(logicalDevice, viewCellBufferMemory, nullptr);
    vkDestroyBuffer(logicalDevice, intersectedTrianglesBuffer, nullptr);
    vkFreeMemory(logicalDevice, intersectedTrianglesBufferMemory, nullptr);
    vkDestroyBuffer(logicalDevice, rayOriginBuffer, nullptr);
    vkFreeMemory(logicalDevice, rayOriginBufferMemory, nullptr);
    vkDestroyBuffer(logicalDevice, absOutputBuffer, nullptr);
    vkFreeMemory(logicalDevice, absOutputBufferMemory, nullptr);
    vkDestroyBuffer(logicalDevice, absWorkingBuffer, nullptr);
    vkFreeMemory(logicalDevice, absWorkingBufferMemory, nullptr);
    vkDestroyBuffer(logicalDevice, pvsVisualizationBuffer, nullptr);
    vkFreeMemory(logicalDevice, pvsVisualizationBufferMemory, nullptr);
    vkDestroyBuffer(logicalDevice, shaderBindingTable, nullptr);
    vkFreeMemory(logicalDevice, shaderBindingTableMemory, nullptr);
    vkDestroyBuffer(logicalDevice, shaderBindingTableABS, nullptr);
    vkFreeMemory(logicalDevice, shaderBindingTableMemoryABS, nullptr);

    vkDestroyFence(logicalDevice, commandBufferFence, nullptr);

    vkDestroyDescriptorSetLayout(logicalDevice, descriptorSetLayout, nullptr);
    vkDestroyDescriptorSetLayout(logicalDevice, descriptorSetLayoutABS, nullptr);

    VkCommandBuffer commandBuffers[] = {
        commandBuffer,
        commandBufferABS
    };
    vkFreeCommandBuffers(logicalDevice, commandPool, 2, commandBuffers);

    vkDestroyPipeline(logicalDevice, pipeline, nullptr);
    vkDestroyPipelineLayout(logicalDevice, pipelineLayout, nullptr);
    vkDestroyPipeline(logicalDevice, pipelineABS, nullptr);
    vkDestroyPipelineLayout(logicalDevice, pipelineABSLayout, nullptr);

    vkDestroyDescriptorPool(logicalDevice, descriptorPool, nullptr);

    vkDestroyAccelerationStructureNV(logicalDevice, topLevelAS.as, nullptr);
    vkFreeMemory(logicalDevice, topLevelAS.deviceMemory, nullptr);
    vkDestroyAccelerationStructureNV(logicalDevice, bottomLevelAS.as, nullptr);
    vkFreeMemory(logicalDevice, bottomLevelAS.deviceMemory, nullptr);
}

QueueFamilyIndices VisibilityManager::findQueueFamilies() {
    QueueFamilyIndices indices;

    uint32_t queueFamilyCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, nullptr);

    std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, queueFamilies.data());

    int i = 0;
    for (const auto& queueFamily : queueFamilies) {
        if (queueFamily.queueFlags & VK_QUEUE_GRAPHICS_BIT) {
            indices.graphicsFamily = i;
        }

        if (indices.isComplete()) {
            break;
        }

        i++;
    }

    return indices;
}

void VisibilityManager::createCommandBuffers() {
    VkCommandPoolCreateInfo cmdPoolInfo = {};
    cmdPoolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    cmdPoolInfo.queueFamilyIndex = findQueueFamilies().graphicsFamily.value();
    cmdPoolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;    // Has to be set otherwise the command buffers can't be re-recorded
    if (vkCreateCommandPool(logicalDevice, &cmdPoolInfo, nullptr, &commandPool)) {
        throw std::runtime_error("failed to create rt command pool!");
    }

    VkCommandBufferAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.commandPool = commandPool;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = 1;

    if (vkAllocateCommandBuffers(logicalDevice, &allocInfo, &commandBuffer) != VK_SUCCESS) {
        throw std::runtime_error("failed to allocate rt command buffer!");
    }

    if (vkAllocateCommandBuffers(logicalDevice, &allocInfo, &commandBufferABS) != VK_SUCCESS) {
        throw std::runtime_error("failed to allocate rt command buffer!");
    }

    if (vkAllocateCommandBuffers(logicalDevice, &allocInfo, &commandBufferEdgeSubdiv) != VK_SUCCESS) {
        throw std::runtime_error("failed to allocate rt command buffer!");
    }

    // Create fences used to wait for command buffer execution completion after submitting them
    VkFenceCreateInfo fenceInfo;
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fenceInfo.pNext = NULL;
    fenceInfo.flags = 0;
    vkCreateFence(logicalDevice, &fenceInfo, NULL, &commandBufferFence);
}
