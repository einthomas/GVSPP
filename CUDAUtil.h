#ifndef CUDAUTIL_H
#define CUDAUTIL_H

#include <glm/glm.hpp>
#include <glm/vec3.hpp>

#include <cuda_runtime.h>
#include <vulkan/vulkan.h>
#include <vulkan/vulkan_core.h>
#ifdef _WIN64
#define NOMINMAX
#include <windows.h>
#include <vulkan/vulkan_win32.h>
#include <VersionHelpers.h>
#include <dxgi1_2.h>
#include <aclapi.h>
#endif

#include <iostream>
#include <exception>
#include <stdexcept>

#include "vulkanutil.h"
#include "sample.h"

/*
struct Samp {
    alignas(4) int triangleID;
    alignas(16) glm::vec3 rayOrigin;
    alignas(16) glm::vec3 hitPos;
    alignas(16) glm::vec3 pos;
};
*/

#ifdef _WIN64
class WindowsSecurityAttributes
{
protected:
    SECURITY_ATTRIBUTES m_winSecurityAttributes;
    PSECURITY_DESCRIPTOR m_winPSecurityDescriptor;

public:
    WindowsSecurityAttributes()
    {
        m_winPSecurityDescriptor = (PSECURITY_DESCRIPTOR)calloc(1, SECURITY_DESCRIPTOR_MIN_LENGTH + 2 * sizeof(void **));
        if (!m_winPSecurityDescriptor) {
            throw std::runtime_error("Failed to allocate memory for security descriptor");
        }

        PSID *ppSID = (PSID *)((PBYTE)m_winPSecurityDescriptor + SECURITY_DESCRIPTOR_MIN_LENGTH);
        PACL *ppACL = (PACL *)((PBYTE)ppSID + sizeof(PSID *));

        InitializeSecurityDescriptor(m_winPSecurityDescriptor, SECURITY_DESCRIPTOR_REVISION);

        SID_IDENTIFIER_AUTHORITY sidIdentifierAuthority = SECURITY_WORLD_SID_AUTHORITY;
        AllocateAndInitializeSid(&sidIdentifierAuthority, 1, SECURITY_WORLD_RID, 0, 0, 0, 0, 0, 0, 0, ppSID);

        EXPLICIT_ACCESS explicitAccess;
        ZeroMemory(&explicitAccess, sizeof(EXPLICIT_ACCESS));
        explicitAccess.grfAccessPermissions = STANDARD_RIGHTS_ALL | SPECIFIC_RIGHTS_ALL;
        explicitAccess.grfAccessMode = SET_ACCESS;
        explicitAccess.grfInheritance = INHERIT_ONLY;
        explicitAccess.Trustee.TrusteeForm = TRUSTEE_IS_SID;
        explicitAccess.Trustee.TrusteeType = TRUSTEE_IS_WELL_KNOWN_GROUP;
        explicitAccess.Trustee.ptstrName = (LPTSTR) * ppSID;

        SetEntriesInAcl(1, &explicitAccess, NULL, ppACL);

        SetSecurityDescriptorDacl(m_winPSecurityDescriptor, TRUE, *ppACL, FALSE);

        m_winSecurityAttributes.nLength = sizeof(m_winSecurityAttributes);
        m_winSecurityAttributes.lpSecurityDescriptor = m_winPSecurityDescriptor;
        m_winSecurityAttributes.bInheritHandle = TRUE;
    }

    SECURITY_ATTRIBUTES *operator&()
    {
        return &m_winSecurityAttributes;
    }
    ~WindowsSecurityAttributes()
    {
        PSID *ppSID = (PSID *)((PBYTE)m_winPSecurityDescriptor + SECURITY_DESCRIPTOR_MIN_LENGTH);
        PACL *ppACL = (PACL *)((PBYTE)ppSID + sizeof(PSID *));

        if (*ppSID) {
            FreeSid(*ppSID);
        }
        if (*ppACL) {
            LocalFree(*ppACL);
        }
        free(m_winPSecurityDescriptor);
    }
};
#endif /* _WIN64 */

class CUDAUtil {
public:
    static int work(int *pvs, int *triangleIDKeys, Sample *sampleValues, std::vector<Sample> &result, int pvsSize, int triangleIDKeysSize);
    static int initCuda(uint8_t *vkDeviceUUID, size_t UUID_SIZE);
    static void generateHaltonSequence(int n, float *sequence);

    static void importCudaExternalMemory(
        void **cudaPtr, cudaExternalMemory_t &cudaMem, VkDeviceMemory &vkMem, VkDeviceSize size,
        VkDevice device
    ) {
        VkExternalMemoryHandleTypeFlagBits handleType = getDefaultMemHandleType();

        cudaExternalMemoryHandleDesc externalMemoryHandleDesc = {};

        if (handleType & VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT) {
            externalMemoryHandleDesc.type = cudaExternalMemoryHandleTypeOpaqueWin32;
        }
        else if (handleType & VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_KMT_BIT) {
            externalMemoryHandleDesc.type = cudaExternalMemoryHandleTypeOpaqueWin32Kmt;
        }
        else if (handleType & VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT) {
            externalMemoryHandleDesc.type = cudaExternalMemoryHandleTypeOpaqueFd;
        }

        externalMemoryHandleDesc.size = size;

    #ifdef _WIN64
        externalMemoryHandleDesc.handle.win32.handle = (HANDLE)getMemHandle(vkMem, handleType, device);
    #else
        externalMemoryHandleDesc.handle.fd = (int)(uintptr_t)getMemHandle(vkMem, handleType, device);
    #endif

        cudaImportExternalMemory(&cudaMem, &externalMemoryHandleDesc);

        cudaExternalMemoryBufferDesc externalMemBufferDesc = {};
        externalMemBufferDesc.offset = 0;
        externalMemBufferDesc.size = size;
        externalMemBufferDesc.flags = 0;

        cudaExternalMemoryGetMappedBuffer(cudaPtr, cudaMem, &externalMemBufferDesc);
    }

    static void createExternalBuffer(
        VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties,
        VkBuffer& buffer, VkDeviceMemory& bufferMemory, VkDevice m_device,
        VkPhysicalDevice m_physicalDevice
    ) {
        VkExternalMemoryHandleTypeFlagsKHR extMemHandleType = getDefaultMemHandleType();

        VkBufferCreateInfo bufferInfo = {};
        bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bufferInfo.size = size;
        bufferInfo.usage = usage;
        bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        if (vkCreateBuffer(m_device, &bufferInfo, nullptr, &buffer) != VK_SUCCESS) {
            std::cout << "failed to create buffer!" << std::endl;
        }

        VkMemoryRequirements memRequirements;
        vkGetBufferMemoryRequirements(m_device, buffer, &memRequirements);

    #ifdef _WIN64
        WindowsSecurityAttributes winSecurityAttributes;

        VkExportMemoryWin32HandleInfoKHR vulkanExportMemoryWin32HandleInfoKHR = {};
        vulkanExportMemoryWin32HandleInfoKHR.sType = VK_STRUCTURE_TYPE_EXPORT_MEMORY_WIN32_HANDLE_INFO_KHR;
        vulkanExportMemoryWin32HandleInfoKHR.pNext = NULL;
        vulkanExportMemoryWin32HandleInfoKHR.pAttributes = &winSecurityAttributes;
        vulkanExportMemoryWin32HandleInfoKHR.dwAccess = DXGI_SHARED_RESOURCE_READ | DXGI_SHARED_RESOURCE_WRITE;
        vulkanExportMemoryWin32HandleInfoKHR.name = (LPCWSTR)NULL;
    #endif
        VkExportMemoryAllocateInfoKHR vulkanExportMemoryAllocateInfoKHR = {};
        vulkanExportMemoryAllocateInfoKHR.sType = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO_KHR;
    #ifdef _WIN64
        vulkanExportMemoryAllocateInfoKHR.pNext = extMemHandleType & VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT_KHR ? &vulkanExportMemoryWin32HandleInfoKHR : NULL;
        vulkanExportMemoryAllocateInfoKHR.handleTypes = extMemHandleType;
    #else
        vulkanExportMemoryAllocateInfoKHR.pNext = NULL;
        vulkanExportMemoryAllocateInfoKHR.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
    #endif
        VkMemoryAllocateInfo allocInfo = {};
        allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocInfo.pNext = &vulkanExportMemoryAllocateInfoKHR;
        allocInfo.allocationSize = memRequirements.size;
        allocInfo.memoryTypeIndex = VulkanUtil::findMemoryType(m_physicalDevice, memRequirements.memoryTypeBits, properties);

        if (vkAllocateMemory(m_device, &allocInfo, nullptr, &bufferMemory) != VK_SUCCESS) {
            std::cout << "failed to allocate external buffer memory!" << std::endl;
        }

        vkBindBufferMemory(m_device, buffer, bufferMemory, 0);
    }

private:
    static VkExternalMemoryHandleTypeFlagBits getDefaultMemHandleType() {
    #ifdef _WIN64
        return IsWindows8Point1OrGreater() ?
               VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT :
               VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_KMT_BIT;
    #else
        return VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
    #endif
    }

    static void* getMemHandle(VkDeviceMemory memory, VkExternalMemoryHandleTypeFlagBits handleType, VkDevice m_device) {
    #ifdef _WIN64
        HANDLE handle = 0;

        VkMemoryGetWin32HandleInfoKHR vkMemoryGetWin32HandleInfoKHR = {};
        vkMemoryGetWin32HandleInfoKHR.sType = VK_STRUCTURE_TYPE_MEMORY_GET_WIN32_HANDLE_INFO_KHR;
        vkMemoryGetWin32HandleInfoKHR.pNext = NULL;
        vkMemoryGetWin32HandleInfoKHR.memory = memory;
        vkMemoryGetWin32HandleInfoKHR.handleType = handleType;

        PFN_vkGetMemoryWin32HandleKHR fpGetMemoryWin32HandleKHR;
        fpGetMemoryWin32HandleKHR = (PFN_vkGetMemoryWin32HandleKHR)vkGetDeviceProcAddr(m_device, "vkGetMemoryWin32HandleKHR");
        if (!fpGetMemoryWin32HandleKHR) {
            std::cout << "Failed to retrieve vkGetMemoryWin32HandleKHR!" << std::endl;
        }
        if (fpGetMemoryWin32HandleKHR(m_device, &vkMemoryGetWin32HandleInfoKHR, &handle) != VK_SUCCESS) {
            std::cout << "Failed to retrieve handle for buffer!" << std::endl;
        }
        return (void *)handle;
    #else
        int fd = -1;

        VkMemoryGetFdInfoKHR vkMemoryGetFdInfoKHR = {};
        vkMemoryGetFdInfoKHR.sType = VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR;
        vkMemoryGetFdInfoKHR.pNext = NULL;
        vkMemoryGetFdInfoKHR.memory = memory;
        vkMemoryGetFdInfoKHR.handleType = handleType;

        PFN_vkGetMemoryFdKHR fpGetMemoryFdKHR;
        fpGetMemoryFdKHR = (PFN_vkGetMemoryFdKHR)vkGetDeviceProcAddr(m_device, "vkGetMemoryFdKHR");
        if (!fpGetMemoryFdKHR) {
            throw std::runtime_error("Failed to retrieve vkGetMemoryWin32HandleKHR!");
        }
        if (fpGetMemoryFdKHR(m_device, &vkMemoryGetFdInfoKHR, &fd) != VK_SUCCESS) {
            throw std::runtime_error("Failed to retrieve handle for buffer!");
        }
        return (void *)(uintptr_t)fd;
    #endif /* _WIN64 */
    }
};

#endif // CUDAUTIL_H
