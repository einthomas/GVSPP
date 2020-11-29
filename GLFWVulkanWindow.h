#ifndef GLFWWINDOW_H
#define GLFWWINDOW_H

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <iostream>
#include <stdexcept>
#include <functional>
#include <cstdlib>
#include <optional>
#include <set>
#include <algorithm>
#include <vector>
#include <cstring>
#include <fstream>
#include <array>

#ifdef _WIN64
#define NOMINMAX
#include <windows.h>
#include <vulkan/vulkan_win32.h>
#include <VersionHelpers.h>
#include <dxgi1_2.h>
#include <aclapi.h>
#endif

class VulkanRenderer;

const int WIDTH = 1024;
const int HEIGHT = 1024;

const std::vector<const char*> validationLayers = {
    "VK_LAYER_KHRONOS_validation"
};

const std::vector<const char*> deviceExtensions = {
    VK_KHR_SWAPCHAIN_EXTENSION_NAME,
    VK_NV_RAY_TRACING_EXTENSION_NAME,
    VK_KHR_MAINTENANCE3_EXTENSION_NAME,
    VK_KHR_GET_MEMORY_REQUIREMENTS_2_EXTENSION_NAME,
    VK_KHR_DEDICATED_ALLOCATION_EXTENSION_NAME,
    VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME,
    VK_KHR_EXTERNAL_SEMAPHORE_EXTENSION_NAME,
    VK_KHR_MULTIVIEW_EXTENSION_NAME,
#ifdef _WIN64
    VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME,
    VK_KHR_EXTERNAL_SEMAPHORE_WIN32_EXTENSION_NAME,
#else
    VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME,
    VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME
#endif /* _WIN64 */
};

struct SwapChainSupportDetails {
    VkSurfaceCapabilitiesKHR capabilities;
    std::vector<VkSurfaceFormatKHR> formats;
    std::vector<VkPresentModeKHR> presentModes;
};

struct QueueFamilyIndices {
    std::optional<uint32_t> graphicsFamily;
    std::optional<uint32_t> presentFamily;
    std::optional<uint32_t> computeFamily;
    std::optional<uint32_t> transferFamily;

    bool isComplete() {
        return graphicsFamily.has_value() && presentFamily.has_value() && computeFamily.has_value() && transferFamily.has_value();
    }
};

const bool enableValidationLayers = false;

class GLFWVulkanWindow {
public:
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    VkDevice device;
    std::array<uint8_t, VK_UUID_SIZE> deviceUUID;
    uint32_t imageCount;
    VkRenderPass renderPass;
    VkQueue graphicsQueue;
    VkExtent2D swapChainImageSize;
    VkCommandPool graphicsCommandPool;
    VkSampleCountFlagBits msaaSamples;
    std::vector<VkFramebuffer> swapChainFramebuffers;
    VkExtent2D swapChainExtent;
    std::vector<VkImage> swapChainImages;
    std::vector<VkImageView> swapChainImageViews;

    void initWindow();
    void initVulkan();
    void initRenderer();
    void mainLoop();
    VkFormat findDepthFormat();

private:
    const int MAX_FRAMES_IN_FLIGHT = 2;
    static float cameraSpeed;

    GLFWwindow* window;
    VkInstance instance;

    VkQueue presentQueue;
    VkSurfaceKHR surface;

    VkPipeline pipeline;
    VkPipelineLayout pipelineLayout;
    VulkanRenderer *renderer;

    VkImage depthImage;
    VkDeviceMemory depthImageMemory;
    VkImageView depthImageView;
    VkImage colorImage;     // MSAA render target
    VkDeviceMemory colorImageMemory;
    VkImageView colorImageView;

    VkSwapchainKHR swapChain;
    VkFormat swapChainImageFormat;

    size_t currentFrame = 0;
    std::vector<VkSemaphore> imageAvailableSemaphores;
    std::vector<VkSemaphore> renderFinishedSemaphores;
    std::vector<VkFence> inFlightFences;
    std::vector<VkFence> imagesInFlight;
    std::vector<VkCommandBuffer> commandBuffers;

    float deltaTime = 0.0f;
    float lastFrame = 0.0f;
    bool firstMouse = true;
    double lastX;
    double lastY;
    double yaw = 0.0f;
    double pitch = 0.0f;

    void createSurface();
    void createInstance();
    bool checkValidationLayerSupport();
    void pickPhysicalDevice();
    bool isDeviceSuitable(VkPhysicalDevice device);
    bool checkDeviceExtensionSupport(VkPhysicalDevice device);
    QueueFamilyIndices findQueueFamilies(VkPhysicalDevice device);
    void createLogicalDevice();
    void createSwapChain();
    SwapChainSupportDetails querySwapChainSupport(VkPhysicalDevice device);
    VkSurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& availableFormats);
    VkPresentModeKHR chooseSwapPresentMode(const std::vector<VkPresentModeKHR>& availablePresentModes);
    VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities);
    void createImageViews();
    void createRenderPass();
    void createFramebuffers();
    void createCommandPool();
    void createSyncObjects();
    void createCommandBuffers();
    void createDepthResources();
    void createColorResources();
    VkFormat findSupportedFormat(const std::vector<VkFormat>& candidates, VkImageTiling tiling, VkFormatFeatureFlags features);
    void cleanup();
    static void mouseCallback(GLFWwindow* window, double xpos, double ypos);
    static void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods);
};

#endif // GLFWWINDOW_H
