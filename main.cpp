//#include <QGuiApplication>
//#include <QtGlobal>
//#include <QVulkanInstance>
#include <iostream>
//#include <QLoggingCategory>

//#include "Window.h"
#include "GLFWVulkanWindow.h"
#include "Renderer.h"

#include <sstream>
#include <string>

int main(int argc, char *argv[]) {
    /*
    QGuiApplication app(argc, argv);

    QVulkanInstance vulkanInstance;

    // Linux
    //if (qEnvironmentVariableIntValue("QT_VK_DEBUG")) {
        //QLoggingCategory::setFilterRules(QStringLiteral("qt.vulkan=true"));
        //vulkanInstance.setLayers(QByteArrayList() << "VK_LAYER_LUNARG_standard_validation"");
    //}
    vulkanInstance.setLayers(QByteArrayList() << "VK_LAYER_KHRONOS_validation");    // TODO: Remove for release build

    if (!vulkanInstance.create()) {
        qWarning("failed to create Vulkan instance");
    }

    Window window;
    window.setDeviceExtensions(
        QByteArrayList()
        << VK_NV_RAY_TRACING_EXTENSION_NAME
        << VK_KHR_MAINTENANCE3_EXTENSION_NAME
        << VK_KHR_GET_MEMORY_REQUIREMENTS_2_EXTENSION_NAME
    );
    window.setVulkanInstance(&vulkanInstance);
    window.resize(800, 600);
    window.show();

    //VkPhysicalDeviceFeatures f = {};
    //f.samplerAnisotropy = VK_TRUE;

    return app.exec();
    */

    GLFWVulkanWindow app;

    try {
        app.initWindow();
        app.initVulkan();
        app.initRenderer();
        app.mainLoop();
    } catch (const std::exception & e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
