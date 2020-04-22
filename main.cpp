#include <QGuiApplication>
#include <QtGlobal>
#include <QVulkanInstance>
#include <iostream>
#include <QLoggingCategory>

#include "Window.h"
#include "Renderer.h"

int main(int argc, char *argv[]) {
    int x[20];
    x[0] = 0;
    x[1] = 1;
    x[2] = 1;
    int end = 3;
    int numInserted = 1;
    int newNumInserted = 0;
    int i = 0;
    for (int j = 0; j < 3; j++) {
        for (int k = 0; k < 2; k++) {
            for (i = 0; i < numInserted; i++) {
                x[end + i] = x[k] + x[end - numInserted + i];
            }
            newNumInserted += i;
            end += i;
        }
        numInserted = newNumInserted;
        newNumInserted = 0;
    }

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
}
