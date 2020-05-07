#include <QGuiApplication>
#include <QtGlobal>
#include <QVulkanInstance>
#include <iostream>
#include <QLoggingCategory>

#include "Window.h"
#include "Renderer.h"

int main(int argc, char *argv[]) {
    int numSubdiv = 3;

    float x[int(std::pow(2, numSubdiv) + 1)];
    x[0] = -1.0;
    x[1] = -2.0;
    x[2] = (x[0] + x[1]) * 0.5;
    int end = 3;
    int numInserted = 1;
    int newNumInserted = 0;
    int i = 0;


    for (int j = 0; j < numSubdiv - 1; j++) {
        for (int i = 0; i < numInserted; i++) {
            for (int k = 0; k < 2; k++) {
                x[end] = (x[k] + x[1 + numInserted + i]) * 0.5;
                qDebug() << x[end];
                end++;
            }
        }
        numInserted *= 2;
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
