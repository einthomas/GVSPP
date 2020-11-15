//#include <QGuiApplication>
//#include <QtGlobal>
//#include <QVulkanInstance>
#include <iostream>
#include <random>
//#include <QLoggingCategory>

//#include "Window.h"
#include "GLFWVulkanWindow.h"
#include "Renderer.h"

#include "gpuHashTable/linearprobing.h"

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/string_cast.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/rotate_vector.hpp>

int main(int argc, char *argv[]) {
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
