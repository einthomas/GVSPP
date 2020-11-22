#include <iostream>
#include <random>
#include "GLFWVulkanWindow.h"
#include "Renderer.h"

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
