/*
#include <QKeyEvent>

#include "Window.h"
#include "Renderer.h"

Window::Window() {
}

QVulkanWindowRenderer *Window::createRenderer() {
    renderer = new VulkanRenderer(this);
    return renderer;
}

void Window::keyReleaseEvent(QKeyEvent *event) {
    switch(event->key()) {
        case Qt::Key_V:
            renderer->togglePVSVisualzation();
            break;
        case Qt::Key_S:
            renderer->saveWindowContentToImage();
            break;
    }
}
*/
