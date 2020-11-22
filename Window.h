/*
#ifndef WINDOW_H
#define WINDOW_H

//#include "qvulkanwindow.h"
#include <QVulkanWindow>

class VulkanRenderer;

class Window : public QVulkanWindow {
public:
    Window();
    QVulkanWindowRenderer *createRenderer() override;

public slots:
    void keyReleaseEvent(QKeyEvent *event);

private:
    VulkanRenderer *renderer;
};

#endif // WINDOW_H
*/
