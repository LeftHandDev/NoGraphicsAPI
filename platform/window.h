#ifndef NGAPI_WINDOW_H
#define NGAPI_WINDOW_H

#include "NoGraphicsAPI.h"

// A minimal, backend-agnostic window + input layer for the samples. Exactly one
// backend (GLFW or SDL3) is compiled in, selected by the build (NGAPI_WINDOW_BACKEND).
namespace ngapi
{
    enum class Key
    {
        A,
        S,
        T,
        R,
        X,
        Escape,
        Up,
        Down,
        Left,
        Right,
        Count
    };

    struct Window;

    // Initializes the windowing backend (idempotent) and opens a window.
    Window* createWindow(const char* title, int width, int height);
    void destroyWindow(Window* window);

    // Creates / destroys a Vulkan surface for the window using the instance from
    // gpuCreateInstance(), which must have been called first.
    GpuSurface createSurface(Window* window);
    void destroySurface(Window* window, GpuSurface surface);

    // Pumps the OS event queue and refreshes input state. Call once per frame.
    void pollEvents(Window* window);

    // True once the user has asked to close the window.
    bool shouldClose(Window* window);

    // Input state, valid after pollEvents(). isKeyDown == currently held;
    // wasKeyPressed == transitioned to down on the most recent pollEvents().
    bool isKeyDown(Window* window, Key key);
    bool wasKeyPressed(Window* window, Key key);
} // namespace ngapi

#endif // NGAPI_WINDOW_H
