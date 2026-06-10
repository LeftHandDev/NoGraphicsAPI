#define GPU_EXPOSE_INTERNAL
#include "window.h"

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <array>
#include <cstddef>

namespace ngapi
{
    struct Window
    {
        GLFWwindow* handle = nullptr;
        std::array<bool, static_cast<size_t>(Key::Count)> down{};
        std::array<bool, static_cast<size_t>(Key::Count)> pressed{};
    };

    static int toGlfwKey(Key key)
    {
        switch (key)
        {
        case Key::A:
            return GLFW_KEY_A;
        case Key::S:
            return GLFW_KEY_S;
        case Key::T:
            return GLFW_KEY_T;
        case Key::R:
            return GLFW_KEY_R;
        case Key::X:
            return GLFW_KEY_X;
        case Key::Escape:
            return GLFW_KEY_ESCAPE;
        case Key::Up:
            return GLFW_KEY_UP;
        case Key::Down:
            return GLFW_KEY_DOWN;
        case Key::Left:
            return GLFW_KEY_LEFT;
        case Key::Right:
            return GLFW_KEY_RIGHT;
        default:
            return GLFW_KEY_UNKNOWN;
        }
    }

    Window* createWindow(const char* title, int width, int height)
    {
        glfwInit();
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API); // Vulkan; no OpenGL context
        glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);   // samples use a fixed-size swapchain

        Window* window = new Window();
        window->handle = glfwCreateWindow(width, height, title, nullptr, nullptr);
        return window;
    }

    void destroyWindow(Window* window)
    {
        if (window == nullptr)
        {
            return;
        }
        if (window->handle != nullptr)
        {
            glfwDestroyWindow(window->handle);
        }
        delete window;
        glfwTerminate();
    }

    GpuSurface createSurface(Window* window)
    {
        VkSurfaceKHR surface = VK_NULL_HANDLE;
        glfwCreateWindowSurface(static_cast<VkInstance>(gpuVulkanInstance()), window->handle, nullptr, &surface);
        return gpuCreateSurface(surface);
    }

    void destroySurface(Window* window, GpuSurface surface)
    {
        // GLFW has no surface-destroy helper; release the VkSurfaceKHR directly,
        // then free the wrapper (gpuDestroySurface only frees the wrapper).
        VkSurfaceKHR vkSurface = static_cast<VkSurfaceKHR>(gpuVulkanSurface(surface));
        vkDestroySurfaceKHR(static_cast<VkInstance>(gpuVulkanInstance()), vkSurface, nullptr);
        gpuDestroySurface(surface);
    }

    void pollEvents(Window* window)
    {
        glfwPollEvents();
        for (size_t i = 0; i < window->down.size(); i++)
        {
            bool now = glfwGetKey(window->handle, toGlfwKey(static_cast<Key>(i))) == GLFW_PRESS;
            window->pressed[i] = now && !window->down[i];
            window->down[i] = now;
        }
    }

    bool shouldClose(Window* window)
    {
        return glfwWindowShouldClose(window->handle);
    }

    bool isKeyDown(Window* window, Key key)
    {
        return window->down[static_cast<size_t>(key)];
    }

    bool wasKeyPressed(Window* window, Key key)
    {
        return window->pressed[static_cast<size_t>(key)];
    }
} // namespace ngapi
