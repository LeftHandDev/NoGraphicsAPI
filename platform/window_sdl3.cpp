#define GPU_EXPOSE_INTERNAL
#include "window.h"

#include <vulkan/vulkan.h>
#include <SDL3/SDL.h>
#include <SDL3/SDL_vulkan.h>

#include <array>
#include <cstddef>

namespace ngapi
{
    struct Window
    {
        SDL_Window* handle = nullptr;
        bool closeRequested = false;
        std::array<bool, static_cast<size_t>(Key::Count)> down{};
        std::array<bool, static_cast<size_t>(Key::Count)> pressed{};
    };

    static int toKeyIndex(SDL_Keycode key)
    {
        switch (key)
        {
        case SDLK_A:
            return static_cast<int>(Key::A);
        case SDLK_S:
            return static_cast<int>(Key::S);
        case SDLK_T:
            return static_cast<int>(Key::T);
        case SDLK_R:
            return static_cast<int>(Key::R);
        case SDLK_X:
            return static_cast<int>(Key::X);
        case SDLK_ESCAPE:
            return static_cast<int>(Key::Escape);
        case SDLK_UP:
            return static_cast<int>(Key::Up);
        case SDLK_DOWN:
            return static_cast<int>(Key::Down);
        case SDLK_LEFT:
            return static_cast<int>(Key::Left);
        case SDLK_RIGHT:
            return static_cast<int>(Key::Right);
        default:
            return -1;
        }
    }

    Window* createWindow(const char* title, int width, int height)
    {
        SDL_Init(SDL_INIT_VIDEO);
        Window* window = new Window();
        window->handle = SDL_CreateWindow(title, width, height, SDL_WINDOW_VULKAN);
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
            SDL_DestroyWindow(window->handle);
        }
        delete window;
        SDL_Quit();
    }

    GpuSurface createSurface(Window* window)
    {
        VkSurfaceKHR surface = VK_NULL_HANDLE;
        SDL_Vulkan_CreateSurface(window->handle, static_cast<VkInstance>(gpuVulkanInstance()), nullptr, &surface);
        return gpuCreateSurface(surface);
    }

    void destroySurface(Window* window, GpuSurface surface)
    {
        VkSurfaceKHR vkSurface = static_cast<VkSurfaceKHR>(gpuVulkanSurface(surface));
        SDL_Vulkan_DestroySurface(static_cast<VkInstance>(gpuVulkanInstance()), vkSurface, nullptr);
        gpuDestroySurface(surface);
    }

    void pollEvents(Window* window)
    {
        window->pressed.fill(false);
        SDL_Event event;
        while (SDL_PollEvent(&event))
        {
            if (event.type == SDL_EVENT_QUIT)
            {
                window->closeRequested = true;
            }
            else if (event.type == SDL_EVENT_KEY_DOWN)
            {
                int i = toKeyIndex(event.key.key);
                if (i >= 0)
                {
                    if (!window->down[i])
                    {
                        window->pressed[i] = true;
                    }
                    window->down[i] = true;
                }
            }
            else if (event.type == SDL_EVENT_KEY_UP)
            {
                int i = toKeyIndex(event.key.key);
                if (i >= 0)
                {
                    window->down[i] = false;
                }
            }
        }
    }

    bool shouldClose(Window* window)
    {
        return window->closeRequested;
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
