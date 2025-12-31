#define GPU_EXPOSE_INTERNAL
#include "Config.h"
#include "NoGraphicsAPI.h"

#ifdef GPU_SURFACE_EXTENSION
#include "External/VkBootstrap.h"

#include <SDL3/SDL.h>
#include <SDL3/SDL_vulkan.h>

struct GpuSurface_T { VkSurfaceKHR surface = VK_NULL_HANDLE; };

GpuSurface SDL_Gpu_CreateSurface(void *sdlWindow)
{
    VkSurfaceKHR surface;
    SDL_Vulkan_CreateSurface(static_cast<SDL_Window*>(sdlWindow), static_cast<VkInstance>(gpuVulkanInstance()), nullptr, &surface);
    return new GpuSurface_T { surface };
}

void SDL_Gpu_DestroySurface(GpuSurface surface)
{
    SDL_Vulkan_DestroySurface(static_cast<VkInstance>(gpuVulkanInstance()), surface->surface, nullptr);
    delete surface;
}

#endif // GPU_SURFACE_EXTENSION