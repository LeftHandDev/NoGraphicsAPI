#define GPU_EXPOSE_INTERNAL
#include "SDL_gpu.h"
#include "External/VkBootstrap.h"

#ifdef GPU_SURFACE_EXTENSION

GpuSurface SDL_Gpu_CreateSurface(void *sdlWindow)
{
    VkSurfaceKHR surface;
    SDL_Vulkan_CreateSurface(static_cast<SDL_Window*>(sdlWindow), static_cast<VkInstance>(gpuVulkanInstance()), nullptr, &surface);
    return gpuCreateSurface(surface);
}

void SDL_Gpu_DestroySurface(GpuSurface surface)
{
    VkSurfaceKHR vkSurface = static_cast<VkSurfaceKHR>(gpuVulkanSurface(surface));
    SDL_Vulkan_DestroySurface(static_cast<VkInstance>(gpuVulkanInstance()), vkSurface, nullptr);
    gpuDestroySurface(surface);
}

#endif