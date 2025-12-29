#ifndef NO_GRAPHICS_API_CONFIG_H
#define NO_GRAPHICS_API_CONFIG_H

#define GPU_SURFACE_EXTENSION
#define GPU_RAY_TRACING_EXTENSION

#ifdef GPU_EXPOSE_INTERNAL
void* gpuVulkanInstance();
#endif // GPU_EXPOSE_INTERNAL

#ifdef GPU_SURFACE_EXTENSION
struct GpuSurface_T;
using GpuSurface = GpuSurface_T*;

GpuSurface SDL_Gpu_CreateSurface(void* window);
void SDL_Gpu_DestroySurface(GpuSurface surface);
#endif // GPU_SURFACE_EXTENSION

#endif // NO_GRAPHICS_API_CONFIG_H