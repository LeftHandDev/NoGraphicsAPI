#ifndef SDL_GPU_H
#define SDL_GPU_H

#include "NoGraphicsAPI.h"

#include <SDL3/SDL_vulkan.h>

#ifdef GPU_SURFACE_EXTENSION
#define SDL_WINDOW_GPU SDL_WINDOW_VULKAN
GpuSurface SDL_Gpu_CreateSurface(SDL_Window* window);
void SDL_Gpu_DestroySurface(GpuSurface surface);
#endif // GPU_SURFACE_EXTENSION

#endif // SDL_GPU_H