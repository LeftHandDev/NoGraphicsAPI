// Windowed MSDF text sample: renders Inter at a range of sizes and colors from a
// single multi-channel signed distance field atlas, demonstrating crisp scaling,
// per-glyph color, and an MTSDF outline. The text is rendered directly into the
// swapchain image at native resolution (no intermediate blit), so it stays sharp.
#include <cstdio>

#include "window.h"

#include "MsdfFont.h"
#include "MsdfDemo.h"

int main()
{
    gpuCreateInstance();
    auto device = gpuCreateDevice(0);

    const uint FRAMES_IN_FLIGHT = 2;

    auto window = ngapi::createWindow("NoGraphicsAPI - MSDF Text", 1280, 720);
    auto surface = ngapi::createSurface(window);

    auto swapchain = gpuCreateSwapchain(device, surface, FRAMES_IN_FLIGHT);
    auto swapchainDesc = gpuSwapchainDesc(swapchain);

    auto queue = gpuCreateQueue(device);
    auto semaphore = gpuCreateSemaphore(device, 0);
    uint64_t nextFrame = 1;

    // Scope the renderer (and its fonts) so their GPU resources are freed while
    // the device is still alive, before gpuDestroyDevice below.
    {
        // The pipeline renders straight into the swapchain image, so its color
        // format must match the swapchain's (dynamic rendering requires it).
        MsdfTextRenderer renderer(device, swapchainDesc.format);
        MsdfFont* inter = renderer.loadFont("assets/fonts/Inter/Inter-msdf.json",
                                            "assets/fonts/Inter/Inter-msdf.png");

        // The scene is (re)built for the swapchain's current size. On Wayland the
        // surface is often resized away from the requested size by the first
        // configure event, and can change again on a window resize.
        uint32_t curW = 0, curH = 0;

        while (!ngapi::shouldClose(window))
        {
            ngapi::pollEvents(window);
            if (ngapi::wasKeyPressed(window, ngapi::Key::Escape))
            {
                break;
            }

            if (nextFrame > FRAMES_IN_FLIGHT)
            {
                gpuWaitSemaphore(semaphore, nextFrame - FRAMES_IN_FLIGHT);
            }

            // Acquiring may rebuild the swapchain (resize); read its size after.
            auto image = gpuSwapchainImage(swapchain);
            auto desc = gpuSwapchainDesc(swapchain);
            if (desc.dimensions.x != curW || desc.dimensions.y != curH)
            {
                // Drain any in-flight frames still reading the glyph buffer before
                // re-laying-out the scene into it.
                if (nextFrame > 1)
                {
                    gpuWaitSemaphore(semaphore, nextFrame - 1);
                }
                curW = desc.dimensions.x;
                curH = desc.dimensions.y;
                buildMsdfDemoScene(renderer, inter, static_cast<float>(curW), static_cast<float>(curH));
                std::printf("MSDF sample: render surface %ux%u (rendered at native res)\n", curW, curH);
            }

            auto cmd = gpuStartCommandRecording(queue);

            // Draw directly into the swapchain image: clears to black, then text.
            renderer.render(cmd, image, curW, curH, LOAD_OP_CLEAR);

            gpuSubmit(queue, Span<GpuCommandBuffer>(&cmd, 1), semaphore, nextFrame);
            gpuPresent(swapchain, semaphore, nextFrame++);
        }

        gpuWaitSemaphore(semaphore, nextFrame - 1);

        // Swapchain drains all queues, so the timeline semaphore is idle when
        // destroyed; do it before tearing the rest down.
        gpuDestroySwapchain(swapchain);
        ngapi::destroySurface(window, surface);
        ngapi::destroyWindow(window);
        gpuDestroySemaphore(semaphore);
        gpuDestroyQueue(queue);
    } // renderer + fonts freed here, device still alive

    gpuDestroyDevice(device);
    gpuDestroyInstance();
    return 0;
}
