// Headless MSDF text test: renders Inter at several sizes and colors (plus an
// MTSDF outline) into a capture texture in a single frame, reads it back and
// compares against a golden. Text layout is fixed, so the output is fully
// deterministic.
#include "test_common.h"

#include "MsdfFont.h"
#include "MsdfDemo.h"

#include <iostream>
#include <string>

int main(int argc, char** argv)
{
    test::Args args = test::parseArgs(argc, argv);

    gpuCreateInstance();
    test::beginValidationCapture();

    auto device = gpuCreateDevice(args.device);
    if (!device)
    {
        std::cerr << "FAIL [msdf]: no suitable device at index " << args.device << "\n";
        return 1;
    }

    // Match the windowed sample's resolution so the test renders the same scene
    // with the same area to lay it out in.
    const uint32_t RENDER_W = 1280;
    const uint32_t RENDER_H = 720;

    // Capture target: rendered into directly (color attachment) then read back.
    GpuTextureDesc captureDesc{
        .type = TEXTURE_2D,
        .dimensions = { RENDER_W, RENDER_H, 1 },
        .format = FORMAT_RGBA8_UNORM,
        .usage = static_cast<USAGE_FLAGS>(USAGE_COLOR_ATTACHMENT | USAGE_TRANSFER_SRC | USAGE_TRANSFER_DST | USAGE_SAMPLED)
    };
    GpuTextureSizeAlign captureSizeAlign = gpuTextureSizeAlign(device, captureDesc);
    void* capturePtr = gpuMalloc(device, captureSizeAlign.size, MEMORY_GPU);
    auto capture = gpuCreateTexture(device, captureDesc, capturePtr);

    auto queue = gpuCreateQueue(device);
    auto semaphore = gpuCreateSemaphore(device, 0);

    test::Image actual;
    {
        MsdfTextRenderer renderer(device, FORMAT_RGBA8_UNORM, NGAPI_TEST_SHADER_DIR);
        MsdfFont* inter = renderer.loadFont(std::string(NGAPI_TEST_ASSET_DIR) + "/fonts/Inter/Inter-msdf.json",
                                            std::string(NGAPI_TEST_ASSET_DIR) + "/fonts/Inter/Inter-msdf.png");

        buildMsdfDemoScene(renderer, inter, static_cast<float>(RENDER_W), static_cast<float>(RENDER_H));

        auto cmd = gpuStartCommandRecording(queue);
        renderer.render(cmd, capture, RENDER_W, RENDER_H, LOAD_OP_CLEAR);
        gpuSubmit(queue, Span<GpuCommandBuffer>(&cmd, 1), semaphore, 1);
        gpuWaitSemaphore(semaphore, 1);

        actual = test::readbackRGBA8(device, queue, capture, RENDER_W, RENDER_H);
    } // renderer + font freed while device is alive

    int rc = test::finalize(args, "msdf", actual);

    gpuDestroyTexture(capture);
    gpuFree(device, capturePtr);
    gpuDestroySemaphore(semaphore);
    gpuDestroyQueue(queue);
    gpuDestroyDevice(device);
    test::endValidationCapture();
    gpuDestroyInstance();

    if (test::validationFailed())
    {
        std::cerr << "FAIL [msdf]: Vulkan validation messages were emitted\n";
        rc = 1;
    }
    return rc;
}
