#ifndef NGAPI_TEST_COMMON_H
#define NGAPI_TEST_COMMON_H

#include "NoGraphicsAPI.h"

#include <cstdint>
#include <string>
#include <vector>

// Shared helpers for the headless rendering tests. Each test renders a fixed,
// deterministic sequence of frames into an RGBA8 capture texture, reads it back,
// and either writes a golden PNG (--generate) or threshold-compares against the
// committed one. Goldens live in NGAPI_TEST_REFERENCE_DIR (the source tree) so they
// are easy to view and review.
namespace test
{
    struct Args
    {
        bool generate = false; // --generate : (re)write the golden instead of comparing
        int threshold = 4;     // --threshold N : max allowed per-channel abs difference (0-255)
        uint32_t device = 0;   // --device N : GPU index to render on
        uint32_t frames = 10;  // --frames N : number of frames to render
    };

    Args parseArgs(int argc, char** argv);

    struct Image
    {
        int width = 0;
        int height = 0;
        std::vector<uint8_t> rgba; // width * height * 4, tightly packed RGBA
    };

    bool writePng(const std::string& path, const Image& img);
    bool readPng(const std::string& path, Image& out);

    // Validation capture: install a debug messenger that records any Vulkan
    // validation/performance message of WARNING or ERROR severity. A test fails
    // (and refuses to write a golden) if anything was recorded.
    //   beginValidationCapture()  -- call right after gpuCreateInstance()
    //   endValidationCapture()    -- call after gpuDestroyDevice(), before gpuDestroyInstance()
    //   validationFailed()        -- query (stays valid after endValidationCapture)
    void beginValidationCapture();
    void endValidationCapture();
    bool validationFailed();

    // Copies an RGBA8_UNORM texture back to CPU memory (texture must allow TRANSFER_SRC).
    Image readbackRGBA8(GpuDevice device, GpuQueue queue, GpuTexture texture, uint32_t width, uint32_t height);

    // Generate-or-compare against tests/reference/<name>.png. Returns a process exit
    // code: 0 = pass (or golden generated), non-zero = mismatch / missing golden / error.
    // On mismatch, writes <name>_actual.png and <name>_diff.png in the current directory.
    int finalize(const Args& args, const std::string& name, const Image& actual);
} // namespace test

#endif // NGAPI_TEST_COMMON_H
