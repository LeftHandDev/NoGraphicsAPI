#define GPU_EXPOSE_INTERNAL
#include "test_common.h"

#include "stb_image.h"
#include "stb_image_write.h"

#include <vulkan/vulkan.h>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>

namespace test
{
    Args parseArgs(int argc, char** argv)
    {
        // The implementation only requests the validation layer when asked
        // (it costs ~10us per recorded command); the tests always want it —
        // validationFailed() is part of every test's pass criteria. parseArgs
        // runs before gpuCreateInstance in every test, so set it here.
#ifdef _WIN32
        _putenv_s("NGAPI_VALIDATION", "1");
#else
        setenv("NGAPI_VALIDATION", "1", 1);
#endif

        Args a;
        for (int i = 1; i < argc; i++)
        {
            std::string s = argv[i];
            if (s == "--generate")
                a.generate = true;
            else if (s == "--threshold" && i + 1 < argc)
                a.threshold = std::atoi(argv[++i]);
            else if (s == "--device" && i + 1 < argc)
                a.device = static_cast<uint32_t>(std::atoi(argv[++i]));
            else if (s == "--frames" && i + 1 < argc)
                a.frames = static_cast<uint32_t>(std::atoi(argv[++i]));
            else
                std::cerr << "warning: ignoring unknown argument '" << s << "'\n";
        }
        return a;
    }

    bool writePng(const std::string& path, const Image& img)
    {
        return stbi_write_png(path.c_str(), img.width, img.height, 4, img.rgba.data(), img.width * 4) != 0;
    }

    bool readPng(const std::string& path, Image& out)
    {
        int w, h, channels;
        stbi_uc* pixels = stbi_load(path.c_str(), &w, &h, &channels, 4);
        if (!pixels)
            return false;
        out.width = w;
        out.height = h;
        out.rgba.assign(pixels, pixels + static_cast<size_t>(w) * h * 4);
        stbi_image_free(pixels);
        return true;
    }

    Image readbackRGBA8(GpuDevice device, GpuQueue queue, GpuTexture texture, uint32_t width, uint32_t height)
    {
        const size_t bytes = static_cast<size_t>(width) * height * 4;
        uint8_t* readback = static_cast<uint8_t*>(gpuMalloc(device, bytes, MEMORY_READBACK));

        auto semaphore = gpuCreateSemaphore(device, 0);
        auto cmd = gpuStartCommandRecording(queue);
        gpuCopyFromTexture(cmd, gpuHostToDevicePointer(device, readback), texture);
        gpuSubmit(queue, Span<GpuCommandBuffer>(&cmd, 1), semaphore, 1);
        gpuWaitSemaphore(semaphore, 1);

        Image img;
        img.width = static_cast<int>(width);
        img.height = static_cast<int>(height);
        img.rgba.assign(readback, readback + bytes);

        gpuDestroySemaphore(semaphore);
        gpuFree(device, readback);
        return img;
    }

    // ---- validation capture -------------------------------------------------
    static bool g_validationFailed = false;
    static VkInstance g_instance = VK_NULL_HANDLE;
    static VkDebugUtilsMessengerEXT g_messenger = VK_NULL_HANDLE;
    static PFN_vkDestroyDebugUtilsMessengerEXT g_destroyMessenger = nullptr;

    static VKAPI_ATTR VkBool32 VKAPI_CALL validationCallback(
        VkDebugUtilsMessageSeverityFlagBitsEXT severity,
        VkDebugUtilsMessageTypeFlagsEXT,
        const VkDebugUtilsMessengerCallbackDataEXT* data,
        void*)
    {
        if (severity & (VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT))
        {
            g_validationFailed = true;
            std::cerr << "[validation] " << (data && data->pMessage ? data->pMessage : "(no message)") << "\n";
        }
        return VK_FALSE;
    }

    void beginValidationCapture()
    {
        g_validationFailed = false;
        g_instance = static_cast<VkInstance>(gpuVulkanInstance());

        auto create = reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(
            vkGetInstanceProcAddr(g_instance, "vkCreateDebugUtilsMessengerEXT"));
        g_destroyMessenger = reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(
            vkGetInstanceProcAddr(g_instance, "vkDestroyDebugUtilsMessengerEXT"));

        if (!create)
        {
            std::cerr << "warning: VK_EXT_debug_utils unavailable; validation is not being captured\n";
            return;
        }

        VkDebugUtilsMessengerCreateInfoEXT info = {};
        info.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
        info.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
        info.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
        info.pfnUserCallback = validationCallback;
        create(g_instance, &info, nullptr, &g_messenger);
    }

    void endValidationCapture()
    {
        if (g_messenger != VK_NULL_HANDLE && g_destroyMessenger != nullptr)
        {
            g_destroyMessenger(g_instance, g_messenger, nullptr);
            g_messenger = VK_NULL_HANDLE;
        }
    }

    bool validationFailed()
    {
        return g_validationFailed;
    }

    // ---- golden generation / comparison ------------------------------------
    int finalize(const Args& args, const std::string& name, const Image& actual)
    {
        const std::string goldenPath = std::string(NGAPI_TEST_REFERENCE_DIR) + "/" + name + ".png";

        if (args.generate)
        {
            std::error_code ec;
            std::filesystem::create_directories(NGAPI_TEST_REFERENCE_DIR, ec);
            if (!writePng(goldenPath, actual))
            {
                std::cerr << "FAIL [" << name << "]: could not write golden " << goldenPath << "\n";
                return 1;
            }
            std::cout << "generated golden " << goldenPath << " (" << actual.width << "x" << actual.height << ")\n";
            return 0;
        }

        Image golden;
        if (!readPng(goldenPath, golden))
        {
            std::cerr << "FAIL [" << name << "]: missing golden " << goldenPath << " (run with --generate)\n";
            return 1;
        }
        if (golden.width != actual.width || golden.height != actual.height)
        {
            std::cerr << "FAIL [" << name << "]: size mismatch (golden " << golden.width << "x" << golden.height
                      << " vs actual " << actual.width << "x" << actual.height << ")\n";
            return 1;
        }

        const size_t pixels = static_cast<size_t>(actual.width) * actual.height;
        int maxDiff = 0;
        size_t badPixels = 0;
        Image diff;
        diff.width = actual.width;
        diff.height = actual.height;
        diff.rgba.assign(actual.rgba.size(), 255);

        for (size_t p = 0; p < pixels; p++)
        {
            bool bad = false;
            for (int c = 0; c < 4; c++)
            {
                int d = std::abs(static_cast<int>(actual.rgba[p * 4 + c]) - static_cast<int>(golden.rgba[p * 4 + c]));
                maxDiff = std::max(maxDiff, d);
                if (d > args.threshold)
                    bad = true;
                if (c < 3)
                    diff.rgba[p * 4 + c] = static_cast<uint8_t>(std::min(255, d * 8)); // amplified for visibility
            }
            if (bad)
                badPixels++;
        }

        if (badPixels == 0)
        {
            std::cout << "PASS [" << name << "]: maxDiff=" << maxDiff << " (threshold=" << args.threshold << ")\n";
            return 0;
        }

        std::cerr << "FAIL [" << name << "]: " << badPixels << "/" << pixels << " pixels exceed threshold "
                  << args.threshold << " (maxDiff=" << maxDiff << ")\n";
        writePng(name + "_actual.png", actual);
        writePng(name + "_diff.png", diff);
        std::cerr << "  wrote " << name << "_actual.png and " << name << "_diff.png\n";
        return 1;
    }
} // namespace test
