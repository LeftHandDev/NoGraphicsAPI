#include "stb_image.h"
#include "stb_image_write.h"

#include "Learning.h"
#include "Tensor.h"

#include <iostream>
#include <random>
#include <chrono>

void learningSample()
{
    Instance instance;
    auto device = instance.device(1);
    try
    {
        std::vector<float> gt;
        int w, h, c;
        auto ptr = stbi_load("assets/Default.png", &w, &h, &c, 3);
        if (!ptr)
        {
            throw std::runtime_error(std::string("failed to load image: ") + stbi_failure_reason());
        }
        for (int y = 0; y < h; y++)
        {
            for (int x = 0; x < w; x++)
            {
                size_t idx = (static_cast<size_t>(y) * w + x) * 3;
                gt.push_back(ptr[idx + 0] / 255.0f);
                gt.push_back(ptr[idx + 1] / 255.0f);
                gt.push_back(ptr[idx + 2] / 255.0f);
            }
        }
        stbi_image_free(ptr);

        const unsigned int N = 256 * 256 * 3;
        Network autoencoder(device, { N, 256, N });

        size_t steps = 100;
        auto y = device->tensor({ gt });

        float noise_ratio = 0.25;

        for (size_t i = 0; i < steps; i++)
        {
            auto a = (y + (device->rand(y.shape()) * 2.f - 1.f) * noise_ratio).detach();
            auto b = (y + (device->rand(y.shape()) * 2.f - 1.f) * noise_ratio).detach();
            auto z = autoencoder.forward(a);
            auto L = z.mse(b);
            L.backward();
            autoencoder.train(0.001f);
            device->submit();
            L.cpu([&](std::vector<float> data)
                  { std::cout << "MSE " << data.front() << "\t" << i << "/" << steps << "\t" << std::endl; });
        }

        std::cout << std::endl;

        auto a = y + (device->rand(y.shape()) * 2.f - 1.f) * noise_ratio;
        auto z = autoencoder.forward(a);
        stbi_write_hdr("input.exr", 256, 256, 3, a.pow(2.2).cpu().data());
        stbi_write_hdr("output.exr", 256, 256, 3, z.pow(2.2).cpu().data());
        stbi_write_hdr("gt.exr", 256, 256, 3, y.pow(2.2).cpu().data());
    }
    catch (const std::exception& e)
    {
        std::cerr << e.what() << '\n';
    }

    return;
}

int main()
{
    learningSample();
    return 0;
}