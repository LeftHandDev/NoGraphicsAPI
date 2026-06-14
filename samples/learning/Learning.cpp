#include "stb_image.h"
#include "stb_image_write.h"

#include "Learning.h"
#include "Tensor.h"

#include <iostream>
#include <random>
#include <chrono>

struct Pixel
{
    float r, g, b;
};

std::vector<float> encode(int x, int y)
{
    std::vector<float> encoding(256 * 256, 0.f);
    encoding[x * 256 + y] = 1.f;
    return encoding;
}

std::vector<float> encode(int x, int y, int L)
{
    float u = x / 255.0f;
    float v = y / 255.0f;
    std::vector<float> e;
    e.reserve(4 * L);
    for (int k = 0; k < L; k++)
    {
        float f = std::pow(2.0f, static_cast<float>(k)) * 3.14159265f;
        e.push_back(std::sin(f * u));
        e.push_back(std::cos(f * u));
        e.push_back(std::sin(f * v));
        e.push_back(std::cos(f * v));
    }
    return e;
}

void learningSample()
{
    Instance instance;
    auto device = instance.device(1);
    try
    {
        std::vector<std::vector<Pixel>> gt;
        std::vector<float> gt_flat;
        int w, h, c;
        auto ptr = stbi_load("assets/Default.png", &w, &h, &c, 3);
        if (!ptr)
        {
            throw std::runtime_error(std::string("failed to load image: ") + stbi_failure_reason());
        }
        gt.resize(h);
        for (int y = 0; y < h; y++)
        {
            gt[y].resize(w);
            for (int x = 0; x < w; x++)
            {
                size_t idx = (static_cast<size_t>(y) * w + x) * 3;
                gt[y][x].r = ptr[idx + 0] / 255.0f;
                gt[y][x].g = ptr[idx + 1] / 255.0f;
                gt[y][x].b = ptr[idx + 2] / 255.0f;
                gt_flat.push_back(ptr[idx + 0] / 255.0f);
                gt_flat.push_back(ptr[idx + 1] / 255.0f);
                gt_flat.push_back(ptr[idx + 2] / 255.0f);
            }
        }
        stbi_image_free(ptr);

        const unsigned int N = 256 * 256 * 3;
        Network autoencoder(device, { N, 256, N });

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<int> dis(0, 255);

        size_t epochs = 100;
        auto y = device->tensor({ gt_flat });

        for (size_t i = 0; i < epochs; i++)
        {
            auto a = (y + (device->rand(y.shape()) * 2.f - 1.f) * 0.1).detach();
            auto b = (y + (device->rand(y.shape()) * 2.f - 1.f) * 0.1).detach();
            auto za = autoencoder.forward(a);
            auto zb = autoencoder.forward(b);
            auto L = 0.5f * (za.mse(b) + zb.mse(a));
            L.backward();
            autoencoder.train(0.001f);
            device->submit();
            std::cout << /*"MSE " << L << "\t" <<*/ i << "/" << epochs << "\t" << std::endl;
            // stbi_write_hdr("output.exr", 256, 256, 3, za.cpu().data());
        }

        std::cout << std::endl;

        auto a = y + (device->rand(y.shape()) * 2.f - 1.f) * 0.1;
        auto z = autoencoder.forward(a);
        auto hdr = z.cpu();
        stbi_write_hdr("input.exr", 256, 256, 3, a.cpu().data());
        stbi_write_hdr("output.exr", 256, 256, 3, hdr.data());
        stbi_write_hdr("gt.exr", 256, 256, 3, gt_flat.data());
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