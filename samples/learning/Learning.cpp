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
    std::vector<float> hdr(256 * 256 * 3, 0.f);

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

        const int L = 10;
        const int D = 4 * L; // Fourier encoding dimension: sin/cos for u and v

        // Precompute the Fourier encoding for every (u, v) pair once
        std::vector<float> encoding_table(256 * 256 * D);
        for (int u = 0; u < 256; u++)
        {
            for (int v = 0; v < 256; v++)
            {
                auto e = encode(u, v, L);
                memcpy(&encoding_table[(u * 256 + v) * D], e.data(), D * sizeof(float));
            }
        }

        Network mlp(device, { D, 256, 256, 256, 3 });

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<int> dis(0, 255);

        size_t epochs = 1024 * 1000;
        unsigned int batches = 1024;

        using namespace std::chrono;
        double acc_data = 0, acc_fwd = 0, acc_bwd = 0, acc_submit = 0;
        std::vector<float> row(batches * D);
        std::vector<float> gt_row(3 * batches);

        for (size_t i = 0; i < epochs; i++)
        {
            for (size_t j = 0; j < batches; j++)
            {
                int u = dis(gen);
                int v = dis(gen);
                memcpy(&row[j * D], &encoding_table[(u * 256 + v) * D], D * sizeof(float));
                gt_row[j * 3 + 0] = gt[u][v].r;
                gt_row[j * 3 + 1] = gt[u][v].g;
                gt_row[j * 3 + 2] = gt[u][v].b;
            }
            auto y = device->tensor(gt_row, { batches, 3 });

            auto pred = mlp.forward(device->tensor(row, { batches, D }));
            auto l = ((pred - y) * (pred - y)).sum() / static_cast<float>(batches);
            l.backward();
            mlp.train(epochs > 500 ? 0.001f : 0.01f);
            device->submit();
            if (i % 1000 == 0)
            {
                std::cout << "\rMSE " << l << "\t" << i << "/" << epochs << "\t" << std::flush;
            }
        }

        std::cout << std::endl;

        for (int i = 0; i < 256; i++)
        {
            for (int j = 0; j < 256; j += 64)
            {
                std::vector<float> uvs;
                for (size_t p = 0; p < 64; p++)
                {
                    auto pixel = encode(i, j + p, L);
                    uvs.insert(uvs.end(), pixel.begin(), pixel.end());
                }
                auto rgb = mlp.forward(device->tensor(uvs, { 64, D })).cpu();

                for (size_t p = 0; p < 64; p++)
                {
                    hdr[(i * 256 + j + p) * 3 + 0] = rgb[0 + p * 3];
                    hdr[(i * 256 + j + p) * 3 + 1] = rgb[1 + p * 3];
                    hdr[(i * 256 + j + p) * 3 + 2] = rgb[2 + p * 3];
                }
            }
            std::cout << "\r" << i << std::flush;
        }

        stbi_write_hdr("output.exr", 256, 256, 3, hdr.data());
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