#include "../../External/stb_image.h"
#include "../../External/stb_image_write.h"
#include "Learning.h"
#include "Tensor.h"

#include <iostream>

void learningSample()
{
    instance instance;
    auto device = instance.device();
    std::vector<float> hdr(256 * 256, 0.f);

    try
    {
        network mlp(device, {2, 256, 256, 3});

        for (size_t i = 0; i < 256; i++)
        {
            for (size_t j = 0; j < 256; j++)
            {
                auto uv = device->tensor({i / 256.f, j / 256.f});
                auto rgb = mlp.forward(std::move(uv)).cpu();
                hdr.push_back(rgb[0]);
                hdr.push_back(rgb[1]);
                hdr.push_back(rgb[2]);
            }
        }

        stbi_write_hdr("C:\\Users\\natha\\source\\repos\\NoGraphicsAPI\\test_output.exr", 256, 256, 3, hdr.data());

        auto x = device->rand({2});
        auto y = mlp.forward(std::move(x));

        std::cout << y << std::endl;

        // x = device->tensor({1.f, 1.f});
        // y = mlp.forward(std::move(x));

        // std::cout << y << std::endl;
    }
    catch (const std::exception &e)
    {
        std::cerr << e.what() << '\n';
    }

    return;
}