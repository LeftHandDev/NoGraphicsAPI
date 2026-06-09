#include "../../External/stb_image.h"
#include "../../External/stb_image_write.h"
#include "Learning.h"
#include "Tensor.h"

#include <iostream>

void learningSample()
{
    Instance instance;
    auto device = instance.device();

    try
    {
        Network mlp(device, { 2, 3, 3, 2 });

        auto x = device->rand({ 2 });

        std::cout << mlp.forward(x) << std::endl;
    }
    catch (const std::exception& e)
    {
        std::cerr << e.what() << '\n';
    }

    return;
}