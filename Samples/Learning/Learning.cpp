#include "Learning.h"
#include "Tensor.h"

#include <iostream>

void learningSample()
{
    instance instance;
    auto device = instance.device();

    auto x = device->tensor({1.f, 2.f, 3.f, 4.f}, {2, 2});
    auto y = device->tensor({4.f, 3.f, 2.f, 1.f}, {2, 2});
    std::cout << x << std::endl
              << y << std::endl;

    auto z = x + y;

    std::cout << z << std::endl;

    // std::cout << x << std::endl
    //           << std::endl
    //           << x.mT() << std::endl;

    // network mlp(device, {2, 3, 2});

    // auto x = device->tensor({1.f, 1.f});
    // auto y = mlp.forward(std::move(x));

    // std::cout << y << std::endl;

    // x = device->tensor({1.f, 1.f});
    // y = mlp.forward(std::move(x));

    // std::cout << y << std::endl;

    return;
}