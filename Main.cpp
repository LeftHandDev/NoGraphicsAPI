#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "External/stb_image.h"
#include "External/stb_image_write.h"

#include "Samples/Compute/Compute.h"
#include "Samples/Graphics/Graphics.h"
#include "Samples/Raytracing/Raytracing.h"

#include <iostream>

int main()
{
    std::cout << "Pick a sample to run:\n1. Compute\n2. Graphics\n3. Raytracing\n> ";
    int choice;
    std::cin >> choice;

    switch (choice)
    {    
        case 1:
            computeSample();
            break;

        case 2:
            graphicsSample();
            break;

        case 3:
            raytracingSample();
            break;
        default:
            std::cout << "Invalid choice\n";
            break;
    }

    return 0;
}