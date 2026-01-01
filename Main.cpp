#define STB_IMAGE_IMPLEMENTATION
#include "External/stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "External/stb_image_write.h"

#include "Samples/Compute/Compute.h"
#include "Samples/Graphics/Graphics.h"

int main()
{
    // computeSample();
    graphicsSample();

    return 0;
}