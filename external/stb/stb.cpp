// Single translation unit that compiles the stb implementations once, so the
// rest of the project only includes the (declaration-only) headers.
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"
