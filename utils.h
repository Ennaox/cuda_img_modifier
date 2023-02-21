#include <FreeImage.h>

__host__ get_image(FIBITMAP bitmap, unsigned int* img,unsigned height, unsigned width, unsigned pitch);
__host__ save_image(FIBITMAP bitmap, unsigned int* img, char** PathDest, unsigned height, unsigned width, unsigned pitch);
__device__ int get_id();

