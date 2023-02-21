#include <FreeImage.h>
#include <ostream>

__host__ void get_image(FIBITMAP *bitmap, unsigned int* img,unsigned height, unsigned width, unsigned pitch);
__host__ void save_image(FIBITMAP *bitmap, unsigned int* img, const char* PathDest, unsigned height, unsigned width, unsigned pitch);
__device__ int get_id();

