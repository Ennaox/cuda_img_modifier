#include <string>
#include <ostream>
#include <FreeImage.h>
#include "utils.h"


__global__ void saturate_green(unsigned int *d_img, int size)
{
  int id = get_id();
 
  if (id < size)
  {
      d_img[id * 3 + 1] = 0xFF;
  }
}

__global__ void saturate_red(unsigned int *d_img, int size)
{
  int id = get_id();
 
  if (id < size)
  {
      d_img[id * 3 + 0] = 0xFF;
  }
}

__global__ void saturate_blue(unsigned int *d_img, int size)
{
  int id = get_id();
 
  if (id < size)
  {
      d_img[id * 3 + 2] = 0xFF;
  }
}

int main(int argc, char** argv)
{
	if(argc <= 1)
	{
		printf("Error: need image to edit\n");
		exit(1);
	}
	bool out_file = false;
	std::string out_name = "out.png";
	std::string file_name = argv[1];
	FreeImage_Initialise();

 	FREE_IMAGE_FORMAT fif = FreeImage_GetFIFFromFilename(file_name.c_str());

 	FIBITMAP *bitmap = FreeImage_Load(fif,file_name.c_str(),0);

 	if(!bitmap)
 	{
 		printf("Image %s can't be opened\n",file_name.c_str());
 		exit(1);
 	}

 	unsigned width  = FreeImage_GetWidth(bitmap);
	unsigned height = FreeImage_GetHeight(bitmap);
	unsigned pitch  = FreeImage_GetPitch(bitmap);

	fprintf(stderr, "Processing Image of size %d x %d\n", width, height);

	unsigned int *h_img = (unsigned int*) malloc(sizeof(unsigned int) * 3 * width * height);
 	unsigned int *d_img = NULL;
 	unsigned int *d_tmp = NULL;

 	cudaMalloc(&d_img,sizeof(unsigned int) * 3 * width * height);
 	cudaMalloc(&d_img,sizeof(unsigned int) * 3 * width * height);

 	get_image(bitmap,h_img,height,width,pitch);

 	cudaMemcpy(d_img, h_img, 3 * width * height * sizeof(unsigned int),cudaMemcpyHostToDevice);
 	cudaMemcpy(d_tmp, h_img, 3 * width * height * sizeof(unsigned int),cudaMemcpyHostToDevice);

 	saturate_red<<<grid,block>>>(d_img, height * width);
 	saturate_green<<<grid,block>>>(d_img, height * width);
 	saturate_blue<<<grid,block>>>(d_img, height * width);

	int nbthread = 32;
	int grid_x = width / nbthread + 1;
	int grid_y = height / nbthread + 1;
	dim3 grid(grid_x, grid_y, 1);
	dim3 block(nbthread, nbthread, 1);

	for(int i = 2; i<argc;i++)
	{
		if(!strcmp(argv[i],"-o"))
		{
			if(out_file)
			{
				printf("Error: output name was already given");
				exit(3);
			}

			i++;
			if(argv[i][0] == '-')
			{
				printf("Error: \"%s\" is not a valide name\n",argv[i]);
				exit(2);
			}
			out_name = std::string(argv[i]);
			continue;
		}

		if(!strcmp(argv[i],"--saturate"))
		{
			i++;
			if(!strcmp(argv[i],"red"))
			{
				printf("Saturate Red not yet implemented!\n");
				cudaDeviceSynchronize();
				continue;
			}
			else if(!strcmp(argv[i],"green"))
			{
				printf("Saturate Green not yet implemented!\n");
				cudaDeviceSynchronize();
				continue;
			}
			else if(!strcmp(argv[i],"blue"))
			{
				printf("Saturate Blue not yet implemented!\n");
				cudaDeviceSynchronize();
				continue;
			}
			else
			{
				printf("Error: \"%s\" is not a valide color\n",argv[i]);
				exit(2);
			}
			continue;
		}

		if(!strcmp(argv[i],"--miror"))
		{
			printf("Miror not yet implemented!\n");
			continue;
		}

		if(!strcmp(argv[i],"--blur"))
		{
			printf("Blur not yet implemented!\n");
			continue;
		}

		if(!strcmp(argv[i],"--grey"))
		{
			printf("Grey not yet implemented!\n");
			continue;
		}

		if(!strcmp(argv[i],"--sobel"))
		{
			printf("Sobel not yet implemented!\n");
			continue;
		}

		if(!strcmp(argv[i],"--popart"))
		{
			printf("Popart not yet implemented!\n");
			continue;
		}
	}

	save_image(bitmap,d_img,out_name.c_str(),height,width,pitch);

	FreeImage_DeInitialise();
	cudaFree(d_img);
	cudaFree(d_tmp);
	free(h_img);
	exit(0);
}