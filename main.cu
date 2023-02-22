#include <string>
#include <stdio.h>
#include "FreeImage.h"
#include "utils.h"

__device__ int get_id()
{
  int size_block = blockDim.x * blockDim.y * blockDim.z;
  int id_block = gridDim.x * gridDim.y * blockIdx.z + gridDim.x * blockIdx.y + blockIdx.x;
  int id_thread = blockDim.x * blockDim.y * threadIdx.z + blockDim.x * threadIdx.y + threadIdx.x;
  int id = id_block * size_block + id_thread;

  return id;
}

__global__ void symetry(unsigned int *d_img, unsigned int *d_tmp, int size)
{
	int id = get_id();
    if (id < size)
	{
		d_img[id*3 + 0] = d_tmp[(size-id)*3 + 0];
	    d_img[id*3 + 1] = d_tmp[(size-id)*3 + 1];
	    d_img[id*3 + 2] = d_tmp[(size-id)*3 + 2];
	}
}

__global__ void grey(unsigned int *d_img, int size)
{
	int id = get_id();
    if (id < size)
	{
		int grey = d_img[id*3+0]*0.299 + d_img[id*3+1]*0.587 + d_img[id*3+2]*0.114;
		d_img[id*3 + 0] = grey;
	    d_img[id*3 + 1] = grey;
	    d_img[id*3 + 2] = grey;
	}
}

__global__ void merge_image(unsigned int* d_img, unsigned int* d_tl, unsigned int* d_tr, unsigned int* d_bl, unsigned int* d_br, unsigned int height, unsigned int width, unsigned int smallheight, unsigned int smallwidth)
{
	int x,y;
	x = blockIdx.x * blockDim.x + threadIdx.x;
	y = blockIdx.y * blockDim.y + threadIdx.y;
	
	if(blockIdx.y!=0)
		printf("Y : %d %d %d %d\n",blockIdx.y,blockDim.y,threadIdx.y,y);
	
	if(x<width && y<height)
	{
		if(x<smallwidth && y<smallheight)
		{
			//Top left
			d_img[(x * width + y)*3 + 0] = d_tl[(x * width + y)*3+0];
			d_img[(x * width + y)*3 + 1] = d_tl[(x * width + y)*3+1];
			d_img[(x * width + y)*3 + 2] = d_tl[(x * width + y)*3+2];
		}
		if(x>smallwidth && y<smallheight)
		{
			//Top right
			d_img[(x * width + y)*3 + 0] = d_tr[((x * width + y) - width/2)*3 + 0];
			d_img[(x * width + y)*3 + 1] = d_tr[((x * width + y) - width/2)*3 + 1];
			d_img[(x * width + y)*3 + 2] = d_tr[((x * width + y) - width/2)*3 + 2];
		}
		if(x<smallwidth && y>smallheight)
		{
			//Bot left
			d_img[(x * width + y)*3 + 0] = d_bl[((x * width + y) - height/2)*3 + 0];
			d_img[(x * width + y)*3 + 1] = d_bl[((x * width + y) - height/2)*3 + 0];
			d_img[(x * width + y)*3 + 2] = d_bl[((x * width + y) - height/2)*3 + 0];
		}
		if(x>smallwidth && y>smallheight)
		{
			//Bot right
			d_img[(x * width + y)*3 + 0] = 100;//d_br[];
			d_img[(x * width + y)*3 + 1] = 100;//d_br[];
			d_img[(x * width + y)*3 + 2] = 100;//d_br[];
		}
	}
}

void popart(FIBITMAP * bitmap, unsigned int *img, unsigned height, unsigned width, unsigned pitch)
{
  FIBITMAP * split =  FreeImage_Rescale(bitmap,width/2,height/2, FILTER_BOX);
  
  unsigned smallpitch  = FreeImage_GetPitch(split);
  unsigned smallwidth  = FreeImage_GetWidth(split);
  unsigned smallheight = FreeImage_GetHeight(split);

  int nbthread = 32;
  int grid_x = smallwidth / nbthread + 1;
  int grid_y = smallheight / nbthread + 1;
  dim3 smallgrid(grid_x, grid_y, 1);
  dim3 smallblock(nbthread, nbthread, 1);

  unsigned int *d_img = NULL;
  unsigned int *d_tl = NULL;
  unsigned int *d_bl = NULL;
  unsigned int *d_tr = NULL;
  unsigned int *d_br = NULL;

 unsigned int *h_img = (unsigned int*) malloc(sizeof(unsigned int) * 3 * (smallwidth) * (smallheight));
 cudaMalloc(&d_img,sizeof(unsigned int) * 3 * (width) * (height));
 cudaMalloc(&d_tl,sizeof(unsigned int) * 3 * (smallwidth) * (smallheight));
 cudaMalloc(&d_bl,sizeof(unsigned int) * 3 * (smallwidth) * (smallheight));
 cudaMalloc(&d_tr,sizeof(unsigned int) * 3 * (smallwidth) * (smallheight));
 cudaMalloc(&d_br,sizeof(unsigned int) * 3 * (smallwidth) * (smallheight));

 get_image(split,h_img,smallheight,smallwidth,smallpitch);
 
 cudaMemcpy(d_tl, h_img, 3 * (smallwidth) * (smallheight) * sizeof(unsigned int),cudaMemcpyHostToDevice);
 //Mettre rouge
 cudaMemcpy(d_bl, h_img, 3 * (smallwidth) * (smallheight) * sizeof(unsigned int),cudaMemcpyHostToDevice);
 //Mettre bleu
 cudaMemcpy(d_tr, h_img, 3 * (smallwidth) * (smallheight) * sizeof(unsigned int),cudaMemcpyHostToDevice);
 //Mettre vert
 cudaMemcpy(d_br, h_img, 3 * (smallwidth) * (smallheight) * sizeof(unsigned int),cudaMemcpyHostToDevice);
 
 //grey<<<smallgrid,smallblock>>>(d_img, height/2 * width/2);

 //fusionner image
 nbthread = 32;
 grid_x = width / nbthread + 1;
 grid_y = height / nbthread + 1;
 printf("grid_x: %d grid_y: %d\n",grid_x,grid_y);
 dim3 grid(grid_x, grid_y,1);
 dim3 block(nbthread, nbthread,1);

 merge_image<<<grid,block>>>(d_img, d_tl, d_tr, d_bl, d_br, height, width, smallheight, smallwidth);
 cudaDeviceSynchronize();
 
 cudaMemcpy(img,d_img,3*width*height*sizeof(unsigned int), cudaMemcpyDeviceToHost);

 cudaFree(d_img);
 cudaFree(d_tl);
 cudaFree(d_bl);
 cudaFree(d_tr);
 cudaFree(d_br);

 free(h_img);
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
 	cudaMalloc(&d_tmp,sizeof(unsigned int) * 3 * width * height);

 	get_image(bitmap,h_img,height,width,pitch);

 	cudaMemcpy(d_img, h_img, 3 * width * height * sizeof(unsigned int),cudaMemcpyHostToDevice);
 	cudaMemcpy(d_tmp, h_img, 3 * width * height * sizeof(unsigned int),cudaMemcpyHostToDevice);


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
				cudaMemcpy(h_img,d_img , 3 * width * height * sizeof(unsigned int),cudaMemcpyDeviceToHost);
				continue;
			}
			else if(!strcmp(argv[i],"green"))
			{
				printf("Saturate Green not yet implemented!\n");
				cudaDeviceSynchronize();
				cudaMemcpy(h_img,d_img , 3 * width * height * sizeof(unsigned int),cudaMemcpyDeviceToHost);
				continue;
			}
			else if(!strcmp(argv[i],"blue"))
			{
				printf("Saturate Blue not yet implemented!\n");
				cudaDeviceSynchronize();
				cudaMemcpy(h_img,d_img , 3 * width * height * sizeof(unsigned int),cudaMemcpyDeviceToHost);
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
			//printf("Miror not yet implemented!\n");
			symetry<<<grid,block>>>(d_img,d_tmp, height * width);
			cudaDeviceSynchronize();
			cudaMemcpy(h_img,d_img , 3 * width * height * sizeof(unsigned int),cudaMemcpyDeviceToHost);
			continue;
		}

		if(!strcmp(argv[i],"--blur"))
		{
			printf("Blur not yet implemented!\n");
			cudaDeviceSynchronize();
			cudaMemcpy(h_img,d_img , 3 * width * height * sizeof(unsigned int),cudaMemcpyDeviceToHost);
			continue;
		}

		if(!strcmp(argv[i],"--grey"))
		{
			grey<<<grid,block>>>(d_img, height * width);
			cudaDeviceSynchronize();
			cudaMemcpy(h_img,d_img , 3 * width * height * sizeof(unsigned int),cudaMemcpyDeviceToHost);
			continue;
		}

		if(!strcmp(argv[i],"--sobel"))
		{
			printf("Sobel not yet implemented!\n");
			cudaDeviceSynchronize();
			cudaMemcpy(h_img,d_img , 3 * width * height * sizeof(unsigned int),cudaMemcpyDeviceToHost);
			continue;
		}

		if(!strcmp(argv[i],"--popart"))
		{
			popart(bitmap, h_img,height,width,pitch);
			continue;
		}
	}
	cudaDeviceSynchronize();
	printf("Saving image...\n");
	save_image(bitmap,h_img,out_name.c_str(),height,width,pitch);

	FreeImage_DeInitialise();
	cudaFree(d_img);
	cudaFree(d_tmp);
	free(h_img);
	exit(0);
}