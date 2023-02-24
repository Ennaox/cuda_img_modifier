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
	
	if(x<width && y<height)
	{
		if(x<smallwidth && y>smallheight)
		{
			//Top left
			d_img[(y * width + x)*3 + 0] = d_tl[(y * width/2 + x+width)*3+0];
			d_img[(y * width + x)*3 + 1] = d_tl[(y * width/2 + x+width)*3+1];
			d_img[(y * width + x)*3 + 2] = d_tl[(y * width/2 + x+width)*3+2];
			//d_img[(y * width + x)*3 + 0] = 255;
			//d_img[(y * width + x)*3 + 1] = 0;
			//d_img[(y * width + x)*3 + 2] = 0;
		}
		if(x>smallwidth && y>smallheight)
		{
			//Top right
			//d_img[(x * width + y)*3 + 0] = d_tr[((x * width + y) - width/2)*3 + 0];
			//d_img[(x * width + y)*3 + 1] = d_tr[((x * width + y) - width/2)*3 + 1];
			//d_img[(x * width + y)*3 + 2] = d_tr[((x * width + y) - width/2)*3 + 2];
			d_img[(y * width + x)*3 + 0] = 0;
			d_img[(y * width + x)*3 + 1] = 255;
			d_img[(y * width + x)*3 + 2] = 0;
		}
		if(x<smallwidth && y<smallheight)
		{
			//Bot left
			//d_img[(x * width + y)*3 + 0] = d_bl[((x * width + y) - height/2)*3 + 0];
			//d_img[(x * width + y)*3 + 1] = d_bl[((x * width + y) - height/2)*3 + 0];
			//d_img[(x * width + y)*3 + 2] = d_bl[((x * width + y) - height/2)*3 + 0];
			d_img[(y * width + x)*3 + 0] = 0;
			d_img[(y * width + x)*3 + 1] = 0;
			d_img[(y * width + x)*3 + 2] = 255;
		}
		if(x>smallwidth && y<smallheight)
		{
			//Bot right
			d_img[(y * width + x)*3 + 0] = 100;//d_br[];
			d_img[(y * width + x)*3 + 1] = 100;//d_br[];
			d_img[(y * width + x)*3 + 2] = 100;//d_br[];
		}
	}
}

void cpu_merge_image(unsigned int* img, unsigned int* bl, unsigned int* br, unsigned int* tl,unsigned int*  tr, unsigned int height, unsigned int width)
{
	for(size_t x = 0; x<width/2; x++)
	{
		for(size_t y = 0; y<height/2; y++)
		{
			img[(y * width + x)*3 + 0] = bl[(y * width/2 + x)*3+0];
			img[(y * width + x)*3 + 1] = bl[(y * width/2 + x)*3+1];
			img[(y * width + x)*3 + 2] = bl[(y * width/2 + x)*3+2];
		}
	}

	for(size_t x = 0; x<width/2; x++)
	{
		for(size_t y = 0; y<height/2; y++)
		{
			img[((y+ height/2)  * width + x)*3 + 0] = tl[(y * width/2 + x)*3+0];
			img[((y + height/2) * width + x)*3 + 1] = tl[(y * width/2 + x)*3+1];
			img[((y + height/2) * width + x)*3 + 2] = tl[(y * width/2 + x)*3+2];
		}
	}

	for(size_t x = 0; x<width/2; x++)
	{
		for(size_t y = 0; y<height/2; y++)
		{
			img[((y + height/2) * width + x + width/2)*3 + 0] = tr[(y * width/2 + x)*3+0];
			img[((y + height/2) * width + x + width/2)*3 + 1] = tr[(y * width/2 + x)*3+1];
			img[((y + height/2) * width + x + width/2)*3 + 2] = tr[(y * width/2 + x)*3+2];
		}
	}	

	for(size_t x = 0; x<width/2; x++)
	{
		for(size_t y = 0; y<height/2; y++)
		{
			img[(y  * width + x + width/2)*3 + 0] = br[(y * width/2 + x)*3+0];
			img[(y * width + x + width/2)*3 + 1] = br[(y * width/2 + x)*3+1];
			img[(y  * width + x + width/2)*3 + 2] = br[(y * width/2 + x)*3+2];
		}
	}	
}

__global__ void saturate_green(unsigned int *d_img, int size)
{
  int id = get_id();
 
  if (id < size)
  {
      d_img[id*3 + 0] = 0xFF - d_img[id*3 + 0];
      d_img[id*3 + 1] = 0xFF / 2;
      d_img[id*3 + 2] /= 4;
  }
}

__global__ void saturate_red(unsigned int *d_img, int size)
{
  int id = get_id();
 
  if (id < size)
  {
      d_img[id*3 + 0] = 0xFF / 2;
      d_img[id*3 + 1] /= 2;
      d_img[id*3 + 2] /= 2;
  }
}

__global__ void saturate_blue(unsigned int *d_img, int size)
{
  int id = get_id();
 
  if (id < size)
  {
      d_img[id * 3 + 0] /= 2;
      d_img[id * 3 + 1] /= 4;
      d_img[id * 3 + 2] = 0xFF / 1.5;
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
 
 unsigned int *h_tl; cudaMallocHost(&h_tl,sizeof(unsigned int) * 3 * (smallwidth) * (smallheight));
 unsigned int *h_bl; cudaMallocHost(&h_bl,sizeof(unsigned int) * 3 * (smallwidth) * (smallheight));
 unsigned int *h_tr; cudaMallocHost(&h_tr,sizeof(unsigned int) * 3 * (smallwidth) * (smallheight));
 unsigned int *h_br; cudaMallocHost(&h_br,sizeof(unsigned int) * 3 * (smallwidth) * (smallheight));
 
 cudaMalloc(&d_img,sizeof(unsigned int) * 3 * (width) * (height));
 cudaMalloc(&d_tl,sizeof(unsigned int) * 3 * (smallwidth) * (smallheight));
 cudaMalloc(&d_bl,sizeof(unsigned int) * 3 * (smallwidth) * (smallheight));
 cudaMalloc(&d_tr,sizeof(unsigned int) * 3 * (smallwidth) * (smallheight));
 cudaMalloc(&d_br,sizeof(unsigned int) * 3 * (smallwidth) * (smallheight));

 get_image(split,h_img,smallheight,smallwidth,smallpitch);

 cudaStream_t stream[4];
 for(int i = 0; i<4; i++)
 {
 	cudaStreamCreate(&stream[i]);
 }

 cudaMemcpyAsync(d_tl, h_img, 3 * (smallwidth) * (smallheight) * sizeof(unsigned int),cudaMemcpyHostToDevice,stream[0]);
 saturate_red<<<smallgrid,smallblock,0,stream[0]>>>(d_tl, smallheight * smallwidth);
 cudaMemcpyAsync(h_tl, d_tl, 3 * (smallwidth) * (smallheight) * sizeof(unsigned int),cudaMemcpyDeviceToHost,stream[0]);
 

 cudaMemcpyAsync(d_bl, h_img, 3 * (smallwidth) * (smallheight) * sizeof(unsigned int),cudaMemcpyHostToDevice,stream[1]);
 saturate_green<<<smallgrid,smallblock,0,stream[1]>>>(d_bl, smallheight * smallwidth);
 cudaMemcpyAsync(h_bl, d_bl, 3 * (smallwidth) * (smallheight) * sizeof(unsigned int),cudaMemcpyDeviceToHost,stream[1]);

 cudaMemcpyAsync(d_tr, h_img, 3 * (smallwidth) * (smallheight) * sizeof(unsigned int),cudaMemcpyHostToDevice,stream[2]);
 saturate_blue<<<smallgrid,smallblock,0,stream[2]>>>(d_tr, smallheight * smallwidth);
 cudaMemcpyAsync(h_tr, d_tr, 3 * (smallwidth) * (smallheight) * sizeof(unsigned int),cudaMemcpyDeviceToHost,stream[2]);

 cudaMemcpyAsync(d_br, h_img, 3 * (smallwidth) * (smallheight) * sizeof(unsigned int),cudaMemcpyHostToDevice,stream[3]);
 grey<<<smallgrid,smallblock,0,stream[3]>>>(d_br, smallheight * smallwidth);
 cudaMemcpyAsync(h_br, d_br, 3 * (smallwidth) * (smallheight) * sizeof(unsigned int),cudaMemcpyDeviceToHost,stream[3]);

 cudaError_t cudaerr = cudaDeviceSynchronize();
				  if (cudaerr != cudaSuccess)
				    printf("kernel launch failed with error \"%s\".\n",cudaGetErrorString(cudaerr));

 /*
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
 */

 cpu_merge_image(img,h_bl,h_br,h_tl,h_tr,height,width);
 cudaFree(d_img);
 cudaFree(d_tl);
 cudaFree(d_bl);
 cudaFree(d_tr);
 cudaFree(d_br);

 free(h_img);
}

__global__ void sobel(unsigned int *d_img, unsigned int *d_tmp, unsigned height, unsigned width,unsigned size)
{
	int x,y;
	x = blockIdx.x * blockDim.x + threadIdx.x;
	y = blockIdx.y * blockDim.y + threadIdx.y;

	unsigned int gx_x = 0;
	unsigned int gx_y = 0;
	unsigned int gx_z = 0;

	unsigned int gy_x = 0;
	unsigned int gy_y = 0;
	unsigned int gy_z = 0;

	if(x*y<size && x>0 && y>0 && x<size-1 && y<size-1)
	{	
		//Horizontal convolution
		//Compute first column
		gx_x += d_tmp[((y-1) * width + x-1)*3 + 0];
		gx_y += d_tmp[((y-1) * width + x-1)*3 + 0];
		gx_z += d_tmp[((y-1) * width + x-1)*3 + 0];

		gx_x +=2* d_tmp[(y * width + x-1)*3 + 0];
		gx_y +=2* d_tmp[(y * width + x-1)*3 + 0];
		gx_z +=2* d_tmp[(y * width + x-1)*3 + 0];

		gx_x += d_tmp[((y+1) * width + x-1)*3 + 0];
		gx_y += d_tmp[((y+1) * width + x-1)*3 + 0];
		gx_z += d_tmp[((y+1) * width + x-1)*3 + 0];

		//Compute
		//Compute third column
		gx_x -= d_tmp[((y-1) * width + x+1)*3 + 0];
		gx_y -= d_tmp[((y-1) * width + x+1)*3 + 0];
		gx_z -= d_tmp[((y-1) * width + x+1)*3 + 0];

		gx_x -=2* d_tmp[(y * width + x+1)*3 + 0];
		gx_y -=2* d_tmp[(y * width + x+1)*3 + 0];
		gx_z -=2* d_tmp[(y * width + x+1)*3 + 0];

		gx_x -= d_tmp[((y+1) * width + x+1)*3 + 0];
		gx_y -= d_tmp[((y+1) * width + x+1)*3 + 0];
		gx_z -= d_tmp[((y+1) * width + x+1)*3 + 0];


		//Vertical convolution
		//Compute first line
		gy_x += d_tmp[((y+1) * width + x-1)*3 + 0];
		gy_y += d_tmp[((y+1) * width + x-1)*3 + 0];
		gy_z += d_tmp[((y+1) * width + x-1)*3 + 0];

		gy_x +=2* d_tmp[((y+1) * width + x)*3 + 0];
		gy_y +=2* d_tmp[((y+1) * width + x)*3 + 0];
		gy_z +=2* d_tmp[((y+1) * width + x)*3 + 0];

		gy_x += d_tmp[((y+1) * width + x+1)*3 + 0];
		gy_y += d_tmp[((y+1) * width + x+1)*3 + 0];
		gy_z += d_tmp[((y+1) * width + x+1)*3 + 0];

		//Compute
		//Compute third column
		gy_x -= d_tmp[((y-1) * width + x-1)*3 + 0];
		gy_y -= d_tmp[((y-1) * width + x-1)*3 + 0];
		gy_z -= d_tmp[((y-1) * width + x-1)*3 + 0];

		gy_x -=2* d_tmp[((y-1) * width + x)*3 + 0];
		gy_y -=2* d_tmp[((y-1) * width + x)*3 + 0];
		gy_z -=2* d_tmp[((y-1) * width + x)*3 + 0];

		gy_x -= d_tmp[((y-1) * width + x+1)*3 + 0];
		gy_y -= d_tmp[((y-1) * width + x+1)*3 + 0];
		gy_z -= d_tmp[((y-1) * width + x+1)*3 + 0];

		if((unsigned char)sqrt((float)(gx_x * gx_x + gy_x * gy_x))<127)
		{
			d_img[(y * width + x)*3 + 0] = 0;
			d_img[(y * width + x)*3 + 1] = 0;
			d_img[(y * width + x)*3 + 2] = 0;
		}
		else
		{
			d_img[(y * width + x)*3 + 0] = 255;
			d_img[(y * width + x)*3 + 1] = 255;
			d_img[(y * width + x)*3 + 2] = 255;	
		}
	}
}

__global__ void negatif(unsigned int *d_img, int size)
{
	int id = get_id();
 
  if (id < size)
  {
      d_img[id * 3 + 0] = 0xFF - d_img[id * 3 + 0];
      d_img[id * 3 + 1] = 0xFF - d_img[id * 3 + 1];
      d_img[id * 3 + 2] = 0xFF - d_img[id * 3 + 2];
  }
}

__global__ void blur(unsigned int *d_img, unsigned int *d_tmp, unsigned height, unsigned width,unsigned size)
{
	int id = get_id();
	//If to prevent 
	if(id < size && id % width != 0 && (id + 1) % width != 0 && (id / width) % height != 0 && ((id + 1) / width) % height != 0)
	{
		unsigned int mean_x = d_img[id * 3 + 0];
		unsigned int mean_y = d_img[id * 3 + 1];
		unsigned int mean_z = d_img[id * 3 + 2];

		mean_x += d_tmp[(id-1) * 3 + 0];
		mean_y += d_tmp[(id-1) * 3 + 1];
		mean_z += d_tmp[(id-1) * 3 + 2];

		mean_x += d_tmp[(id+1) * 3 + 0];
		mean_y += d_tmp[(id+1) * 3 + 1];
		mean_z += d_tmp[(id+1) * 3 + 2];

		mean_x += d_tmp[(id-width) * 3 + 0];
		mean_y += d_tmp[(id-width) * 3 + 1];
		mean_z += d_tmp[(id-width) * 3 + 2];

		mean_x += d_tmp[(id+width) * 3 + 0];
		mean_y += d_tmp[(id+width) * 3 + 1];
		mean_z += d_tmp[(id+width) * 3 + 2];

		d_img[id * 3 + 0] = mean_x / (unsigned char)5;
    d_img[id * 3 + 1] = mean_y / (unsigned char)5;
    d_img[id * 3 + 2] = mean_z / (unsigned char)5;
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
 	cudaMalloc(&d_tmp,sizeof(unsigned int) * 3 * width * height);

 	get_image(bitmap,h_img,height,width,pitch);

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
				cudaMemcpy(d_img, h_img, 3 * width * height * sizeof(unsigned int),cudaMemcpyHostToDevice);
				saturate_red<<<grid,block>>>(d_img, height * width);
				cudaError_t cudaerr = cudaDeviceSynchronize();
				  if (cudaerr != cudaSuccess)
				    printf("kernel launch failed with error \"%s\".\n",cudaGetErrorString(cudaerr));
				
				cudaMemcpy(h_img,d_img , 3 * width * height * sizeof(unsigned int),cudaMemcpyDeviceToHost);
				continue;
			}
			else if(!strcmp(argv[i],"green"))
			{
				cudaMemcpy(d_img, h_img, 3 * width * height * sizeof(unsigned int),cudaMemcpyHostToDevice);
				saturate_green<<<grid,block>>>(d_img, height * width);
				cudaError_t cudaerr = cudaDeviceSynchronize();
				  if (cudaerr != cudaSuccess)
				    printf("kernel launch failed with error \"%s\".\n",cudaGetErrorString(cudaerr));
				cudaMemcpy(h_img,d_img , 3 * width * height * sizeof(unsigned int),cudaMemcpyDeviceToHost);
				continue;
			}
			else if(!strcmp(argv[i],"blue"))
			{
				cudaMemcpy(d_img, h_img, 3 * width * height * sizeof(unsigned int),cudaMemcpyHostToDevice);
				saturate_blue<<<grid,block>>>(d_img, height * width);
				cudaError_t cudaerr = cudaDeviceSynchronize();
				  if (cudaerr != cudaSuccess)
				    printf("kernel launch failed with error \"%s\".\n",cudaGetErrorString(cudaerr));
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
		if(!strcmp(argv[i],"--negatif"))
			{
				cudaMemcpy(d_img, h_img, 3 * width * height * sizeof(unsigned int),cudaMemcpyHostToDevice);
				negatif<<<grid,block>>>(d_img, height * width);
				cudaError_t cudaerr = cudaDeviceSynchronize();
				  if (cudaerr != cudaSuccess)
				    printf("kernel launch failed with error \"%s\".\n",cudaGetErrorString(cudaerr));
				cudaMemcpy(h_img,d_img , 3 * width * height * sizeof(unsigned int),cudaMemcpyDeviceToHost);
				continue;
			}
		if(!strcmp(argv[i],"--flip"))
		{
			//printf("Miror not yet implemented!\n");
			cudaMemcpy(d_img, h_img, 3 * width * height * sizeof(unsigned int),cudaMemcpyHostToDevice);
 			cudaMemcpy(d_tmp, h_img, 3 * width * height * sizeof(unsigned int),cudaMemcpyHostToDevice);
			symetry<<<grid,block>>>(d_img,d_tmp, height * width);
			cudaError_t cudaerr = cudaDeviceSynchronize();
				  if (cudaerr != cudaSuccess)
				    printf("kernel launch failed with error \"%s\".\n",cudaGetErrorString(cudaerr));
			cudaMemcpy(h_img,d_img , 3 * width * height * sizeof(unsigned int),cudaMemcpyDeviceToHost);
			continue;
		}

		if(!strcmp(argv[i],"--blur"))
		{
			i++;
			for(int z =0; z<atoi(argv[i]); z++)
			{
				cudaMemcpy(d_img, h_img, 3 * width * height * sizeof(unsigned int),cudaMemcpyHostToDevice);
			 	cudaMemcpy(d_tmp, h_img, 3 * width * height * sizeof(unsigned int),cudaMemcpyHostToDevice);
			 	blur<<<grid,block>>>(d_img,d_tmp,height,width, height*width);
				
				cudaError_t cudaerr = cudaDeviceSynchronize();
					  if (cudaerr != cudaSuccess)
					    printf("kernel launch failed with error \"%s\".\n",cudaGetErrorString(cudaerr));
				cudaMemcpy(h_img,d_img , 3 * width * height * sizeof(unsigned int),cudaMemcpyDeviceToHost);
			}
			continue;
		}

		if(!strcmp(argv[i],"--grey"))
		{
			cudaMemcpy(d_img, h_img, 3 * width * height * sizeof(unsigned int),cudaMemcpyHostToDevice);
			grey<<<grid,block>>>(d_img, height * width);
			cudaError_t cudaerr = cudaDeviceSynchronize();
				  if (cudaerr != cudaSuccess)
				    printf("kernel launch failed with error \"%s\".\n",cudaGetErrorString(cudaerr));
			cudaMemcpy(h_img,d_img , 3 * width * height * sizeof(unsigned int),cudaMemcpyDeviceToHost);
			continue;
		}

		if(!strcmp(argv[i],"--sobel"))
		{
			cudaMemcpy(d_img, h_img, 3 * width * height * sizeof(unsigned int),cudaMemcpyHostToDevice);
			grey<<<grid,block>>>(d_img, height * width);
			cudaMemcpy(d_tmp, d_img, 3 * width * height * sizeof(unsigned int),cudaMemcpyDeviceToDevice);
			sobel<<<grid,block>>>(d_img,d_tmp,height,width,height*width);
			cudaError_t cudaerr = cudaDeviceSynchronize();
				  if (cudaerr != cudaSuccess)
				    printf("kernel launch failed with error \"%s\".\n",cudaGetErrorString(cudaerr));
			cudaMemcpy(h_img,d_img , 3 * width * height * sizeof(unsigned int),cudaMemcpyDeviceToHost);
			continue;
		}

		if(!strcmp(argv[i],"--popart"))
		{
			popart(bitmap, h_img,height,width,pitch);
			continue;
		}
	}
	cudaError_t cudaerr = cudaDeviceSynchronize();
				  if (cudaerr != cudaSuccess)
				    printf("kernel launch failed with error \"%s\".\n",cudaGetErrorString(cudaerr));
	printf("Saving image...\n");
	save_image(bitmap,h_img,out_name.c_str(),height,width,pitch);

	FreeImage_DeInitialise();
	cudaFree(d_img);
	cudaFree(d_tmp);
	free(h_img);
	exit(0);
}