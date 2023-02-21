#include <string>
#include <ostream>
#include <FreeImage.h>
#include "utils.h"

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

 	CudaMalloc(&d_img,sizeof(unsigned int) * 3 * width * height);
 	CudaMalloc(&d_img,sizeof(unsigned int) * 3 * width * height);

 	get_image(bitmap,h_img,height,width,pitch);

 	memcpy(d_img, img, 3 * width * height * sizeof(unsigned int));
 	memcpy(d_tmp, img, 3 * width * height * sizeof(unsigned int));

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
				continue;
			}
			else if(!strcmp(argv[i],"green"))
			{
				printf("Saturate Green not yet implemented!\n");
				continue;
			}
			else if(!strcmp(argv[i],"blue"))
			{
				printf("Saturate Blue not yet implemented!\n");
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
			printf("Miror not yet implemented!\n");
			continue;
		}
	}



	FreeImage_DeInitialise();
	CudaFree(d_img);
	CudaFree(d_tmp);
	free(h_img);
	exit(0);
}