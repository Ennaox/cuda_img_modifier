SOURCES = main.cu utils.cu
OBJECTS = $(SOURCES:.cu=.o)

default: all

all: main

main: $(OBJECTS)
	nvcc $^  -lfreeimage -o $@ -g

%.o: %.cu
	nvcc $< -c -o $@ 
clean:
	rm -f *.o main
