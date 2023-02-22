SOURCES = main.cu utils.cu
OBJECTS = $(SOURCES:.cu=.o)

default: all

all: main

main: $(OBJECTS)
	nvcc -I${HOME}/softs/FreeImage/include $^ -L${HOME}/softs/FreeImage/lib/ -lfreeimage -o $@ -g

%.o: %.cu
	nvcc -I${HOME}/softs/FreeImage/include $< -L${HOME}/softs/FreeImage/lib/ -dc -o $@ -g
clean:
	rm -f *.o main
