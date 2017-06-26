INCLUDES=../utils
CXXFLAGS=-std=c++11 -Wno-deprecated-gpu-targets

all: main.cu 
	nvcc -I$(INCLUDES) $(CXXFLAGS) main.cu -o main 

clean:
	rm main
