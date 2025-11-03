NVCC := nvcc
CXX := g++
ARCH := -arch=sm_75
CXXSTD := -std=c++17
OPT := -O3

INCDIR := include
SRCDIR := src
BINDIR := bin

INCLUDES := -I$(INCDIR)

CU_SRCS := $(SRCDIR)/gray.cu $(SRCDIR)/boxblur.cu $(SRCDIR)/sobel.cu $(SRCDIR)/main.cu
CPP_SRCS := $(SRCDIR)/image_io.cpp

OBJS := $(CU_SRCS:.cu=.o) $(CPP_SRCS:.cpp=.o)

all: $(BINDIR)/imgproc

$(SRCDIR)/%.o: $(SRCDIR)/%.cu
	$(NVCC) $(OPT) $(ARCH) $(INCLUDES) -c $< -o $@

$(SRCDIR)/%.o: $(SRCDIR)/%.cpp
	$(NVCC) $(OPT) $(CXXSTD) $(INCLUDES) -c $< -o $@

$(BINDIR)/imgproc: $(OBJS)
	mkdir -p $(BINDIR)
	$(NVCC) $(OPT) $(ARCH) $(INCLUDES) -o $@ $(OBJS)

clean:
	rm -f $(SRCDIR)/*.o
	rm -rf $(BINDIR)

