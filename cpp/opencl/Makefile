# Compiler and flags
CXX = g++
CXXFLAGS = -std=c++17 -O2 -Wall

# OpenCL library and include paths
OPENCL_INC = -I/path/to/OpenCL/include
OPENCL_LIB = -L/path/to/OpenCL/lib -lOpenCL

# Source files
SRC = core.cpp network.cpp neuronblock.cpp main.cpp
OBJ = $(SRC:.cpp=.o)

# Output binary
TARGET = neural_net_opencl

# Build rules
all: $(TARGET)

$(TARGET): $(OBJ)
	$(CXX) $(CXXFLAGS) $(OBJ) -o $@ $(OPENCL_LIB)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(OPENCL_INC) -c $< -o $@

clean:
	rm -f $(OBJ) $(TARGET)
