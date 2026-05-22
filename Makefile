# Define the compiler
CXX = g++

# Define the include directory
INC_DIR = ./include

# Define compiler flags (e.g., -g for debugging, -Wall for all warnings)
CXXFLAGS = -g -Wall -I$(INC_DIR) -fsanitize=address

# Define the target executable name
TARGET = neuralNetworkCpp

# Automatically find all .cpp files in the directory
SRCS = $(wildcard *.cpp)

# Convert the .cpp filenames to .o filenames
OBJS = $(SRCS:.cpp=.o)

# Default rule to build the executable
all: $(TARGET)

# Link the object files to create the executable
$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(OBJS)

# Rule to clean up generated files
clean:
	rm -f $(TARGET) $(OBJS)

.PHONY: all clean