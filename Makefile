# Makefile for Shallow Water Equations Solver with CUDA

# --- Compilers ---
NVCC = nvcc
CXX = $(NVCC) # Use NVCC to compile both .cu and .cc files
LD = $(NVCC)  # Use NVCC for final linking

# --- Compiler Flags ---
# General optimization and warning flags for both host and device code

# COMMON_FLAGS = -O3 -std=c++17
COMMON_FLAGS = --compiler-options -Wall -std=c++17

# CUDA specific flags for device code
# Adjust sm_75 to your GPU's compute capability (e.g., sm_86 for Ampere, sm_75 for Turing/Volta)
# You can find your GPU's compute capability by running `nvidia-smi -q | grep "Compute Capability"`
NVCCFLAGS = $(COMMON_FLAGS)

# CXXFLAGS are passed to NVCC when compiling .cc files
CXXFLAGS = $(COMMON_FLAGS)

# --- Include Directories ---
# Include paths for source files and HDF5
COMMON_INCLUDE_DIRS = -I. -I$(HDF5_ROOT)/include
NVCCFLAGS += $(COMMON_INCLUDE_DIRS) -I$(CUDA_PATH)/include
CXXFLAGS += $(COMMON_INCLUDE_DIRS)

# --- Linker Flags ---
# Libraries for HDF5 and CUDA runtime
LDFLAGS = -lm \
          -L$(HDF5_ROOT)/lib -lhdf5 -lhdf5_hl \
          -L$(CUDA_PATH)/lib64

# --- Object Files ---
# C++ object files
CXX_OBJS = main.o swe.o xdmf_writer.o
# CUDA object files
CUDA_OBJS = swe_cuda_kernels.o

# All object files required for the final executable
OBJS = $(CXX_OBJS) $(CUDA_OBJS)

# --- Target Executable Name ---
TARGET = swe

# --- Main Target ---
$(TARGET): $(OBJS)
	$(LD) -o $@ $^ $(LDFLAGS)

# --- Compilation Rules ---
# Rule for CUDA source files (.cu)
# NVCC automatically handles the compilation of .cu files into object files
%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# Rule for C++ source files (.cc)
# NVCC is used here as the C++ compiler, passing CXXFLAGS
%.o: %.cc
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Rule for C++ source files (.cpp) - included for completeness if you use .cpp
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# --- Clean Target ---
clean:
	rm -f $(TARGET) $(OBJS) *~
