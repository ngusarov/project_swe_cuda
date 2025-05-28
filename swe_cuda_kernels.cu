// swe_cuda_kernels.cu
#include "swe_cuda_kernels.cuh"
#include <cmath> // For std::fabs, sqrt, std::max
#include <iostream> // For error reporting
#include <algorithm> // For std::swap

// Helper device function for 2D indexing
__device__ inline double get_val_2d(const double* data, std::size_t i, std::size_t j, std::size_t nx) {
    return data[j * nx + i];
}

// Helper device function for 2D indexing (writeable)
__device__ inline double& set_val_2d(double* data, std::size_t i, std::size_t j, std::size_t nx) {
    return data[j * nx + i];
}

/**
 * @brief CUDA kernel to compute the shallow water equations for inner cells.
 *
 * This kernel parallelizes the `compute_kernel` logic from the CPU version.
 * Each thread computes the new h, hu, hv values for a single cell (i, j).
 *
 * @param nx Number of cells in x-direction.
 * @param ny Number of cells in y-direction.
 * @param dt Time step.
 * @param dx Cell size in x-direction.
 * @param dy Cell size in y-direction.
 * @param h0_d Device pointer to water height at previous time step.
 * @param hu0_d Device pointer to x-velocity at previous time step.
 * @param hv0_d Device pointer to y-velocity at previous time step.
 * @param zdx_d Device pointer to x-derivative of topography.
 * @param zdy_d Device pointer to y-derivative of topography.
 * @param h_d Device pointer to water height at current time step (output).
 * @param hu_d Device pointer to x-velocity at current time step (output).
 * @param hv_d Device pointer to y-velocity at current time step (output).
 */
__global__ void compute_kernel_gpu(std::size_t nx, std::size_t ny,
                                   double dt, double dx, double dy,
                                   const double* h0_d, const double* hu0_d, const double* hv0_d,
                                   const double* zdx_d, const double* zdy_d,
                                   double* h_d, double* hu_d, double* hv_d) {
  // Calculate global index (row-major)
  std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

  // Map 1D index to 2D (i, j) for inner cells (1 to nx-2, 1 to ny-2)
  std::size_t inner_nx = nx - 2;
  std::size_t inner_ny = ny - 2;

  if (idx < inner_nx * inner_ny) {
    std::size_t j_inner = idx / inner_nx;
    std::size_t i_inner = idx % inner_nx;

    // Map inner grid (i_inner, j_inner) to full grid (i, j)
    std::size_t i = i_inner + 1;
    std::size_t j = j_inner + 1;

    const double C1x = 0.5 * dt / dx;
    const double C1y = 0.5 * dt / dy;
    const double C2 = dt * G_CONSTANT; // Use G_CONSTANT here
    constexpr double C3 = 0.5 * G_CONSTANT; // Use G_CONSTANT here

    double h0_ij_m1 = get_val_2d(h0_d, i, j - 1, nx);
    double h0_ij_p1 = get_val_2d(h0_d, i, j + 1, nx);
    double h0_i_m1_j = get_val_2d(h0_d, i - 1, j, nx);
    double h0_i_p1_j = get_val_2d(h0_d, i + 1, j, nx);

    double hu0_ij_m1 = get_val_2d(hu0_d, i, j - 1, nx);
    double hu0_ij_p1 = get_val_2d(hu0_d, i, j + 1, nx);
    double hu0_i_m1_j = get_val_2d(hu0_d, i - 1, j, nx);
    double hu0_i_p1_j = get_val_2d(hu0_d, i + 1, j, nx);

    double hv0_ij_m1 = get_val_2d(hv0_d, i, j - 1, nx);
    double hv0_ij_p1 = get_val_2d(hv0_d, i, j + 1, nx);
    double hv0_i_m1_j = get_val_2d(hv0_d, i - 1, j, nx);
    double hv0_i_p1_j = get_val_2d(hv0_d, i + 1, j, nx);

    double zdx_ij = get_val_2d(zdx_d, i, j, nx);
    double zdy_ij = get_val_2d(zdy_d, i, j, nx);

    double hij = 0.25 * (h0_ij_m1 + h0_ij_p1 + h0_i_m1_j + h0_i_p1_j)
                 + C1x * (hu0_i_m1_j - hu0_i_p1_j)
                 + C1y * (hv0_ij_m1 - hv0_ij_p1);
    if (hij < 0.0) {
      hij = 1.0e-5;
    }

    set_val_2d(h_d, i, j, nx) = hij;

    if (hij > 0.0001) {
      set_val_2d(hu_d, i, j, nx) =
        0.25 * (hu0_ij_m1 + hu0_ij_p1 + hu0_i_m1_j + hu0_i_p1_j) - C2 * hij * zdx_ij
        + C1x
            * (hu0_i_m1_j * hu0_i_m1_j / h0_i_m1_j + C3 * h0_i_m1_j * h0_i_m1_j
               - hu0_i_p1_j * hu0_i_p1_j / h0_i_p1_j - C3 * h0_i_p1_j * h0_i_p1_j)
        + C1y
            * (hu0_ij_m1 * hv0_ij_m1 / h0_ij_m1
               - hu0_ij_p1 * hv0_ij_p1 / h0_ij_p1);

      set_val_2d(hv_d, i, j, nx) =
        0.25 * (hv0_ij_m1 + hv0_ij_p1 + hv0_i_m1_j + hv0_i_p1_j) - C2 * hij * zdy_ij
        + C1x
            * (hu0_i_m1_j * hv0_i_m1_j / h0_i_m1_j
               - hu0_i_p1_j * hv0_i_p1_j / h0_i_p1_j)
        + C1y
            * (hv0_ij_m1 * hv0_ij_m1 / h0_ij_m1 + C3 * h0_ij_m1 * h0_ij_m1
               - hv0_ij_p1 * hv0_ij_p1 / h0_ij_p1 - C3 * h0_ij_p1 * h0_ij_p1);
    } else {
      set_val_2d(hu_d, i, j, nx) = 0.0;
      set_val_2d(hv_d, i, j, nx) = 0.0;
    }
  }
}

/**
 * @brief CUDA kernel to update horizontal boundary conditions (top and bottom).
 * Each thread handles one column for both top and bottom boundaries.
 *
 * @param nx Number of cells in x-direction.
 * @param ny Number of cells in y-direction.
 * @param coef Coefficient for reflective/non-reflective boundaries for velocity.
 * @param h0_d Device pointer to water height at previous time step.
 * @param hu0_d Device pointer to x-velocity at previous time step.
 * @param hv0_d Device pointer to y-velocity at previous time step.
 * @param h_d Device pointer to water height at current time step (output).
 * @param hu_d Device pointer to x-velocity at current time step (output).
 * @param hv_d Device pointer to y-velocity at current time step (output).
 */
__global__ void update_bcs_gpu_horizontal(std::size_t nx, std::size_t ny, double coef,
                                          const double* h0_d, const double* hu0_d, const double* hv0_d,
                                          double* h_d, double* hu_d, double* hv_d) {
  std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < nx) {
    // Top boundary (j=0)
    set_val_2d(h_d, i, 0, nx) = get_val_2d(h0_d, i, 1, nx);
    set_val_2d(hu_d, i, 0, nx) = get_val_2d(hu0_d, i, 1, nx);
    set_val_2d(hv_d, i, 0, nx) = coef * get_val_2d(hv0_d, i, 1, nx);

    // Bottom boundary (j=ny-1)
    set_val_2d(h_d, i, ny - 1, nx) = get_val_2d(h0_d, i, ny - 2, nx);
    set_val_2d(hu_d, i, ny - 1, nx) = get_val_2d(hu0_d, i, ny - 2, nx);
    set_val_2d(hv_d, i, ny - 1, nx) = coef * get_val_2d(hv0_d, i, ny - 2, nx);
  }
}

/**
 * @brief CUDA kernel to update vertical boundary conditions (left and right).
 * Each thread handles one row for both left and right boundaries.
 *
 * @param nx Number of cells in x-direction.
 * @param ny Number of cells in y-direction.
 * @param coef Coefficient for reflective/non-reflective boundaries for velocity.
 * @param h0_d Device pointer to water height at previous time step.
 * @param hu0_d Device pointer to x-velocity at previous time step.
 * @param hv0_d Device pointer to y-velocity at previous time step.
 * @param h_d Device pointer to water height at current time step (output).
 * @param hu_d Device pointer to x-velocity at current time step (output).
 * @param hv_d Device pointer to y-velocity at current time step (output).
 */
__global__ void update_bcs_gpu_vertical(std::size_t nx, std::size_t ny, double coef,
                                        const double* h0_d, const double* hu0_d, const double* hv0_d,
                                        double* h_d, double* hu_d, double* hv_d) {
  std::size_t j = blockIdx.x * blockDim.x + threadIdx.x;

  if (j < ny) {
    // Left boundary (i=0)
    set_val_2d(h_d, 0, j, nx) = get_val_2d(h0_d, 1, j, nx);
    set_val_2d(hu_d, 0, j, nx) = coef * get_val_2d(hu0_d, 1, j, nx);
    set_val_2d(hv_d, 0, j, nx) = get_val_2d(hv0_d, 1, j, nx);

    // Right boundary (i=nx-1)
    set_val_2d(h_d, nx - 1, j, nx) = get_val_2d(h0_d, nx - 2, j, nx);
    set_val_2d(hu_d, nx - 1, j, nx) = coef * get_val_2d(hu0_d, nx - 2, j, nx);
    set_val_2d(hv_d, nx - 1, j, nx) = get_val_2d(hv0_d, nx - 2, j, nx);
  }
}

/**
 * @brief CUDA kernel to compute nu_sqr for each inner cell.
 * This is the first step for computing the time step (CFL condition).
 *
 * @param nx Number of cells in x-direction.
 * @param ny Number of cells in y-direction.
 * @param h_d Device pointer to water height.
 * @param hu_d Device pointer to x-velocity.
 * @param hv_d Device pointer to y-velocity.
 * @param nu_sqr_d Device pointer to store nu_u^2 + nu_v^2 for each cell.
 */
__global__ void compute_nu_sqr_kernel(std::size_t nx, std::size_t ny,
                                      const double* h_d, const double* hu_d, const double* hv_d,
                                      double* nu_sqr_d) {
  std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

  std::size_t inner_nx = nx - 2;
  std::size_t inner_ny = ny - 2;

  if (idx < inner_nx * inner_ny) {
    std::size_t j = (idx / inner_nx) + 1; // Map to full grid row
    std::size_t i = (idx % inner_nx) + 1; // Map to full grid col

    double h_val = get_val_2d(h_d, i, j, nx);
    double hu_val = get_val_2d(hu_d, i, j, nx);
    double hv_val = get_val_2d(hv_d, i, j, nx);

    // Use G_CONSTANT here
    double nu_u = std::fabs(hu_val) / h_val + sqrt(G_CONSTANT * h_val);
    double nu_v = std::fabs(hv_val) / h_val + sqrt(G_CONSTANT * h_val);

    // Store in the flattened 1D array
    nu_sqr_d[idx] = nu_u * nu_u + nu_v * nu_v;
  }
}

/**
 * @brief CUDA kernel for parallel reduction (finding maximum).
 * This kernel performs one step of a parallel reduction.
 *
 * @param input_d Device pointer to the input array (or previous reduction step's output).
 * @param output_d Device pointer to the output array for this reduction step.
 * @param n Number of elements in the input array for this step.
 */
__global__ void reduce_max_kernel(double* input_d, double* output_d, std::size_t n) {
    extern __shared__ double sdata[]; // Shared memory for reduction

    std::size_t tid = threadIdx.x;
    std::size_t i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

    sdata[tid] = 0.0; // Initialize shared memory element

    // Load data from global memory to shared memory
    if (i < n) {
        sdata[tid] = input_d[i];
    }
    if (i + blockDim.x < n) {
        sdata[tid] = fmax(sdata[tid], input_d[i + blockDim.x]);
    }
    __syncthreads();

    // Perform reduction in shared memory
    for (std::size_t s = blockDim.x / 2; s > 0; s /= 2) {
        if (tid < s) {
            sdata[tid] = fmax(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }

    // Write result for this block to global memory
    if (tid == 0) {
        output_d[blockIdx.x] = sdata[0];
    }
}


// Host wrapper functions

/**
 * @brief Host wrapper to launch the compute_kernel_gpu.
 *
 * @param nx Number of cells in x-direction.
 * @param ny Number of cells in y-direction.
 * @param dt Time step.
 * @param dx Cell size in x-direction.
 * @param dy Cell size in y-direction.
 * @param h0 SWEMatrixGPU for h0.
 * @param hu0 SWEMatrixGPU for hu0.
 * @param hv0 SWEMatrixGPU for hv0.
 * @param zdx SWEMatrixGPU for zdx.
 * @param zdy SWEMatrixGPU for zdy.
 * @param h SWEMatrixGPU for h (output).
 * @param hu SWEMatrixGPU for hu (output).
 * @param hv SWEMatrixGPU for hv (output).
 */
void swe_solve_step_gpu(std::size_t nx, std::size_t ny,
                        double dt, double dx, double dy,
                        const SWEMatrixGPU& h0, const SWEMatrixGPU& hu0, const SWEMatrixGPU& hv0,
                        const SWEMatrixGPU& zdx, const SWEMatrixGPU& zdy,
                        SWEMatrixGPU& h, SWEMatrixGPU& hu, SWEMatrixGPU& hv) {
  // Only inner cells are computed by this kernel
  std::size_t inner_nx = nx - 2;
  std::size_t inner_ny = ny - 2;
  std::size_t num_elements = inner_nx * inner_ny;

  if (num_elements == 0) return; // No inner cells to compute

  int threadsPerBlock = 256; // Standard block size
  int blocksPerGrid = (num_elements + threadsPerBlock - 1) / threadsPerBlock;

  compute_kernel_gpu<<<blocksPerGrid, threadsPerBlock>>>(
      nx, ny, dt, dx, dy,
      h0.data(), hu0.data(), hv0.data(),
      zdx.data(), zdy.data(),
      h.data(), hu.data(), hv.data());

  cudaDeviceSynchronize(); // Ensure kernel completes
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
      std::cerr << "CUDA error in swe_solve_step_gpu: " << cudaGetErrorString(err) << std::endl;
  }
}

/**
 * @brief Host wrapper to launch CUDA kernels for updating boundary conditions.
 *
 * @param nx Number of cells in x-direction.
 * @param ny Number of cells in y-direction.
 * @param reflective Boolean indicating reflective boundaries.
 * @param h0 SWEMatrixGPU for h0.
 * @param hu0 SWEMatrixGPU for hu0.
 * @param hv0 SWEMatrixGPU for hv0.
 * @param h SWEMatrixGPU for h (output).
 * @param hu SWEMatrixGPU for hu (output).
 * @param hv SWEMatrixGPU for hv (output).
 */
void swe_update_bcs_gpu(std::size_t nx, std::size_t ny, bool reflective,
                        const SWEMatrixGPU& h0, const SWEMatrixGPU& hu0, const SWEMatrixGPU& hv0,
                        SWEMatrixGPU& h, SWEMatrixGPU& hu, SWEMatrixGPU& hv) {
  double coef = reflective ? -1.0 : 1.0;

  int threadsPerBlock = 256;

  // Horizontal boundaries (top and bottom)
  if (nx > 0 && ny >= 2) { // Need at least 2 rows for horizontal boundary updates
    int blocksPerGrid_h = (nx + threadsPerBlock - 1) / threadsPerBlock;
    update_bcs_gpu_horizontal<<<blocksPerGrid_h, threadsPerBlock>>>(
        nx, ny, coef,
        h0.data(), hu0.data(), hv0.data(),
        h.data(), hu.data(), hv.data());
    cudaDeviceSynchronize();
    cudaError_t err_h = cudaGetLastError();
    if (err_h != cudaSuccess) {
        std::cerr << "CUDA error in update_bcs_gpu_horizontal: " << cudaGetErrorString(err_h) << std::endl;
    }
  }


  // Vertical boundaries (left and right)
  if (ny > 0 && nx >= 2) { // Need at least 2 columns for vertical boundary updates
    int blocksPerGrid_v = (ny + threadsPerBlock - 1) / threadsPerBlock;
    update_bcs_gpu_vertical<<<blocksPerGrid_v, threadsPerBlock>>>(
        nx, ny, coef,
        h0.data(), hu0.data(), hv0.data(),
        h.data(), hu.data(), hv.data());
    cudaDeviceSynchronize();
    cudaError_t err_v = cudaGetLastError();
    if (err_v != cudaSuccess) {
        std::cerr << "CUDA error in update_bcs_gpu_vertical: " << cudaGetErrorString(err_v) << std::endl;
    }
  }
}

/**
 * @brief Host wrapper to compute the maximum of nu_sqr on the GPU.
 * This involves launching compute_nu_sqr_kernel and then a reduction kernel.
 *
 * @param nx Number of cells in x-direction.
 * @param ny Number of cells in y-direction.
 * @param h SWEMatrixGPU for h.
 * @param hu SWEMatrixGPU for hu.
 * @param hv SWEMatrixGPU for hv.
 * @return The maximum nu_sqr value.
 */
double swe_compute_max_nu_sqr_gpu(std::size_t nx, std::size_t ny,
                                  const SWEMatrixGPU& h, const SWEMatrixGPU& hu, const SWEMatrixGPU& hv) {
  std::size_t inner_nx = nx - 2;
  std::size_t inner_ny = ny - 2;
  std::size_t num_elements = inner_nx * inner_ny;

  if (num_elements == 0) return 0.0; // Handle empty inner grid

  double* nu_sqr_flat_d = nullptr;
  double* temp_reduction_buffer_d = nullptr;
  double max_val = 0.0;

  cudaError_t err;

  // Allocate memory for flattened nu_sqr values
  err = cudaMallocManaged(&nu_sqr_flat_d, num_elements * sizeof(double));
  if (err != cudaSuccess) {
      std::cerr << "CUDA error cudaMallocManaged nu_sqr_flat_d: " << cudaGetErrorString(err) << std::endl;
      return 0.0;
  }

  int threadsPerBlock = 256;
  int blocksPerGrid_nu = (num_elements + threadsPerBlock - 1) / threadsPerBlock;

  // Allocate memory for the output of the first reduction stage
  // The number of elements after the first reduction is blocksPerGrid_nu
  err = cudaMallocManaged(&temp_reduction_buffer_d, blocksPerGrid_nu * sizeof(double));
  if (err != cudaSuccess) {
      std::cerr << "CUDA error cudaMallocManaged temp_reduction_buffer_d: " << cudaGetErrorString(err) << std::endl;
      cudaFree(nu_sqr_flat_d);
      return 0.0;
  }

  // Step 1: Compute nu_sqr for each inner cell into the flattened array
  compute_nu_sqr_kernel<<<blocksPerGrid_nu, threadsPerBlock>>>(
      nx, ny, h.data(), hu.data(), hv.data(), nu_sqr_flat_d);
  cudaDeviceSynchronize();
  err = cudaGetLastError();
  if (err != cudaSuccess) {
      std::cerr << "CUDA error in compute_nu_sqr_kernel (flat): " << cudaGetErrorString(err) << std::endl;
      cudaFree(nu_sqr_flat_d);
      cudaFree(temp_reduction_buffer_d);
      return 0.0;
  }

  // Step 2: Perform reduction to find the maximum value
  std::size_t current_n = num_elements;
  double* input_ptr = nu_sqr_flat_d;
  double* output_ptr = temp_reduction_buffer_d;

  while (current_n > 1) {
      int reduce_blocks = (current_n + (threadsPerBlock * 2) - 1) / (threadsPerBlock * 2);
      if (reduce_blocks == 0 && current_n > 0) reduce_blocks = 1;

      reduce_max_kernel<<<reduce_blocks, threadsPerBlock, threadsPerBlock * sizeof(double)>>>(
          input_ptr, output_ptr, current_n);
      cudaDeviceSynchronize();
      err = cudaGetLastError();
      if (err != cudaSuccess) {
          std::cerr << "CUDA error in reduce_max_kernel: " << cudaGetErrorString(err) << std::endl;
          cudaFree(nu_sqr_flat_d);
          cudaFree(temp_reduction_buffer_d);
          return 0.0;
      }

      std::swap(input_ptr, output_ptr);
      current_n = reduce_blocks;
  }

  // Final result is in input_ptr (due to the last swap)
  if (current_n == 1) {
      cudaMemcpy(&max_val, input_ptr, sizeof(double), cudaMemcpyDeviceToHost);
      cudaDeviceSynchronize();
      err = cudaGetLastError();
      if (err != cudaSuccess) {
          std::cerr << "CUDA error after cudaMemcpy (final result): " << cudaGetErrorString(err) << std::endl;
      }
  }

  cudaFree(nu_sqr_flat_d);
  cudaFree(temp_reduction_buffer_d);
  
  return max_val;
}