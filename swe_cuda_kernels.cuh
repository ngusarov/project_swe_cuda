// swe_cuda_kernels.cuh
#ifndef SWE_CUDA_KERNELS_CUH
#define SWE_CUDA_KERNELS_CUH

#include "swe_matrix_gpu.hh"
#include <cstddef> // For std::size_t
#include <vector>  // For host-side data for reduction result

// Define g as a preprocessor macro to be a compile-time constant for kernels
// This value is taken directly from SWESolver::g in swe.hh
#define G_CONSTANT 127267.20000000

// Device functions (can be called from other kernels or host functions with __device__)
__device__ inline double get_val_2d(const double* data, std::size_t i, std::size_t j, std::size_t nx);

// Kernels
__global__ void compute_kernel_gpu(std::size_t nx, std::size_t ny,
                                   double dt, double dx, double dy,
                                   const double* h0_d, const double* hu0_d, const double* hv0_d,
                                   const double* zdx_d, const double* zdy_d,
                                   double* h_d, double* hu_d, double* hv_d);

__global__ void update_bcs_gpu_horizontal(std::size_t nx, std::size_t ny, double coef,
                                          const double* h0_d, const double* hu0_d, const double* hv0_d,
                                          double* h_d, double* hu_d, double* hv_d);

__global__ void update_bcs_gpu_vertical(std::size_t nx, std::size_t ny, double coef,
                                        const double* h0_d, const double* hu0_d, const double* hv0_d,
                                        double* h_d, double* hu_d, double* hv_d);

__global__ void compute_nu_sqr_kernel(std::size_t nx, std::size_t ny,
                                      const double* h_d, const double* hu_d, const double* hv_d,
                                      double* nu_sqr_d);

__global__ void reduce_max_kernel(double* input_d, double* output_d, std::size_t n);


// Host wrapper functions
void swe_solve_step_gpu(std::size_t nx, std::size_t ny,
                        double dt, double dx, double dy,
                        const SWEMatrixGPU& h0, const SWEMatrixGPU& hu0, const SWEMatrixGPU& hv0,
                        const SWEMatrixGPU& zdx, const SWEMatrixGPU& zdy,
                        SWEMatrixGPU& h, SWEMatrixGPU& hu, SWEMatrixGPU& hv);

void swe_update_bcs_gpu(std::size_t nx, std::size_t ny, bool reflective,
                        const SWEMatrixGPU& h0, const SWEMatrixGPU& hu0, const SWEMatrixGPU& hv0,
                        SWEMatrixGPU& h, SWEMatrixGPU& hu, SWEMatrixGPU& hv);

double swe_compute_max_nu_sqr_gpu(std::size_t nx, std::size_t ny,
                                  const SWEMatrixGPU& h, const SWEMatrixGPU& hu, const SWEMatrixGPU& hv);

#endif // SWE_CUDA_KERNELS_CUH