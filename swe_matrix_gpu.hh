// swe_matrix_gpu.hh
#ifndef SWE_MATRIX_GPU_HH
#define SWE_MATRIX_GPU_HH

#include <cuda_runtime.h>
#include <cassert>
#include <vector> 

/**
 * @brief A class to manage 2D data on the GPU using CUDA Unified Memory.
 *
 * This class provides a 2D array-like interface for data stored in CUDA
 * managed memory, allowing access from both host and device.
 * It is designed to store data in row-major order, consistent with
 * std::vector<double> vec[j * nx_ + i] indexing.
 */
class SWEMatrixGPU {
public:
  /**
   * @brief Constructor for SWEMatrixGPU.
   * @param nx Number of columns (x-dimension size).
   * @param ny Number of rows (y-dimension size).
   */
  __host__ SWEMatrixGPU(std::size_t nx, std::size_t ny) : nx_(nx), ny_(ny), data_(nullptr), is_copy_(false) {
    // Allocate unified memory, accessible from both host and device
    cudaMallocManaged(&data_, nx_ * ny_ * sizeof(double));
    // Error checking for cudaMallocManaged
    if (data_ == nullptr) {
        assert(false && "CUDA managed memory allocation failed!");
    }
  }

  /**
   * @brief Copy constructor.
   * @param other The SWEMatrixGPU object to copy from.
   * @note This performs a shallow copy of the pointer and marks it as a copy.
   * The original object remains responsible for deallocation.
   */
  __host__ __device__ SWEMatrixGPU(const SWEMatrixGPU &other)
      : nx_(other.nx_), ny_(other.ny_), data_(other.data_), is_copy_(true) {}

  /**
   * @brief Destructor.
   * @note Frees memory only if this object is the original owner (not a copy).
   */
  __host__ ~SWEMatrixGPU() {
    if (!is_copy_ && data_ != nullptr) {
      cudaFree(data_);
      data_ = nullptr; // Prevent double free
    }
  }

  /**
   * @brief 2D accessor for elements (read/write).
   * @param i Column index (x-coordinate).
   * @param j Row index (y-coordinate).
   * @return Reference to the element at (i, j).
   */
  __host__ __device__ inline double &operator()(std::size_t i, std::size_t j) {
    // Stored in row-major: index = row * num_cols + col
    return data_[j * nx_ + i];
  }

  /**
   * @brief 2D accessor for elements (read-only).
   * @param i Column index (x-coordinate).
   * @param j Row index (y-coordinate).
   * @return Constant reference to the element at (i, j).
   */
  __host__ __device__ inline const double &operator()(std::size_t i, std::size_t j) const {
    // Stored in row-major: index = row * num_cols + col
    return data_[j * nx_ + i];
  }

  /**
   * @brief Get the number of rows (y-dimension size).
   * @return Number of rows.
   */
  __host__ __device__ std::size_t rows() const { return ny_; }

  /**
   * @brief Get the number of columns (x-dimension size).
   * @return Number of columns.
   */
  __host__ __device__ std::size_t cols() const { return nx_; }

  /**
   * @brief Get a pointer to the raw data on the GPU.
   * @return Pointer to the beginning of the data array.
   */
  __host__ double *data() { return data_; }

  /**
   * @brief Get a constant pointer to the raw data on the GPU.
   * @return Constant pointer to the beginning of the data array.
   */
  __host__ const double *data() const { return data_; }

  /**
   * @brief Copies data from a host std::vector to the GPU matrix.
   * @param host_data The host vector containing the data.
   * @note Assumes host_data size matches the matrix dimensions.
   */
  __host__ void copy_from_host(const std::vector<double>& host_data) {
    assert(host_data.size() == nx_ * ny_ && "Host data size mismatch for GPU copy!");
    cudaMemcpy(data_, host_data.data(), nx_ * ny_ * sizeof(double), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize(); // Ensure copy completes
  }

  /**
   * @brief Copies data from the GPU matrix to a host std::vector.
   * @param host_data The host vector to copy data into.
   * @note Resizes host_data if necessary.
   */
  __host__ void copy_to_host(std::vector<double>& host_data) const {
    host_data.resize(nx_ * ny_);
    cudaMemcpy(host_data.data(), data_, nx_ * ny_ * sizeof(double), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize(); // Ensure copy completes
  }

  /**
   * @brief Fills the entire matrix with a specified value.
   * @param value The value to fill the matrix with.
   */
  __host__ void fill(double value) {
    // For arbitrary values, a dedicated kernel would be more efficient for large matrices.
    // For simplicity, using a host loop with unified memory, which CUDA can optimize.
    for (std::size_t idx = 0; idx < nx_ * ny_; ++idx) {
        data_[idx] = value;
    }
  }


private:
  std::size_t nx_; ///< Number of columns (x-dimension size)
  std::size_t ny_; ///< Number of rows (y-dimension size)
  double *data_;   ///< Pointer to the data in CUDA managed memory
  bool is_copy_;   ///< Flag to indicate if this object is a shallow copy
};

#endif // SWE_MATRIX_GPU_HH