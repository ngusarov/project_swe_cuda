// swe.cc
#include "swe.hh"
#include "xdmf_writer.hh" // Keep this for output

#include <iostream>
#include <cstddef>
#include <vector>
#include <string>
#include <cassert>
#include <hdf5.h>
#include <hdf5_hl.h>
#include <cstdio>
#include <cmath>
#include <memory>
#include <algorithm> // For std::fill, std::swap

namespace
{

// This function remains for reading HDF5 to host std::vector
void
read_2d_array_from_DF5(const std::string &filename,
                       const std::string &dataset_name,
                       std::vector<double> &data,
                       std::size_t &nx, // Output parameter
                       std::size_t &ny) // Output parameter
{
  hid_t file_id, dataset_id, dataspace_id;
  hsize_t dims[2];
  herr_t status;

  file_id = H5Fopen(filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
  if (file_id < 0)
  {
    std::cerr << "Error opening HDF5 file: " << filename << std::endl;
    return;
  }

  dataset_id = H5Dopen2(file_id, dataset_name.c_str(), H5P_DEFAULT);
  if (dataset_id < 0)
  {
    std::cerr << "Error opening dataset: " << dataset_name << std::endl;
    H5Fclose(file_id);
    return;
  }

  dataspace_id = H5Dget_space(dataset_id);
  if (dataspace_id < 0)
  {
    std::cerr << "Error getting dataspace" << std::endl;
    H5Dclose(dataset_id);
    H5Fclose(file_id);
    return;
  }

  status = H5Sget_simple_extent_dims(dataspace_id, dims, NULL);
  if (status < 0)
  {
    std::cerr << "Error getting dimensions" << std::endl;
    H5Sclose(dataspace_id);
    H5Dclose(dataset_id);
    H5Fclose(file_id);
    return;
  }
  nx = dims[0]; // Set nx from HDF5
  ny = dims[1]; // Set ny from HDF5

  data.resize(nx * ny);

  status = H5Dread(dataset_id, H5T_IEEE_F64LE, H5S_ALL, H5S_ALL, H5P_DEFAULT, data.data());
  if (status < 0)
  {
    std::cerr << "Error reading data" << std::endl;
    H5Sclose(dataspace_id);
    H5Dclose(dataset_id);
    H5Fclose(file_id);
    data.clear();
    return;
  }

  H5Sclose(dataspace_id);
  H5Dclose(dataset_id);
  H5Fclose(file_id);
}

// Helper for 2D indexing on host vectors (if still needed for init functions)
inline double &at_host(std::vector<double> &vec, const std::size_t i, const std::size_t j, const std::size_t local_nx)
{
  return vec[j * local_nx + i];
}

inline const double &at_host(const std::vector<double> &vec, const std::size_t i, const std::size_t j, const std::size_t local_nx)
{
  return vec[j * local_nx + i];
}

} // namespace

SWESolver::SWESolver(const int test_case_id, const std::size_t nx, const std::size_t ny) :
  nx_(nx), ny_(ny), // Initialize nx_ and ny_ FIRST
  size_x_(500.0), size_y_(500.0),
  dx_(size_x_ / nx_), dy_(size_y_ / ny_), // Then use them for dx_ and dy_
  reflective_(true)
{
  // Assert minimum grid size for kernel validity
  assert(nx_ >= 3 && ny_ >= 3 && "Grid dimensions must be at least 3x3 for inner cell computations and boundary conditions.");

  // Allocate GPU matrices
  h0_gpu_ = std::make_unique<SWEMatrixGPU>(nx_, ny_);
  h1_gpu_ = std::make_unique<SWEMatrixGPU>(nx_, ny_);
  hu0_gpu_ = std::make_unique<SWEMatrixGPU>(nx_, ny_);
  hu1_gpu_ = std::make_unique<SWEMatrixGPU>(nx_, ny_);
  hv0_gpu_ = std::make_unique<SWEMatrixGPU>(nx_, ny_);
  hv1_gpu_ = std::make_unique<SWEMatrixGPU>(nx_, ny_);
  z_gpu_ = std::make_unique<SWEMatrixGPU>(nx_, ny_);
  zdx_gpu_ = std::make_unique<SWEMatrixGPU>(nx_, ny_);
  zdy_gpu_ = std::make_unique<SWEMatrixGPU>(nx_, ny_);

  assert(test_case_id == 1 || test_case_id == 2);
  if (test_case_id == 1)
  {
    this->reflective_ = true;
    this->init_gaussian();
  }
  else if (test_case_id == 2)
  {
    this->reflective_ = false;
    this->init_dummy_tsunami();
  }
  else
  {
    assert(false);
  }
}

SWESolver::SWESolver(const std::string &h5_file, const double domain_size_x, const double domain_size_y) :
  size_x_(domain_size_x), size_y_(domain_size_y), // Initialize size_x_, size_y_ first
  nx_(0), ny_(0), // Initialize nx_, ny_ to 0, they will be set by init_from_HDF5_file_data_load
  dx_(0.0), dy_(0.0), // Initialize dx_, dy_ to 0.0
  reflective_(false)
{
  // Step 1: Load data from HDF5, this will set member variables nx_ and ny_
  this->init_from_HDF5_file_data_load(h5_file);

  // Step 2: Now that nx_ and ny_ are known, calculate dx_ and dy_
  if (nx_ == 0 || ny_ == 0) {
    std::cerr << "Error: nx or ny is zero after reading from HDF5 file '" << h5_file << "'." << std::endl;
    assert(false && "nx or ny is zero after HDF5 read.");
  }
  dx_ = size_x_ / nx_;
  dy_ = size_y_ / ny_;

  // Step 3: Assert minimum grid size
  assert(nx_ >= 3 && ny_ >= 3 && "Grid dimensions must be at least 3x3 for inner cell computations and boundary conditions.");

  // Step 4: Allocate GPU matrices AFTER nx_ and ny_ are known
  h0_gpu_ = std::make_unique<SWEMatrixGPU>(nx_, ny_);
  h1_gpu_ = std::make_unique<SWEMatrixGPU>(nx_, ny_);
  hu0_gpu_ = std::make_unique<SWEMatrixGPU>(nx_, ny_);
  hu1_gpu_ = std::make_unique<SWEMatrixGPU>(nx_, ny_);
  hv0_gpu_ = std::make_unique<SWEMatrixGPU>(nx_, ny_);
  hv1_gpu_ = std::make_unique<SWEMatrixGPU>(nx_, ny_);
  z_gpu_ = std::make_unique<SWEMatrixGPU>(nx_, ny_);
  zdx_gpu_ = std::make_unique<SWEMatrixGPU>(nx_, ny_);
  zdy_gpu_ = std::make_unique<SWEMatrixGPU>(nx_, ny_);

  // Step 5: Copy host data (which was loaded by init_from_HDF5_file_data_load) to GPU
  h0_gpu_->copy_from_host(this->h0_host_);
  hu0_gpu_->copy_from_host(this->hu0_host_);
  hv0_gpu_->copy_from_host(this->hv0_host_);
  z_gpu_->copy_from_host(this->z_host_);

  // Step 6: Initialize other GPU matrices
  h1_gpu_->fill(0.0);
  hu1_gpu_->fill(0.0);
  hv1_gpu_->fill(0.0);

  // Step 7: Initialize derivatives
  this->init_dx_dy();
}

void
SWESolver::init_from_HDF5_file_data_load(const std::string &h5_file)
{
  // Read data into host vectors. This call sets this->nx_ and this->ny_
  read_2d_array_from_DF5(h5_file, "h0", this->h0_host_, this->nx_, this->ny_);
  read_2d_array_from_DF5(h5_file, "hu0", this->hu0_host_, this->nx_, this->ny_); // nx_, ny_ are re-read, should be consistent
  read_2d_array_from_DF5(h5_file, "hv0", this->hv0_host_, this->nx_, this->ny_); // nx_, ny_ are re-read, should be consistent
  read_2d_array_from_DF5(h5_file, "topography", this->z_host_, this->nx_, this->ny_); // nx_, ny_ are re-read, should be consistent
}

void
SWESolver::init_gaussian()
{
  hu0_host_.resize(nx_ * ny_, 0.0);
  hv0_host_.resize(nx_ * ny_, 0.0);
  std::fill(hu0_host_.begin(), hu0_host_.end(), 0.0);
  std::fill(hv0_host_.begin(), hv0_host_.end(), 0.0);

  h0_host_.clear();
  h0_host_.reserve(nx_ * ny_);

  const double x0_0 = size_x_ / 4.0;
  const double y0_0 = size_y_ / 3.0;
  const double x0_1 = size_x_ / 2.0;
  const double y0_1 = 0.75 * size_y_;

  for (std::size_t j = 0; j < ny_; ++j)
  {
    for (std::size_t i = 0; i < nx_; ++i)
    {
      const double x = dx_ * (static_cast<double>(i) + 0.5);
      const double y = dy_ * (static_cast<double>(j) + 0.5);
      const double gauss_0 = 10.0 * std::exp(-((x - x0_0) * (x - x0_0) + (y - y0_0) * (y - y0_0)) / 1000.0);
      const double gauss_1 = 10.0 * std::exp(-((x - x0_1) * (x - x0_1) + (y - y0_1) * (y - y0_1)) / 1000.0);

      h0_host_.push_back(10.0 + gauss_0 + gauss_1);
    }
  }

  z_host_.resize(this->h0_host_.size());
  std::fill(z_host_.begin(), z_host_.end(), 0.0);

  // Copy host data to GPU
  h0_gpu_->copy_from_host(this->h0_host_);
  hu0_gpu_->copy_from_host(this->hu0_host_);
  hv0_gpu_->copy_from_host(this->hv0_host_);
  z_gpu_->copy_from_host(this->z_host_);

  // Initialize other GPU matrices
  h1_gpu_->fill(0.0);
  hu1_gpu_->fill(0.0);
  hv1_gpu_->fill(0.0);

  this->init_dx_dy();
}

void
SWESolver::init_dummy_tsunami()
{
  hu0_host_.resize(nx_ * ny_);
  hv0_host_.resize(nx_ * ny_);
  std::fill(hu0_host_.begin(), hu0_host_.end(), 0.0);
  std::fill(hv0_host_.begin(), hv0_host_.end(), 0.0);

  h0_host_.resize(nx_ * ny_);
  z_host_.resize(nx_ * ny_);

  const double x0_0 = 0.6 * size_x_;
  const double y0_0 = 0.6 * size_y_;
  const double x0_1 = 0.4 * size_x_;
  const double y0_1 = 0.4 * size_y_;
  const double x0_2 = 0.7 * size_x_;
  const double y0_2 = 0.3 * size_y_;

  // Creating topography and initial water height on host
  for (std::size_t j = 0; j < ny_; ++j)
  {
    for (std::size_t i = 0; i < nx_; ++i)
    {
      const double x = dx_ * (static_cast<double>(i) + 0.5);
      const double y = dy_ * (static_cast<double>(j) + 0.5);

      const double gauss_0 = 2.0 * std::exp(-((x - x0_0) * (x - x0_0) + (y - y0_0) * (y - y0_0)) / 3000.0);
      const double gauss_1 = 3.0 * std::exp(-((x - x0_1) * (x - x0_1) + (y - y0_1) * (y - y0_1)) / 10000.0);
      const double gauss_2 = 5.0 * std::exp(-((x - x0_2) * (x - x0_2) + (y - y0_2) * (y - y0_2)) / 100.0);

      const double z = -1.0 + gauss_0 + gauss_1;
      at_host(z_host_, i, j, nx_) = z;

      double h0 = z < 0.0 ? -z + gauss_2 : 0.00001;
      at_host(h0_host_, i, j, nx_) = h0;
    }
  }

  // Copy host data to GPU
  h0_gpu_->copy_from_host(this->h0_host_);
  hu0_gpu_->copy_from_host(this->hu0_host_);
  hv0_gpu_->copy_from_host(this->hv0_host_);
  z_gpu_->copy_from_host(this->z_host_);

  // Initialize other GPU matrices
  h1_gpu_->fill(0.0);
  hu1_gpu_->fill(0.0);
  hv1_gpu_->fill(0.0);

  this->init_dx_dy();
}

void
SWESolver::init_dummy_slope()
{
  hu0_host_.resize(nx_ * ny_);
  hv0_host_.resize(nx_ * ny_);
  std::fill(hu0_host_.begin(), hu0_host_.end(), 0.0);
  std::fill(hv0_host_.begin(), hv0_host_.end(), 0.0);

  h0_host_.resize(nx_ * ny_);
  z_host_.resize(nx_ * ny_);

  const double dz = 10.0;

  // Creating topography and initial water height on host
  for (std::size_t j = 0; j < ny_; ++j)
  {
    for (std::size_t i = 0; i < nx_; ++i)
    {
      const double x = dx_ * (static_cast<double>(i) + 0.5);
      const double y = dy_ * (static_cast<double>(j) + 0.5);
      static_cast<void>(y); // Suppress unused variable warning

      const double z = -10.0 - 0.5 * dz + dz / size_x_ * x;
      at_host(z_host_, i, j, nx_) = z;

      double h0 = z < 0.0 ? -z : 0.00001;
      at_host(h0_host_, i, j, nx_) = h0;
    }
  }

  // Copy host data to GPU
  h0_gpu_->copy_from_host(this->h0_host_);
  hu0_gpu_->copy_from_host(this->hu0_host_);
  hv0_gpu_->copy_from_host(this->hv0_host_);
  z_gpu_->copy_from_host(this->z_host_);

  // Initialize other GPU matrices
  h1_gpu_->fill(0.0);
  hu1_gpu_->fill(0.0);
  hv1_gpu_->fill(0.0);

  this->init_dx_dy();
}

void
SWESolver::init_dx_dy()
{
  // Allocate host vectors for zdx and zdy
  zdx_host_.resize(z_host_.size(), 0.0); // Use z_host_ size which is set
  zdy_host_.resize(z_host_.size(), 0.0);

  // Perform calculation on host (or could be a small kernel)
  for (std::size_t j = 1; j < ny_ - 1; ++j)
  {
    for (std::size_t i = 1; i < nx_ - 1; ++i)
    {
      at_host(this->zdx_host_, i, j, nx_) = 0.5 * (at_host(this->z_host_, i + 1, j, nx_) - at_host(this->z_host_, i - 1, j, nx_)) / dx_;
      at_host(this->zdy_host_, i, j, nx_) = 0.5 * (at_host(this->z_host_, i, j + 1, nx_) - at_host(this->z_host_, i, j - 1, nx_)) / dy_;
    }
  }

  // Copy host zdx and zdy to GPU
  zdx_gpu_->copy_from_host(this->zdx_host_);
  zdy_gpu_->copy_from_host(this->zdy_host_);
}

void
SWESolver::solve(const double Tend, const bool full_log, const std::size_t output_n, const std::string &fname_prefix)
{
  std::shared_ptr<XDMFWriter> writer;
  // Host vector for writing output
  std::vector<double> h_output_host;

  if (output_n > 0)
  {
    // Copy initial topography to host for XDMFWriter
    z_gpu_->copy_to_host(z_host_);
    // Use nx_ and ny_ which are member variables, size_x_ and size_y_ are also members.
    writer = std::make_shared<XDMFWriter>(fname_prefix, this->nx_, this->ny_, static_cast<std::size_t>(this->size_x_), static_cast<std::size_t>(this->size_y_), this->z_host_);
    // Copy initial h0 from GPU to host for writing
    h0_gpu_->copy_to_host(h_output_host);
    writer->add_h(h_output_host, 0.0);
  }

  double T = 0.0;

  // Pointers to GPU matrices for current and next time steps
  SWEMatrixGPU* h_current = h0_gpu_.get();
  SWEMatrixGPU* hu_current = hu0_gpu_.get();
  SWEMatrixGPU* hv_current = hv0_gpu_.get();

  SWEMatrixGPU* h_next = h1_gpu_.get();
  SWEMatrixGPU* hu_next = hu1_gpu_.get();
  SWEMatrixGPU* hv_next = hv1_gpu_.get();

  std::cout << "Solving SWE on GPU..." << std::endl;

  std::size_t nt = 1;
  while (T < Tend)
  {
    // Compute time step on GPU
    const double dt = this->compute_time_step(*h_current, *hu_current, *hv_current, T, Tend);

    const double T1 = T + dt;

    printf("Computing T: %2.4f hr  (dt = %.2e s) -- %3.3f%%", T1, dt * 3600, 100 * T1 / Tend);
    std::cout << (full_log ? "\n" : "\r") << std::flush;

    // Update boundary conditions on GPU
    this->update_bcs(*h_current, *hu_current, *hv_current, *h_next, *hu_next, *hv_next);

    // Solve one step on GPU
    this->solve_step(dt, *h_current, *hu_current, *hv_current, *h_next, *hu_next, *hv_next);

    if (output_n > 0 && nt % output_n == 0)
    {
      // Copy current h from GPU to host for writing
      h_next->copy_to_host(h_output_host);
      writer->add_h(h_output_host, T1);
    }
    ++nt;

    // Swap GPU pointers for current and next solutions
    std::swap(h_current, h_next);
    std::swap(hu_current, hu_next);
    std::swap(hv_current, hv_next);

    T = T1;
  }

  // After the loop, h_current, hu_current, hv_current hold the final solution.
  // If h1_gpu_ is intended to always hold the final solution, copy to it if h_current is h0_gpu_.
  // This ensures h1_gpu_ always contains the last computed state.
  if (h_current == h0_gpu_.get()) { // This means h0_gpu_ now holds the final solution due to an odd number of swaps
      cudaMemcpy(h1_gpu_->data(), h0_gpu_->data(), nx_ * ny_ * sizeof(double), cudaMemcpyDeviceToDevice);
      cudaMemcpy(hu1_gpu_->data(), hu0_gpu_->data(), nx_ * ny_ * sizeof(double), cudaMemcpyDeviceToDevice);
      cudaMemcpy(hv1_gpu_->data(), hv0_gpu_->data(), nx_ * ny_ * sizeof(double), cudaMemcpyDeviceToDevice);
  }
  // If h_current is already h1_gpu_, then h1_gpu_ already holds the final solution.


  if (output_n > 0)
  {
    // Copy final h from GPU to host for writing
    // Ensure we copy from the correct final buffer, which is now consistently in h1_gpu_
    h1_gpu_->copy_to_host(h_output_host);
    writer->add_h(h_output_host, T);
  }

  std::cout << "Finished solving SWE on GPU." << std::endl;
}

double
SWESolver::compute_time_step(const SWEMatrixGPU &h,
                             const SWEMatrixGPU &hu,
                             const SWEMatrixGPU &hv,
                             const double T,
                             const double Tend) const
{
  // Use the GPU function for max_nu_sqr
  double max_nu_sqr = swe_compute_max_nu_sqr_gpu(nx_, ny_, h, hu, hv);

  double dt = std::min(dx_, dy_) / (sqrt(2.0 * max_nu_sqr));
  return std::min(dt, Tend - T);
}

void
SWESolver::solve_step(const double dt,
                      const SWEMatrixGPU &h0,
                      const SWEMatrixGPU &hu0,
                      const SWEMatrixGPU &hv0,
                      SWEMatrixGPU &h,
                      SWEMatrixGPU &hu,
                      SWEMatrixGPU &hv) const
{
  // Call the host wrapper for the CUDA kernel
  swe_solve_step_gpu(nx_, ny_, dt, dx_, dy_,
                     h0, hu0, hv0, *zdx_gpu_, *zdy_gpu_, h, hu, hv);
}

void
SWESolver::update_bcs(const SWEMatrixGPU &h0,
                      const SWEMatrixGPU &hu0,
                      const SWEMatrixGPU &hv0,
                      SWEMatrixGPU &h,
                      SWEMatrixGPU &hu,
                      SWEMatrixGPU &hv) const
{
  // Call the host wrapper for the CUDA boundary condition kernels
  swe_update_bcs_gpu(nx_, ny_, reflective_, h0, hu0, hv0, h, hu, hv);
}