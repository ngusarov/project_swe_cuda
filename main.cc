#include "swe.hh"

#include <string>
#include <cstddef>
#include <iostream>
#include <chrono>   // For timing
#include <cstdlib>  // For std::atoi, std::stod

int
main(int argc, char **argv)
{
  // Default values
  int test_case_id = 1;
  double Tend = 0.01;     // Default for water drops
  std::size_t nx = 100;
  std::size_t ny = 100;
  int threadsPerBlock = 256; // Default CUDA threads per block
  // std::string output_fname = "output"; // Default output filename prefix
  bool full_log = false; // Disable full logging for performance runs
  std::size_t output_n = 10; // Disable output file generation for performance runs

  // Parse command line arguments
  if (argc > 1) {
    test_case_id = std::atoi(argv[1]);
  }
  if (argc > 2) {
    nx = std::atoi(argv[2]);
    ny = nx; // Assuming square grid for simplicity in benchmarking
  }
  if (argc > 3) {
    Tend = std::stod(argv[3]);
  }
  if (argc > 4) {
    threadsPerBlock = std::atoi(argv[4]);
  }
  // if (argc > 5) {
  //   output_fname = argv[5]; // Optional: custom output filename prefix
  // }

  std::string output_fname = "output_"+std::to_string(test_case_id)+"_"+std::to_string(nx); // Default output filename prefix

  // Adjust Tend based on test_case_id if not explicitly provided or if defaults are used
  if (argc <= 3) { // If Tend was not provided as an argument
      if (test_case_id == 1) {
          Tend = 0.01;
      } else if (test_case_id == 2) {
          Tend = 1.0;
      } else if (test_case_id == 3) { // For file-based case, use a default Tend
          Tend = 0.2;
      }
  }


  // Initialize SWESolver based on test_case_id
  SWESolver* solver = nullptr;
  if (test_case_id == 1 || test_case_id == 2) {
    solver = new SWESolver(test_case_id, nx, ny, threadsPerBlock);
  } else if (test_case_id == 3) {
    // For test_case_id 3, we need an HDF5 file.
    // The problem sizes (nx, ny) should correspond to the file.
    // We'll assume a naming convention like "Data_nx<size+1>_500km.h5" for simplicity
    // and a fixed domain size of 500km.
    std::string h5_filename = "Data_nx" + std::to_string(nx + 1) + "_500km.h5";
    double domain_size = 500.0; // Assuming 500km for file-based cases
    solver = new SWESolver(h5_filename, domain_size, domain_size, threadsPerBlock);
  } else {
    std::cerr << "Invalid test_case_id: " << test_case_id << std::endl;
    return 1;
  }

  // Measure execution time
  auto t1 = std::chrono::high_resolution_clock::now();
  std::size_t total_iterations = solver->solve(Tend, full_log, output_n, output_fname);
  auto t2 = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = t2 - t1;

  std::cout << "[TIMING] " << elapsed.count() << " [s]" << std::endl;
  std::cout << "[ITERATIONS] " << total_iterations << std::endl;

  delete solver; // Clean up allocated solver object

  return 0;
}