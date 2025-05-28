#ifndef XDMF_WRITER_HH
#define XDMF_WRITER_HH

#include <vector>
#include <string>
#include <cstddef> // For std::size_t

/**
 * @brief This class is used to write the mesh and the solution of the SWE to an XDMF file that read HDF5 files.
 */
class XDMFWriter
{
public:
  /**
   * @brief Constructor for the XDMFWriter class.
   * @param filename_prefix The prefix of the filename to write. This will also be the directory name.
   * @param nx The number of cells in the x direction.
   * @param ny The number of cells in the y direction.
   * @param size_x The size of the domain in the x direction (e.g., in km).
   * @param size_y The size of the domain in the y direction (e.g., in km).
   * @param topography Topography field (one value per cell).
   */
  XDMFWriter(const std::string& filename_prefix,
             const std::size_t nx,
             const std::size_t ny,
             const double domain_size_x, // Changed from std::size_t to double to match SWESolver
             const double domain_size_y, // Changed from std::size_t to double to match SWESolver
             const std::vector<double>& topography);

  /**
   * @brief Adds a new solution step to the XDMF file.
   *
   * It updates the XDMF file to include the new solution step and
   * writes the solution to an HDF5 file.
   *
   * @brief h The water height solution to write.
   * @brief t The solution's instant.
   */
  void add_h(const std::vector<double>& h, const double t);

private:
  /**
   * @brief Write the root XDMF file that calls the HDF5 files for the
   * mesh and the solution at the time steps.
   *
   * This function may be called after each time step to update the XDMF file
   * for including the last generated solution.
   */
  void write_root_xdmf() const;

  /**
   * @brief Write the topography to an HDF5 file.
   * @param topography The topography field to write.
   */
  void write_topography_hdf5(const std::vector<double>& topography) const;

  /**
   * @brief Create the vertices of the mesh.
   * @param vertices The vector to store the vertices.
   */
  void create_vertices(std::vector<double>& vertices) const;

  /**
   * @brief Create the cells of the mesh.
   * @param cells The vector to store the cells.
   */
  void create_cells(std::vector<int>& cells) const;

  /**
   * @brief Write the mesh (vertices + cells) to an HDF5 file.
   */
  void write_mesh_hdf5() const;

  /**
   * @brief Write a 1D array to an HDF5 file.
   * @param filename The name of the HDF5 file.
   * @param dataset_name The name of the dataset to write.
   * @param data The data to write.
   */
  static void write_array_to_hdf5(const std::string& filename,
                                  const std::string& dataset_name,
                                  const std::vector<double>& data);

  /// @brief The prefix of the filename to write (also the directory name).
  const std::string filename_prefix_;
  /// @brief The number of cells in the x direction.
  const std::size_t nx_;
  /// @brief The number of cells in the y direction.
  const std::size_t ny_;
  /// @brief The size of the domain in the x direction.
  const double size_x_; // Changed from std::size_t
  /// @brief The size of the domain in the y direction.
  const double size_y_; // Changed from std::size_t
  /// @brief The time steps of the solution.
  std::vector<double> time_steps_;
};

#endif // XDMF_WRITER_HH