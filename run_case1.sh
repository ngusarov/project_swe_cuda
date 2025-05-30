#!/bin/bash -l

#SBATCH --nodes=1
#SBATCH --time=08:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --qos=math-454
#SBATCH --account=math-454

module purge
module load gcc cuda hdf5

# Compile the SWE solver
# Assuming you have a Makefile or build script that compiles main.cc, swe.cc, swe_cuda_kernels.cu
# Example compilation (adjust as needed for your build system):
# nvcc -std=c++17 main.cc swe.cc swe_cuda_kernels.cu -o swe -I. -lhdf5 -lhdf5_hl

TEST_CASE_ID=1 # Water drops in a box
TEND=0.5      # Simulation time in hours for this case
REPEAT=${1:-3} # Number of repetitions, default to 3

# Problem sizes
declare -a SIZES=(100 200 500 1000)
# declare -a SIZES=(1500 2000)
# Threads per block values

declare -a TPBS=(1 2 4 8 16 32 64 128 192 256 384 512 640 768 896 1024)

echo "Starting benchmarks for Test Case ${TEST_CASE_ID} (Water Drops)"

for SIZE in "${SIZES[@]}"; do
  OUTPUT_FILE="case${TEST_CASE_ID}_nx${SIZE}_nout10.csv"
  echo "nx,ny,threadsPerBlock,repetition,time(s),iterations" > "$OUTPUT_FILE"
  echo "Generating results in ${OUTPUT_FILE}"

  for TPB in "${TPBS[@]}"; do
    for R in $(seq 1 "$REPEAT"); do
      echo "[RUN] Case=${TEST_CASE_ID} | Size=${SIZE}x${SIZE} | ThreadsPerBlock=${TPB} | Repetition=${R}"

      # Use a unique temporary file name for parallel safety
      TMPFILE="tmp_case${TEST_CASE_ID}_nx${SIZE}_tpb${TPB}_r${R}_${RANDOM}.txt"

      # Execute the SWE solver with arguments: test_case_id nx ny Tend threadsPerBlock
      srun ./swe "$TEST_CASE_ID" "$SIZE" "$TEND" "$TPB" | tee "$TMPFILE"

      TIME=$(grep "\[TIMING\]" "$TMPFILE" | awk '{print $(NF-1)}')
      ITERS=$(grep "\[ITERATIONS\]" "$TMPFILE" | awk '{print $(NF-0)}')

      if [[ -n "$TIME" && -n "$ITERS" ]]; then
        echo "$SIZE,$SIZE,$TPB,$R,$TIME,$ITERS" >> "$OUTPUT_FILE"
      else
        echo "WARNING: Missing data for Case ${TEST_CASE_ID} Size ${SIZE}x${SIZE} TPB=${TPB} Repetition=${R}"
      fi

      rm -f "$TMPFILE"
    done
  done
  echo "Completed benchmarks for Case ${TEST_CASE_ID}, Size ${SIZE}x${SIZE}. Results saved to ${OUTPUT_FILE}"
done

echo "All benchmarks for Test Case ${TEST_CASE_ID} completed."