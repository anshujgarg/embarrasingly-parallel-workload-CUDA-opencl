# embarrasingly-parallel-workload-CUDA-opencl
Set of embarassingly parallel (data parallel) workloads written in CUDA and OpenCL


# Corresponding research paper
These workloads were used in the research paper:
"FaaSter: Accelerated Functions-as-a-Service with Heterogeneous GPUs"
https://doi.org/10.1109/HiPC53243.2021.00057

# Description
Each workload has a CUDA and OpenCL version.
A "datagen.c" file is present to generate the input data for the
workload.
Go through comments in the code file for details

# Compilation

## cuda files
nvcc   &lt;filename&gt;   -o&lt;executable_name&gt;

## data gen files
gcc data_gen.c -odata_gen

## opencl files

Each open workload has a "compile.sh" file to compile
the opencl workload.

NOTE that the compilation command are for AMDAPP SDK 3.0 for opencl.
  


