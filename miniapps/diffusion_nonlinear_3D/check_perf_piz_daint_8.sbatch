#!/bin/bash -l
#SBATCH --job-name="8jrjl_perf"
#SBATCH --partition=normal
#SBATCH --time=01:00:00
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --partition=normal
#SBATCH --constraint=gpu
#SBATCH --account=c23
#SBATCH --output="perf_test-%j.out"
#SBATCH --error="perf_test-%j.err"

module load daint-gpu
module unload PrgEnv-cray
module load PrgEnv-gnu
module load cudatoolkit

# Note that this depends on a local Julia install and
# some corresponding environment variables similar to
# the following are likely in your login file, so make sure they're up to date
export PATH=$SCRATCH/julia_local/julia/bin/:$PATH
export JULIA_CUDA_USE_BINARYBUILDER=false
export JULIA_DEPOT_PATH=$SCRATCH/julia_local/julia_depot
# See https://github.com/psanan/piz_daint_julia_gpu4geo_install

# These locations are relative to this directory, so you must launch from here
LOC_A=.
SCRIPT_A=diff_3D_nonlin_multixpu_perf.jl

# This one aims to simply deactivate plots
LOC_B=ref
SCRIPT_B=diff_3D_nonlin_multixpu_perf_noplots.jl

# This one comments out the @hide_communication, since
# that might actually *help* for a single node
LOC_C=ref
SCRIPT_C=diff_3D_nonlin_multixpu_perf_noplots_nohide.jl

# This is the one used for the Ludovic's weak scaling test.
# It has even more monitoring stuff commented out, and a GC.gc() before tic()
LOC_D=ref
SCRIPT_D=diff_3D_nonlin_multixpu_perf_mod.jl


JULIA="julia --check-bounds=no -O3"

# Note that we don't use e.g --project=$LOC_A,
# so the things you need must be in the default environment,
# including a dev'd JustRelax.jl


# Problem size
export RES=512
export NX=$RES
export NY=$RES
export NZ=$RES


export USE_GPU=true
printf "We mostly care about A performing as well or better than B and C\n"

printf "\n====== A (GPU) =====\n"
srun bash -c "LD_PRELOAD=\"/usr/lib64/libcuda.so:/usr/local/cuda/lib64/libcudart.so\" $JULIA $LOC_A/$SCRIPT_A"

printf "\n====== B (GPU) =====\n"
srun bash -c "LD_PRELOAD=\"/usr/lib64/libcuda.so:/usr/local/cuda/lib64/libcudart.so\" $JULIA $LOC_B/$SCRIPT_B"

printf "\n====== C (GPU) =====\n"
srun bash -c "LD_PRELOAD=\"/usr/lib64/libcuda.so:/usr/local/cuda/lib64/libcudart.so\" $JULIA $LOC_C/$SCRIPT_C"

printf "\n====== D (GPU) =====\n"
srun bash -c "LD_PRELOAD=\"/usr/lib64/libcuda.so:/usr/local/cuda/lib64/libcudart.so\" $JULIA $LOC_D/$SCRIPT_D"

# To save resources, skipping slower, threaded tests

#export USE_GPU=false
#
#printf "\n====== A =====\n"
#srun  $JULIA  $LOC_A/$SCRIPT_A
#
#printf "\n====== B =====\n"
#srun $JULIA $LOC_B/$SCRIPT_B
#
#printf "\n====== C =====\n"
#srun $JULIA $LOC_C/$SCRIPT_C
#
#printf "\n====== D =====\n"
#srun $JULIA $LOC_D/$SCRIPT_D


printf "\n====== Finished =====\n"
