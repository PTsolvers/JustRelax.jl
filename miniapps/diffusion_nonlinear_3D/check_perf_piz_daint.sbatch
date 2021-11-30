#!/bin/bash -l
#SBATCH --job-name="jrjl_perf"
#SBATCH --partition=normal
#SBATCH --time=01:00:00
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

LOC_B=$SCRATCH/JustRelax.jl/miniapps/diffusion_nonlinear_3D
SCRIPT_B=diff_3D_nonlin_multixpu_perf.jl

LOC_A=$SCRATCH/PseudoTransientDiffusion.jl/scripts/diff_3D
SCRIPT_A=diff_3D_nonlin_multixpu_perf_noplots.jl

# This is the one used for the weak scaling test.
# It has even more monitoring stuff commented out, and a GC.gc() before tic()
#SCRIPT_A=diff_3D_nonlin_multixpu_perf_mod.jl

# This one comments out the @hide_communication, since
# that might actually *help* for a single node
LOC_C=$SCRATCH/PseudoTransientDiffusion.jl/scripts/diff_3D
SCRIPT_C=diff_3D_nonlin_multixpu_perf_noplots_nohide.jl


JULIA="julia --check-bounds=no -O3"

# Note that we don't use e.g --project=$LOC_A,
# so the things you need must be in the default environment,
# including a dev'd JustRelax.jl

# Smoke test for running anything at all with Julia
# (useful since we are using our own install of Julia)
srun $JULIA hello.jl


# Problem size
export RES=512
export NX=$RES
export NY=$RES
export NZ=$RES


export USE_GPU=true
printf "See if A, B, and C perform the same!\n"

printf "\n====== A (GPU) =====\n"
srun bash -c "LD_PRELOAD=\"/usr/lib64/libcuda.so:/usr/local/cuda/lib64/libcudart.so\" $JULIA $LOC_A/$SCRIPT_A"

printf "\n====== B (GPU) =====\n"
srun bash -c "LD_PRELOAD=\"/usr/lib64/libcuda.so:/usr/local/cuda/lib64/libcudart.so\" $JULIA $LOC_B/$SCRIPT_B"

printf "\n====== C (GPU) =====\n"
srun bash -c "LD_PRELOAD=\"/usr/lib64/libcuda.so:/usr/local/cuda/lib64/libcudart.so\" $JULIA $LOC_C/$SCRIPT_C"



export USE_GPU=false
printf "See if A, B, and C perform the same!\n"

printf "\n====== A =====\n"
srun  $JULIA  $LOC_A/$SCRIPT_A

printf "\n====== B =====\n"
srun $JULIA $LOC_B/$SCRIPT_B

printf "\n====== C =====\n"
srun $JULIA $LOC_C/$SCRIPT_C

printf "\n====== Finished =====\n"
