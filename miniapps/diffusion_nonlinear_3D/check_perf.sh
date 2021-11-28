#!/usr/bin/env sh

# A minimalist script to compare performance between two Julia scripts

LOC_A=$HOME/code/PseudoTransientDiffusion.jl/scripts/diff_3D
SCRIPT_A=diff_3D_nonlin_multixpu_perf.jl

LOC_B=$HOME/code/JustRelax.jl/miniapps/diffusion_nonlinear_3D
SCRIPT_B=diff_3D_nonlin_multixpu_perf.jl

JULIA="julia --check-bounds=no -O3"

printf "Confirm that A and B perform the same!\n"

printf "\n====== A =====\n"
$JULIA --project=$LOC_A $LOC_A/$SCRIPT_A


printf "\n====== B =====\n"
$JULIA --project=$LOC_B $LOC_B/$SCRIPT_B
