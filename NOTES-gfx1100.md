# RDNA (wave32, e.g. gfx1100) note for the xgboost submodule

This branch points the `xgboost` submodule at `yananlong/xgboost`, branch
`rocm/gfx1100-wave-size`, which is `release/3.2.0` (0aa24c5e) plus one commit:
"Fix HIP GPU split dispatch for runtime warp size" (0f719f3).

Why: `EvaluateSplitsKernel` in `src/tree/gpu_hist/evaluate_splits.cu` is launched
with a hard-coded 64-thread block on HIP (CDNA wave64 assumption). On RDNA GPUs
the wave size is 32, so the warp-level reductions in the kernel no longer match
the block size and split evaluation can silently return wrong splits.

Fix: the HIP launch queries the device warp size at runtime (32 or 64) and
launches the kernel with a matching block size, with a device-side check that
`blockDim.x == warpSize`. Behaviour on wave64 devices is unchanged.

Related upstream PR: AMD-Ecosystem/xgboost PR #17.
