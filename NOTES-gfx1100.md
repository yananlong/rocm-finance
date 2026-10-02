# RDNA (wave32, e.g. gfx1100) note for the xgboost submodule

This branch points the `xgboost` submodule at `yananlong/xgboost`, branch
`rocm/gfx1100-wave-size`, which is `release/3.2.0` (0aa24c5e) plus three commits:

1. "Fix HIP GPU split dispatch for runtime warp size" (0f719f3)
2. "Fix HIP multi-target GPU split evaluation for 32-wide wavefronts" (7c9b868)
3. "Use dh::WarpThreads(ordinal) for the HIP single-target split launch" (b4af261)

The submodule gitlink is at b4af261.

The GPU wavefront size is 32 on RDNA (gfx10+, e.g. gfx1100) and 64 on GCN/CDNA
(gfx9, e.g. gfx942). The upstream HIP port assumed 64 in two places, which
silently breaks split evaluation on RDNA. Nothing is hard-coded to a particular
architecture: the host queries the warp size at runtime from
`hipDeviceAttributeWarpSize`, and only 32 or 64 is accepted (anything else is a
fatal error).

## Fix 1: single-target `EvaluateSplitsKernel` launch

`EvaluateSplitsKernel` in `src/tree/gpu_hist/evaluate_splits.cu` was launched
with a hard-coded 64-thread block on HIP (CDNA wave64 assumption). The kernel
requires block size == warp size, so on RDNA the warp-level reductions no longer
matched the block size and split evaluation could silently return wrong splits.

The HIP launch now calls `dh::WarpThreads(ctx->Ordinal())`, which reads
`hipDeviceAttributeWarpSize`, and instantiates the kernel with a matching block
size (32 or 64) through a single launch lambda. The device-side check
`blockDim.x == warpSize` is kept. Behaviour on wave64 devices is unchanged.

## Fix 2: multi-target split evaluation (`multi_strategy="multi_output_tree"`)

`dh::WarpThreads()` in `src/common/device_helpers.hip.h` was a constexpr 64. The
multi-target evaluator (`src/tree/gpu_hist/multi_evaluate_splits.cu`) treats every
`dh::WarpThreads()` threads of a 512-thread block as one warp: it indexes the
per-warp `cub::WarpScan` / `cub::WarpReduce` temp storage with it, strides the bin
loop with it and sizes the launch grid with it. On wave32 each "64-thread warp"
was really two independent wavefronts, so the prefix scan restarted at lane 32,
each half picked its own best bin, and training aborted on invalid splits.

Now:
- Device code: `dh::WarpThreads()` is device-only and returns the wavefront size
  of the offload target being compiled (`HIPCUB_DEVICE_WARP_THREADS`). Host code
  cannot call it, so it cannot silently get a value for the wrong target.
- Host code: `dh::WarpThreads(ordinal)` queries `hipDeviceAttributeWarpSize` at
  runtime (32 or 64 only, otherwise fatal). The multi-target launch grid is sized
  with this queried value.

The marker string `XGB_HIP_WARP_DISPATCH_V4` appears in the unsupported-warp-size
error in `device_helpers.hip.h`.

## Testing

On gfx1100 a 26-case suite passes 26/26 (the previous revision, with only fix 1,
passed 24/26: the multi-target cases failed). Default-path predictions are
bit-identical to the previous revision.

## Upstream

AMD-Ecosystem/xgboost PR #17 covers only fix 1 (the single-target launch). Fix 2
(multi-target `dh::WarpThreads`) is not part of that PR.
