from __future__ import annotations

import time
from typing import Tuple

import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


if TRITON_AVAILABLE:
    @triton.jit
    def _matmul_kernel_fused_dequant_row(
        h_ptr, Wd_q_ptr, Wd_scale_ptr, hB_ptr, At_ptr, c_ptr,
        M, N, K, r,
        stride_hm, stride_hk,
        stride_wk, stride_wn,
        stride_sc,
        stride_hBm, stride_hBr,
        stride_Atr, stride_Atn,
        stride_cm, stride_cn,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        GROUP_M: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        num_pid_m = tl.cdiv(M, BLOCK_M)
        num_pid_n = tl.cdiv(N, BLOCK_N)
        num_pid_in_group = GROUP_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
        pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m

        offs_m = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
        offs_n = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
        offs_k = tl.arange(0, BLOCK_K)

        h_ptrs = h_ptr + offs_m[:, None] * stride_hm + offs_k[None, :] * stride_hk
        Wd_q_ptrs = Wd_q_ptr + offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn

        n_mask = offs_n < N
        Wd_scale = tl.load(Wd_scale_ptr + offs_n * stride_sc, mask=n_mask, other=1.0).to(tl.float16)

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k in range(tl.cdiv(K, BLOCK_K)):
            k_mask = offs_k < K - k * BLOCK_K
            h_frag = tl.load(h_ptrs, mask=k_mask[None, :], other=0.0)
            Wd_q_frag = tl.load(Wd_q_ptrs, mask=k_mask[:, None], other=0)
            Wd_fp_frag = Wd_q_frag.to(tl.float16) * Wd_scale[None, :]
            acc += tl.dot(h_frag, Wd_fp_frag)
            h_ptrs += BLOCK_K * stride_hk
            Wd_q_ptrs += BLOCK_K * stride_wk

        offs_m_hB = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        m_mask = offs_m_hB < M
        for rr in range(r):
            hBr = tl.load(hB_ptr + offs_m_hB * stride_hBm + rr * stride_hBr, mask=m_mask, other=0.0)
            Atr = tl.load(At_ptr + rr * stride_Atr + offs_n * stride_Atn, mask=n_mask, other=0.0)
            acc += hBr[:, None].to(tl.float32) * Atr[None, :].to(tl.float32)

        c_offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        c_offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        c_ptrs = c_ptr + stride_cm * c_offs_m[:, None] + stride_cn * c_offs_n[None, :]
        c_mask = (c_offs_m[:, None] < M) & (c_offs_n[None, :] < N)
        tl.store(c_ptrs, acc.to(tl.float16), mask=c_mask)


def quantize_Wd_per_row_int8(Wd: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    scale = Wd.float().abs().amax(dim=1, keepdim=True).clamp(min=1e-12) / 127
    Wd_q = (Wd.float() / scale).round().clamp(-127, 127).to(torch.int8)
    return Wd_q.t().contiguous(), scale.squeeze(1).to(torch.float16)


def make_kernel_fused_dequant_row(BLOCK_M=128, BLOCK_N=128, BLOCK_K=64, GROUP_M=4, num_warps=8, num_stages=3):
    def call(h, Wd_q_t, Wd_scale, A, B):
        if not TRITON_AVAILABLE:
            Wd_fp_t = Wd_q_t.to(torch.float16) * Wd_scale[None, :]
            return h @ Wd_fp_t + (h @ B.t()) @ A.t()
        M, K = h.shape
        _, N = Wd_q_t.shape
        r = A.shape[1]
        hB = h @ B.t()
        At = A.t().contiguous()
        C = torch.empty((M, N), device=h.device, dtype=torch.float16)
        grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"]),)
        _matmul_kernel_fused_dequant_row[grid](
            h, Wd_q_t, Wd_scale, hB, At, C,
            M, N, K, r,
            h.stride(0), h.stride(1),
            Wd_q_t.stride(0), Wd_q_t.stride(1),
            Wd_scale.stride(0),
            hB.stride(0), hB.stride(1),
            At.stride(0), At.stride(1),
            C.stride(0), C.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, GROUP_M=GROUP_M,
            num_warps=num_warps, num_stages=num_stages,
        )
        return C
    call.kernel_name = f"archead_fused_dequant_bm{BLOCK_M}_bn{BLOCK_N}_bk{BLOCK_K}"
    return call


def median_ms(fn, *, warmup: int = 10, iters: int = 50) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    vals = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        vals.append((time.perf_counter() - t0) * 1000.0)
    vals.sort()
    return vals[len(vals) // 2]

