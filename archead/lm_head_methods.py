from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm

from .modeling import encode
from .triton_archead import make_kernel_fused_dequant_row, median_ms, quantize_Wd_per_row_int8
from .utils import empty_cache


@dataclass
class LMHeadConfig:
    name: str
    core: str = "am"  # am | archead
    core_rank: int = 8
    group_R: int = 64
    row_alpha: float = 0.0
    col_beta: float = 0.0
    metric_power: float = 0.75
    metric_ridge: float = 1e-3
    residual: str = "am_rownorm_i4"  # none | archead_i8 | am_rownorm_i4 | am_norow_i4
    residual_rank: int = 5
    quant_group: int = 64
    packed: bool = True


def lm_config(method: str, model_id: str) -> LMHeadConfig:
    low = model_id.lower()
    if method == "am_lrgq_adaptive":
        if "gemma" in low:
            return LMHeadConfig(method, core="am_sqrt", core_rank=8, metric_power=1.0, residual="am_norow_i4", residual_rank=3)
        return LMHeadConfig(method, core="am", core_rank=8, metric_power=0.75, residual="am_rownorm_i4", residual_rank=5)
    if method == "archead_lm":
        if "gemma" in low:
            return LMHeadConfig(method, core="archead", core_rank=14, group_R=16, row_alpha=1.25, col_beta=0.25, residual="archead_i8", residual_rank=6)
        return LMHeadConfig(method, core="archead", core_rank=10, group_R=64, row_alpha=0.0, col_beta=0.0, residual="archead_i8", residual_rank=6)
    if method == "hybrid_archead_core_amres":
        if "gemma" in low:
            return LMHeadConfig(method, core="archead", core_rank=14, group_R=16, row_alpha=1.25, col_beta=0.25, metric_power=1.0, residual="am_norow_i4", residual_rank=3)
        return LMHeadConfig(method, core="archead", core_rank=10, group_R=64, row_alpha=0.0, col_beta=0.0, metric_power=0.75, residual="am_rownorm_i4", residual_rank=5)
    if method == "hybrid_amcore_archead_res":
        return LMHeadConfig(method, core="am", core_rank=8, metric_power=0.75, residual="archead_i8", residual_rank=6)
    if method == "archead_core_only":
        cfg = lm_config("archead_lm", model_id)
        cfg.name = method
        cfg.residual = "none"
        cfg.residual_rank = 0
        return cfg
    if method == "gptq_int4":
        return LMHeadConfig(name="gptq_int4", core="gptq_int4")
    if method == "archead_unpacked":
        cfg = lm_config("archead_lm", model_id)
        cfg.name = "archead_unpacked"
        cfg.packed = False
        return cfg
    raise ValueError(method)


HEAD_BASELINE_METHODS = {"row_int8", "group_int4", "svd8_group_int4"}


@torch.no_grad()
def compress_gptq(W_cpu: torch.Tensor, h_train_cpu: torch.Tensor, bits: int = 4, group_size: int = 64, damp: float = 0.01, block_size: int = 128, *, device: str) -> CompressedHead:
    """Standalone GPTQ implementation for the LM-head projection W (V x D)."""
    W = W_cpu.to(device=device, dtype=torch.float32)
    H_train = h_train_cpu.to(device=device, dtype=torch.float32)
    V, D = W.shape
    
    H = (H_train.T @ H_train) / H_train.shape[0]
    dead = torch.diag(H) == 0.0
    H[dead, dead] = 1.0
    W[:, dead] = 0.0
    
    damp_val = damp * torch.diag(H).mean()
    H[range(D), range(D)] += damp_val
    
    try:
        H_inv = torch.cholesky_inverse(torch.linalg.cholesky(H))
    except RuntimeError:
        H_inv = torch.inverse(H)
        
    H_inv = torch.linalg.cholesky(H_inv, upper=True)
    
    W_q = torch.zeros_like(W)
    qmax = (1 << (bits - 1)) - 1
    
    for i1 in tqdm(range(0, D, block_size), desc="GPTQ", leave=False):
        i2 = min(i1 + block_size, D)
        count = i2 - i1
        
        W_block = W[:, i1:i2].clone()
        Q_block = torch.zeros_like(W_block)
        Err_block = torch.zeros_like(W_block)
        H_inv_block = H_inv[i1:i2, i1:i2]
        
        for j in range(count):
            global_j = i1 + j
            if global_j % group_size == 0:
                group_end = min(global_j + group_size, D)
                if group_size <= block_size and (j + group_size) <= count:
                    W_group = W_block[:, j:j+group_size]
                else:
                    W_group = W[:, global_j:group_end]
                sc = W_group.abs().amax(dim=1).clamp(min=1e-12) / qmax
                
            w = W_block[:, j]
            q = (w / sc).round().clamp(-qmax, qmax) * sc
            
            Q_block[:, j] = q
            err = (w - q) / H_inv_block[j, j]
            Err_block[:, j] = err
            
            W_block[:, j+1:] -= err.unsqueeze(1) @ H_inv_block[j, j+1:].unsqueeze(0)
            
        W_q[:, i1:i2] = Q_block
        W[:, i1:i2] = Q_block
        
        if i2 < D:
            W[:, i2:] -= Err_block @ H_inv[i1:i2, i2:]

    total_bytes = (V * D * bits) // 8 + (V * (D // group_size)) * 2
    dense_bf16 = V * D * 2
    stats = {
        "name": f"gptq_int{bits}_g{group_size}",
        "vocab_size": V,
        "hidden_size": D,
        "core_bytes": int(total_bytes),
        "residual_bytes": 0,
        "total_bytes": int(total_bytes),
        "byte_ratio_vs_bf16": float(total_bytes / dense_bf16),
        "compression_ratio_vs_bf16": float(dense_bf16 / total_bytes),
    }
    return CompressedHead(W_q, None, None, stats)


@torch.no_grad()
def lowrank_rand(M: torch.Tensor, r: int, oversampling: int = 10) -> Tuple[torch.Tensor, torch.Tensor]:
    m, n = M.shape
    k = min(r + oversampling, min(m, n))
    gen = torch.Generator(device=M.device).manual_seed(0)
    Mf = M.float()
    Omega = torch.randn(n, k, device=M.device, dtype=torch.float32, generator=gen)
    Y = Mf @ Omega
    Q, _ = torch.linalg.qr(Y)
    Bt = Q.t() @ Mf
    U_s, S_s, Vt_s = torch.linalg.svd(Bt, full_matrices=False)
    U = Q @ U_s
    return (U[:, :r] * S_s[:r]).to(M.dtype), Vt_s[:r].to(M.dtype)


def _pad_cols(X: torch.Tensor, g: int):
    pad = (g - X.shape[1] % g) % g
    return (F.pad(X, (0, pad)), pad) if pad else (X, 0)


def pack_int4(X: torch.Tensor) -> torch.Tensor:
    if X.shape[-1] % 2 != 0:
        X = F.pad(X, (0, 1))
    X_uint8 = X.to(torch.uint8) & 0x0F
    return X_uint8[..., 0::2] | (X_uint8[..., 1::2] << 4)


def unpack_int4(X_packed: torch.Tensor) -> torch.Tensor:
    low = (X_packed & 0x0F).to(torch.int8)
    low[low >= 8] -= 16
    high = ((X_packed >> 4) & 0x0F).to(torch.int8)
    high[high >= 8] -= 16
    unpacked = torch.empty(X_packed.shape[:-1] + (X_packed.shape[-1]*2,), dtype=torch.int8, device=X_packed.device)
    unpacked[..., 0::2] = low
    unpacked[..., 1::2] = high
    return unpacked


@torch.no_grad()
def quant_group(X: torch.Tensor, bits: int, group_size: int, scale_bytes_per: int = 2) -> Tuple[torch.Tensor, int]:
    X = X.float()
    group_size = min(group_size, X.shape[1])
    Xp, pad = _pad_cols(X, group_size)
    rows, cols = Xp.shape
    groups = cols // group_size
    qmax = (1 << (bits - 1)) - 1
    Xg = Xp.reshape(rows, groups, group_size)
    sc = Xg.abs().amax(dim=2, keepdim=True).clamp(min=1e-12) / qmax
    Xq = (Xg / sc).round().clamp(-qmax, qmax) * sc
    Xq = Xq.reshape(rows, cols)
    if pad:
        Xq = Xq[:, :-pad]
    data = (X.numel() * bits + 7) // 8
    scales = rows * groups * scale_bytes_per
    return Xq, data + scales


@torch.no_grad()
def sc4_quant_R(R: torch.Tensor, g: int) -> Tuple[torch.Tensor, int]:
    V, D = R.shape
    assert D % g == 0, f"D={D} must divide group_R={g}"
    Rg = R.reshape(V, D // g, g)
    sc = Rg.abs().amax(dim=2, keepdim=True).clamp(min=1e-12) / 7
    scmax = sc.amax().half().float()
    scq = ((sc / scmax * 15).round().clamp(1, 15) / 15 * scmax).half().float()
    Rq = ((Rg / scq).round().clamp(-7, 7) * scq).reshape(V, D)
    return Rq, V * D // 2 + V * (D // g) // 2 + 2


@torch.no_grad()
def int4_dq_per_row(X: torch.Tensor) -> Tuple[torch.Tensor, int]:
    s0 = X.float().abs().amax(dim=1, keepdim=True).clamp(min=1e-12) / 7
    smax = s0.amax().half().float()
    s = ((s0 / smax * 127).round().clamp(1, 127) / 127 * smax).half().float()
    return (X.float() / s).round().clamp(-7, 7) * s, X.numel() // 2 + X.shape[0] + 2


@torch.no_grad()
def int8_dq_per_row(X: torch.Tensor) -> Tuple[torch.Tensor, int]:
    s0 = X.float().abs().amax(dim=1, keepdim=True).clamp(min=1e-12) / 127
    smax = s0.amax().half().float()
    s = ((s0 / smax * 127).round().clamp(1, 127) / 127 * smax).half().float()
    return (X.float() / s).round().clamp(-127, 127) * s, X.numel() + X.shape[0] + 2


@torch.no_grad()
def sc4_quant_R_packed(R: torch.Tensor, g: int):
    V, D = R.shape
    Rg = R.reshape(V, D // g, g)
    sc = Rg.abs().amax(dim=2, keepdim=True).clamp(min=1e-12) / 7
    scmax = sc.amax().half().float()
    scq = ((sc / scmax * 15).round().clamp(1, 15) / 15 * scmax).half().float()
    R_int8 = (Rg / scq).round().clamp(-7, 7).to(torch.int8).reshape(V, D)
    scq_int = (scq / scmax * 15).round().clamp(1, 15).to(torch.int8).reshape(V, D // g)
    return pack_int4(R_int8), pack_int4(scq_int), scmax.half()

@torch.no_grad()
def quant_group_packed(X: torch.Tensor, bits: int, group_size: int):
    X = X.float()
    group_size = min(group_size, X.shape[1])
    Xp, pad = _pad_cols(X, group_size)
    rows, cols = Xp.shape
    groups = cols // group_size
    qmax = (1 << (bits - 1)) - 1
    Xg = Xp.reshape(rows, groups, group_size)
    sc = Xg.abs().amax(dim=2, keepdim=True).clamp(min=1e-12) / qmax
    X_int8 = (Xg / sc).round().clamp(-qmax, qmax).to(torch.int8).reshape(rows, cols)
    X_packed = pack_int4(X_int8) if bits == 4 else X_int8
    return X_packed, sc.reshape(rows, groups).half(), pad

@torch.no_grad()
def int4_dq_per_row_packed(X: torch.Tensor):
    s0 = X.float().abs().amax(dim=1, keepdim=True).clamp(min=1e-12) / 7
    smax = s0.amax().half().float()
    s = ((s0 / smax * 127).round().clamp(1, 127) / 127 * smax).half().float()
    X_int8 = (X.float() / s).round().clamp(-7, 7).to(torch.int8)
    s_int = (s / smax * 127).round().clamp(1, 127).to(torch.int8).squeeze(1)
    return pack_int4(X_int8), s_int, smax.half()

@torch.no_grad()
def int8_dq_per_row_packed(X: torch.Tensor):
    s0 = X.float().abs().amax(dim=1, keepdim=True).clamp(min=1e-12) / 127
    smax = s0.amax().half().float()
    s = ((s0 / smax * 127).round().clamp(1, 127) / 127 * smax).half().float()
    X_int8 = (X.float() / s).round().clamp(-127, 127).to(torch.int8)
    s_int = (s / smax * 127).round().clamp(1, 127).to(torch.int8).squeeze(1)
    return X_int8, s_int, smax.half()


@torch.no_grad()
def make_metric(h_train: torch.Tensor, power: float, ridge: float) -> Tuple[torch.Tensor, torch.Tensor]:
    h = h_train.float()
    d = h.shape[1]
    C = (h.T @ h) / h.shape[0]
    C = C + ridge * torch.eye(d, device=h.device, dtype=h.dtype) * C.diag().mean()
    e, Q = torch.linalg.eigh(C)
    e = e.clamp(min=1e-12)
    return (Q * e.pow(power).unsqueeze(0)) @ Q.T, (Q * e.pow(-power).unsqueeze(0)) @ Q.T


@torch.no_grad()
def extract_lm_cache(model, processor, kind: str, texts: List[str], *, device: str, seq_len: int, tokens: int, batch_size: int):
    hs, ys, total = [], [], 0
    for i in tqdm(range(0, len(texts), batch_size), desc="lm-cache"):
        enc = encode(processor, kind, texts[i:i + batch_size], max_length=seq_len, device=device)
        ids = enc.get("input_ids")
        if ids is None or ids.shape[1] < 2:
            continue
        out = model(**enc, use_cache=False, output_hidden_states=True, return_dict=True)
        h = out.hidden_states[-1][:, :-1, :]
        y = ids[:, 1:]
        if "attention_mask" in enc:
            mask = enc["attention_mask"][:, 1:].bool()
            h, y = h[mask], y[mask]
        else:
            h, y = h.reshape(-1, h.shape[-1]), y.reshape(-1)
        remain = tokens - total
        hs.append(h[:remain].detach().cpu().half())
        ys.append(y[:remain].detach().cpu().long())
        total += hs[-1].shape[0]
        del out, h, y, enc
        empty_cache()
        if total >= tokens:
            break
    if not hs:
        raise RuntimeError("No LM cache tokens extracted")
    return {"h": torch.cat(hs), "y": torch.cat(ys)}


class CompressedHead(nn.Module):
    def __init__(self, Wd: torch.Tensor, Aw: Optional[torch.Tensor], Bc: Optional[torch.Tensor], stats: Dict):
        super().__init__()
        self.register_buffer("Wd", Wd.contiguous().half())
        self.stats = stats
        if Aw is not None and Bc is not None:
            self.register_buffer("Aw", Aw.contiguous().half())
            self.register_buffer("Bc", Bc.contiguous().half())
        else:
            self.Aw = None
            self.Bc = None

    def forward(self, h):
        h = h.to(self.Wd.dtype)
        out = h @ self.Wd.T
        if self.Aw is not None:
            out = out + (h @ self.Bc.T) @ self.Aw.T
        return out



class ARCHeadPacked(nn.Module):
    def __init__(self, R_packed, R_scales_packed, R_scale_max, 
                 A_int8, A_scales, A_pad, 
                 B_int8, B_scales, B_pad,
                 Aw_packed, Aw_scales, Aw_smax,
                 Bc_packed, Bc_scales, Bc_pad,
                 residual_type, stats):
        super().__init__()
        self.register_buffer("R_packed", R_packed.contiguous())
        self.register_buffer("R_scales_packed", R_scales_packed.contiguous())
        self.register_buffer("R_scale_max", R_scale_max.contiguous())
        
        self.register_buffer("A_int8", A_int8.contiguous())
        self.register_buffer("A_scales", A_scales.contiguous())
        self.A_pad = A_pad
        
        self.register_buffer("B_int8", B_int8.contiguous())
        self.register_buffer("B_scales", B_scales.contiguous())
        self.B_pad = B_pad
        
        if Aw_packed is not None:
            self.register_buffer("Aw_packed", Aw_packed.contiguous())
            self.register_buffer("Aw_scales", Aw_scales.contiguous())
            self.register_buffer("Aw_smax", Aw_smax.contiguous())
            
            self.register_buffer("Bc_packed", Bc_packed.contiguous())
            self.register_buffer("Bc_scales", Bc_scales.contiguous())
            self.Bc_pad = Bc_pad
        else:
            self.Aw_packed = None
            
        self.residual_type = residual_type
        self.stats = stats

    def forward(self, h):
        h = h.to(torch.float16)
        
        if not hasattr(self, "_triton_kernel"):
            self._triton_kernel = None
            self._triton_cache = None
            
        from .triton_archead import TRITON_AVAILABLE, make_kernel_fused_dequant_row, quantize_Wd_per_row_int8
        
        if TRITON_AVAILABLE:
            if self._triton_kernel is None:
                self._triton_kernel = make_kernel_fused_dequant_row()
                
                R_int8 = unpack_int4(self.R_packed)
                R_scales_int8 = unpack_int4(self.R_scales_packed)
                scq = (R_scales_int8.float() / 15.0) * self.R_scale_max.float()
                
                V, D = R_int8.shape
                groups = scq.shape[1]
                g = D // groups
                R_float = (R_int8.reshape(V, groups, g).float() * scq.unsqueeze(-1)).reshape(V, D).half()
                
                A_rows = self.A_scales.shape[0]
                A_groups = self.A_scales.shape[1]
                A_gsize = self.A_int8.shape[-1] // A_groups
                A_g = self.A_int8.reshape(A_rows, A_groups, A_gsize)
                A_float = (A_g.float() * self.A_scales.unsqueeze(-1)).reshape(A_rows, -1)
                if self.A_pad: A_float = A_float[:, :-self.A_pad]
                A_float = A_float.half()
                
                B_rows = self.B_scales.shape[0]
                B_groups = self.B_scales.shape[1]
                B_gsize = self.B_int8.shape[-1] // B_groups
                B_g = self.B_int8.reshape(B_rows, B_groups, B_gsize)
                B_float = (B_g.float() * self.B_scales.unsqueeze(-1)).reshape(B_rows, -1)
                if self.B_pad: B_float = B_float[:, :-self.B_pad]
                B_float = B_float.half()
                
                Wd_q_t, Wd_scale = quantize_Wd_per_row_int8(R_float)
                self._triton_cache = (Wd_q_t, Wd_scale, A_float.T, B_float)

            Wd_q_t, Wd_scale, A_float, B_float = self._triton_cache
            
            original_shape = h.shape
            h_flat = h.view(-1, h.shape[-1])
            out_flat = self._triton_kernel(h_flat, Wd_q_t, Wd_scale, A_float, B_float)
            out = out_flat.view(*original_shape[:-1], -1)
        else:
            R_int8 = unpack_int4(self.R_packed)
            R_scales_int8 = unpack_int4(self.R_scales_packed)
            scq = (R_scales_int8.float() / 15.0) * self.R_scale_max.float()
            
            V, D = R_int8.shape
            groups = scq.shape[1]
            g = D // groups
            R_float = (R_int8.reshape(V, groups, g).float() * scq.unsqueeze(-1)).reshape(V, D).half()
            
            A_rows = self.A_scales.shape[0]
            A_groups = self.A_scales.shape[1]
            A_gsize = self.A_int8.shape[-1] // A_groups
            A_g = self.A_int8.reshape(A_rows, A_groups, A_gsize)
            A_float = (A_g.float() * self.A_scales.unsqueeze(-1)).reshape(A_rows, -1)
            if self.A_pad: A_float = A_float[:, :-self.A_pad]
            A_float = A_float.half()
            
            B_rows = self.B_scales.shape[0]
            B_groups = self.B_scales.shape[1]
            B_gsize = self.B_int8.shape[-1] // B_groups
            B_g = self.B_int8.reshape(B_rows, B_groups, B_gsize)
            B_float = (B_g.float() * self.B_scales.unsqueeze(-1)).reshape(B_rows, -1)
            if self.B_pad: B_float = B_float[:, :-self.B_pad]
            B_float = B_float.half()
            
            out = h @ R_float.T + (h @ B_float.T) @ A_float
            
        if self.Aw_packed is not None:
            if self.residual_type in ["archead_i8", "am_rownorm_i8"]:
                s = (self.Aw_scales.float() / 127.0) * self.Aw_smax.float()
                Aw_float = (self.Aw_packed.float() * s.unsqueeze(-1)).half()
            else:
                Aw_int8 = unpack_int4(self.Aw_packed)
                s = (self.Aw_scales.float() / 127.0) * self.Aw_smax.float()
                Aw_float = (Aw_int8.float() * s.unsqueeze(-1)).half()
                
            Bc_rows = self.Bc_scales.shape[0]
            if Aw_float.shape[-1] > Bc_rows:
                Aw_float = Aw_float[..., :Bc_rows]
                
            Bc_groups = self.Bc_scales.shape[1]
            Bc_gsize = self.Bc_packed.shape[-1] // Bc_groups
            Bc_g = self.Bc_packed.reshape(Bc_rows, Bc_groups, Bc_gsize)
            Bc_float = (Bc_g.float() * self.Bc_scales.unsqueeze(-1)).reshape(Bc_rows, -1)
            if self.Bc_pad: Bc_float = Bc_float[:, :-self.Bc_pad]
            Bc_float = Bc_float.half()
            
            out = out + (h @ Bc_float.T) @ Aw_float.T
            
        return out



@torch.no_grad()
def compress_head_baseline(W_cpu: torch.Tensor, method: str, *, device: str) -> CompressedHead:
    W = W_cpu.to(device=device, dtype=torch.float32)
    V, D = W.shape
    dense_bf16 = V * D * 2

    if method == "row_int8":
        Wd, total = quant_group(W, 8, D, scale_bytes_per=2)
        Awq = Bcq = None
    elif method == "group_int4":
        Wd, total = quant_group(W, 4, 64, scale_bytes_per=2)
        Awq = Bcq = None
    elif method == "svd8_group_int4":
        A, B = lowrank_rand(W, 8)
        Aq, a_bytes = quant_group(A.T, 8, A.shape[0], scale_bytes_per=4)
        Bq, b_bytes = quant_group(B, 8, B.shape[1], scale_bytes_per=4)
        low = Aq.T @ Bq
        Rq, r_bytes = quant_group(W - low, 4, 64, scale_bytes_per=2)
        Wd = low + Rq
        total = a_bytes + b_bytes + r_bytes
        Awq = Bcq = None
    else:
        raise ValueError(method)

    stats = {
        "name": method,
        "vocab_size": V,
        "hidden_size": D,
        "core_bytes": int(total),
        "residual_bytes": 0,
        "total_bytes": int(total),
        "byte_ratio_vs_bf16": float(total / dense_bf16),
        "compression_ratio_vs_bf16": float(dense_bf16 / total),
    }
    return CompressedHead(Wd, Awq, Bcq, stats)


@torch.no_grad()
def compress_lm_head(W_cpu: torch.Tensor, h_train_cpu: torch.Tensor, cfg: LMHeadConfig, *, device: str) -> CompressedHead:
    if getattr(cfg, "core", "") == "gptq_int4" or cfg == "gptq_int4":
        return compress_gptq(W_cpu, h_train_cpu, bits=4, group_size=128, device=device)
        
    W = W_cpu.to(device=device, dtype=torch.float32)
    H = h_train_cpu.to(device=device, dtype=torch.float32)
    V, D = W.shape

    if cfg.core.startswith("am"):
        Wsvd = W
        if cfg.core == "am_sqrt":
            Wsvd = W * (W.abs().amax(dim=1, keepdim=True).sqrt() + 1e-8)
        A, B = lowrank_rand(Wsvd, cfg.core_rank)
        Aq, a_bytes = quant_group(A.T, 5, A.shape[0] if cfg.core == "am_sqrt" else 128, scale_bytes_per=4)
        Bq, b_bytes = quant_group(B, 8, B.shape[1] if cfg.core == "am_sqrt" else 64, scale_bytes_per=4)
        low = Aq.T @ Bq
        Rq, r_bytes = quant_group(W - low, 4, cfg.quant_group, scale_bytes_per=1)
        Wd = low + Rq
        core_bytes = a_bytes + b_bytes + r_bytes
    elif cfg.core == "archead":
        row_w = W.abs().amax(dim=1, keepdim=True).pow(cfg.row_alpha) + 1e-8
        Ww = W * row_w
        if cfg.col_beta:
            Ww = Ww * (W.abs().amax(dim=0, keepdim=True).pow(cfg.col_beta) + 1e-8)
        A, B = lowrank_rand(Ww, cfg.core_rank)
        
        # Get dense version for residual computation to match exact numerics
        Rq_float, r_bytes = sc4_quant_R(W - A @ B, cfg.group_R)
        Aq_t_float, a_bytes = quant_group(A.T, 5, 64, scale_bytes_per=4)
        Bq_float, b_bytes = quant_group(B, 8, 32, scale_bytes_per=4)
        Wd = Aq_t_float.T @ Bq_float + Rq_float
        
        R_packed, R_scales_packed, R_scale_max = sc4_quant_R_packed(W - A @ B, cfg.group_R)
        A_int8, A_scales, A_pad = quant_group_packed(A.T, 5, 64)
        B_int8, B_scales, B_pad = quant_group_packed(B, 8, 32)
        core_bytes = sum(p.numel() * p.element_size() for p in [R_packed, R_scales_packed, R_scale_max, A_int8, A_scales, B_int8, B_scales])
    else:
        raise ValueError(cfg.core)

    Cs, Ci = make_metric(H, cfg.metric_power, cfg.metric_ridge)
    Awq = Bcq = None
    residual_bytes = 0
    Aw_packed = Aw_scales = Aw_smax = Bc_packed = Bc_scales = Bc_pad = None
    
    if cfg.residual != "none" and cfg.residual_rank > 0:
        Err = W - Wd
        if cfg.residual == "archead_i8":
            A, B = lowrank_rand(Err @ Cs, cfg.residual_rank)
            Aw, Bc = A, B @ Ci
            Awq, aw_bytes = int8_dq_per_row(Aw)
            Bcq, bc_bytes = quant_group(Bc, 8, Bc.shape[1], scale_bytes_per=4)
            if cfg.core == "archead":
                Aw_packed, Aw_scales, Aw_smax = int8_dq_per_row_packed(Aw)
                Bc_packed, Bc_scales, Bc_pad = quant_group_packed(Bc, 8, Bc.shape[1])
                residual_bytes = sum(p.numel() * p.element_size() for p in [Aw_packed, Aw_scales, Aw_smax, Bc_packed, Bc_scales])
        elif cfg.residual == "am_rownorm_i4":
            rn = (W @ Cs).norm(dim=1, keepdim=True) + 1e-8
            A, B = lowrank_rand((Err * rn) @ Cs, cfg.residual_rank)
            Aw, Bc = A / rn, B @ Ci
            Awq, aw_bytes = int4_dq_per_row(Aw)
            Bcq, bc_bytes = quant_group(Bc, 8, Bc.shape[1], scale_bytes_per=4)
            if cfg.core == "archead":
                Aw_packed, Aw_scales, Aw_smax = int4_dq_per_row_packed(Aw)
                Bc_packed, Bc_scales, Bc_pad = quant_group_packed(Bc, 8, Bc.shape[1])
                residual_bytes = sum(p.numel() * p.element_size() for p in [Aw_packed, Aw_scales, Aw_smax, Bc_packed, Bc_scales])
        elif cfg.residual == "am_norow_i4":
            A, B = lowrank_rand(Err @ Cs, cfg.residual_rank)
            Aw, Bc = A, B @ Ci
            Awq, aw_bytes = int4_dq_per_row(Aw)
            Bcq, bc_bytes = quant_group(Bc, 8, Bc.shape[1], scale_bytes_per=4)
            if cfg.core == "archead":
                Aw_packed, Aw_scales, Aw_smax = int4_dq_per_row_packed(Aw)
                Bc_packed, Bc_scales, Bc_pad = quant_group_packed(Bc, 8, Bc.shape[1])
                residual_bytes = sum(p.numel() * p.element_size() for p in [Aw_packed, Aw_scales, Aw_smax, Bc_packed, Bc_scales])
        else:
            raise ValueError(cfg.residual)
        if cfg.core != "archead":
            residual_bytes = aw_bytes + bc_bytes

    total = core_bytes + residual_bytes
    dense_bf16 = V * D * 2
    stats = asdict(cfg)
    stats.update(
        {
            "vocab_size": V,
            "hidden_size": D,
            "core_bytes": int(core_bytes),
            "residual_bytes": int(residual_bytes),
            "total_bytes": int(total),
            "byte_ratio_vs_bf16": float(total / dense_bf16),
            "compression_ratio_vs_bf16": float(dense_bf16 / total),
        }
    )
    if cfg.core == "archead" and getattr(cfg, "packed", True):
        return ARCHeadPacked(
            R_packed, R_scales_packed, R_scale_max,
            A_int8, A_scales, A_pad,
            B_int8, B_scales, B_pad,
            Aw_packed, Aw_scales, Aw_smax,
            Bc_packed, Bc_scales, Bc_pad,
            cfg.residual, stats
        )
    return CompressedHead(Wd, Awq, Bcq, stats)


@torch.no_grad()
def replace_model_lm_head(model, head: CompressedHead):
    # Instead of replacing weight.data (which triggers full materialization of Wd + Aw@Bc),
    # we replace the entire module to utilize CompressedHead's fused-like forward pass.
    model.set_output_embeddings(head)
    return model


@torch.no_grad()
def benchmark_archead_triton(head, *, device: str, batch: int = 128, warmup: int = 10, iters: int = 50) -> Dict:
    if isinstance(head, ARCHeadPacked):
        return {"triton_available": True, "reason": "Integrated into forward"}
    if not hasattr(head, "Aw") or head.Aw is None:
        return {"triton_available": False, "reason": "no residual factors"}
    Wd = head.Wd.to(device)
    Aw = head.Aw.to(device)
    Bc = head.Bc.to(device)
    V, D = Wd.shape
    h = torch.randn(batch, D, device=device, dtype=torch.float16)
    Wd_q_t, Wd_scale = quantize_Wd_per_row_int8(Wd)
    kernel = make_kernel_fused_dequant_row(128, 128, 64, 4, 8, 3)
    ref = lambda: h @ Wd.T + (h @ Bc.T) @ Aw.T
    ker = lambda: kernel(h, Wd_q_t, Wd_scale, Aw, Bc)
    if not torch.cuda.is_available() or not device.startswith("cuda"):
        return {"triton_available": False, "reason": "cuda unavailable"}
    ref_ms = median_ms(ref, warmup=warmup, iters=iters)
    ker_ms = median_ms(ker, warmup=warmup, iters=iters)
    return {
        "triton_available": True,
        "head_ref_ms": ref_ms,
        "head_kernel_ms": ker_ms,
        "head_latency_ratio": ref_ms / ker_ms if ker_ms > 0 else 0.0,
        "kernel_name": getattr(kernel, "kernel_name", "archead_fused"),
    }
