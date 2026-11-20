# Abbreviations explained:  agn = agent | toks = tokens | tot = total | d_ = dimension | n_<name> = number of <name>

import math
from dataclasses import dataclass
import numpy as np

try:
    from model.constants import energy_const, e_per_flop, p_static, e_sram_per_byte
except ModuleNotFoundError:   # run as a plain script: model/ is on sys.path, not the repo root
    from model.constants import energy_const, e_per_flop, p_static, e_sram_per_byte


# general constants
WEIGHTS_SIZE_BYTES = 2011539712
DEFAULT_PREFIX_TOKS = 3000
DEFAULT_AGN_STEP_TOKS = 400
NUM_AGN_STEPS = 12
MEBI = 2**20
N_PARAMS = 3.0e9            # parameter count used for weight FLOPs (tracker value; Llama-3.2-3B is ~3.21e9)
N_HEADS = 24                # query heads per layer for Llama-3.2-3B
DEFAULT_N_UBATCH = 512      # llama.cpp default micro-batch: prefill runs the model once per chunk of this many tokens


# KV size relative to f16, from llama.cpp block formats (32 values per block):
# f16 = 64 B, q8_0 = 32 B + 2 B scale = 34 B, q4_0 = 16 B + 2 B scale = 18 B.
CODEC_RATIO = {
    "f16": 1.0,
    "q8_0": 34 / 64,
    "q4_0": 18 / 64
}


@dataclass(frozen=True)
class ModelConfig:
    """Static description of a model. Immutable after creation."""
    name: str
    n_layers: int               # number of layers in the model
    n_kv: int                   # number of KV heads
    d_head: int                 # dimensions per head
    weights_bytes: int          # total bytes of weights (file size)
    n_heads: int = N_HEADS      # number of query heads
    n_params: float = N_PARAMS  # parameter count (for weight FLOPs = 2 * n_params per token)

    @property
    def kv_tok_bytes_f16(self) -> float:
        """KV bytes stored per token (K and V, all layers) at f16 precision."""
        # first 2 for K and V. Second 2 for 2 bytes in f16 precision
        return 2 * self.n_layers * self.n_kv * self.d_head * 2

    @property
    def attn_flops_coeff(self) -> float:
        """Attention FLOPs per (query token x key token): QK^T and AV are 2*d_head each,
        times query heads and layers."""
        return 4 * self.d_head * self.n_heads * self.n_layers


LLAMA_3B = ModelConfig(name="llama", n_layers=28, n_kv=8, d_head=128, weights_bytes=WEIGHTS_SIZE_BYTES)


###################################################################################
#   COUNT MODEL
###################################################################################
def prefill_count(ref_model: ModelConfig,
                  prefix_toks: int = DEFAULT_PREFIX_TOKS,
                  agn_step_toks: int = DEFAULT_AGN_STEP_TOKS,
                  n_agn_steps: int = NUM_AGN_STEPS,
                  codec_ratio: float = 1.0) -> tuple[int, int, float]:
    """Prefill tokens and KV cache size for a fixed-size agent loop.
    Returns (reuse_toks, naive_toks, kv_cache_mebibytes)."""
    naive_tok_tot = 0
    for i in range(1, n_agn_steps + 1):
        naive_tok_tot += prefix_toks + agn_step_toks * i

    reuse_tok_tot = prefix_toks + agn_step_toks * n_agn_steps
    kv_cache_mebibytes = reuse_tok_tot * ref_model.kv_tok_bytes_f16 * codec_ratio / MEBI
    return reuse_tok_tot, naive_tok_tot, kv_cache_mebibytes


def single_decode_count(ref_model: ModelConfig, depth_toks: int, codec_ratio: float = 1.0) -> float:
    """Bytes read to generate one token."""
    kv_bytes_read = depth_toks * ref_model.kv_tok_bytes_f16 * codec_ratio
    return ref_model.weights_bytes + kv_bytes_read


def decode_run_count(ref_model: ModelConfig, depth_start_toks: int,
                     codec_ratio: float, n_run_toks: int) -> float:
    """Total bytes read to generate n_run_toks tokens, starting at depth_start_toks.
    Depth grows by one per generated token."""
    total_bytes = 0.0
    # L is the number of KV entries read when generating that token
    for L in range(depth_start_toks, depth_start_toks + n_run_toks):
        total_bytes += single_decode_count(ref_model=ref_model, depth_toks=L, codec_ratio=codec_ratio)
    return total_bytes


def prefill_dram_count(ref_model: ModelConfig, n_new_toks: int, depth_toks: int,
                       codec_ratio: float = 1.0, n_ubatch: int = DEFAULT_N_UBATCH) -> float:
    """DRAM bytes moved to prefill n_new_toks tokens on top of depth_toks cached tokens.
    llama.cpp runs prefill in chunks of n_ubatch tokens. Per chunk:
      - all weights are streamed once (they don't fit in cache),
      - attention reads the KV of every token up to the end of the chunk.
    Plus the KV of the new tokens is written once.
    Activations are ignored (small vs weights). KV read is a lower bound: flash attention
    can re-read KV tiles more than once per chunk."""
    if n_new_toks <= 0:
        return 0.0
    kv_tok = ref_model.kv_tok_bytes_f16 * codec_ratio
    n_chunks = math.ceil(n_new_toks / n_ubatch)

    weight_bytes = n_chunks * ref_model.weights_bytes
    kv_write_bytes = n_new_toks * kv_tok
    kv_read_bytes = 0.0
    for c in range(n_chunks):
        chunk_end = min((c + 1) * n_ubatch, n_new_toks)      # tokens of this prefill done after this chunk
        kv_read_bytes += (depth_toks + chunk_end) * kv_tok
    return weight_bytes + kv_write_bytes + kv_read_bytes


###################################################################################
#   FLOP MODEL  (industry convention: one multiply-add = 2 FLOPs)
###################################################################################
def prefill_flops(ref_model: ModelConfig, n_new_toks: int, depth_toks: int) -> float:
    """FLOPs to prefill n_new_toks on top of depth_toks cached tokens.
    Weights: 2 * P per token. Attention (causal): token i sees depth + i keys, so the total is
    attn_coeff * N * (d + N/2). Chunking into ubatches does not change this total, because the
    keys each token sees are the same however the prompt is split."""
    N, d = n_new_toks, depth_toks
    weight_flops = 2 * ref_model.n_params * N
    attn_flops = ref_model.attn_flops_coeff * N * (d + N / 2)
    return weight_flops + attn_flops


def single_decode_flops(ref_model: ModelConfig, depth_toks: int) -> float:
    """FLOPs to generate one token that attends to depth_toks KV entries."""
    return 2 * ref_model.n_params + ref_model.attn_flops_coeff * depth_toks


def decode_run_flops(ref_model: ModelConfig, depth_start_toks: int, n_run_toks: int) -> float:
    """FLOPs for n_run_toks decode tokens starting at depth_start_toks (same depths as decode_run_count).
    Exact sum: depths d0 .. d0+n-1 add up to n*d0 + n(n-1)/2."""
    n, d0 = n_run_toks, depth_start_toks
    depth_sum = n * d0 + n * (n - 1) / 2
    return 2 * ref_model.n_params * n + ref_model.attn_flops_coeff * depth_sum


def transformer_flops(ref_model: ModelConfig, n_new_toks: int, n_cached_toks: int) -> float:
    """FLOPs for decode and prefill, assumes flash attention doesn't compute any extra values in its tiles"""
    N, C = n_new_toks, n_cached_toks
    weight_flops = 2 * ref_model.n_params * N
    attn_flops = ref_model.attn_flops_coeff * N * (N + 2*C + 1)
    return weight_flops + attn_flops


###################################################################################
#   TIME MODEL
###################################################################################
def fit_decode_time(samples: list[tuple[int, float]], kv_tok_bytes_f16: float,
                    weights_bytes: int) -> tuple[float, float, float, float]:
    """Line fit of seconds-per-token against depth. samples is a list of (depth, avg_st)
    pairs from ONE platform, f16 KV only.
    Returns (slope, intercept (weights_bytes/w_BW + t_overhead), BW_kv, BW_w_eff)."""
    depths, avg_sts = zip(*samples)
    slope, intercept = np.polyfit(depths, avg_sts, 1)
    BW_kv = kv_tok_bytes_f16/slope
    BW_w_eff = weights_bytes/intercept
    return slope, intercept, BW_kv, BW_w_eff


def single_decode_time(depth_toks: int, codec_ratio: float, kv_tok_bytes_f16: float,
                       weights_bytes: int, BW_kv: float, BW_w_eff: float) -> float:
    # Using codec ratio since line fitting based on f16
    return (depth_toks*codec_ratio*kv_tok_bytes_f16/BW_kv) + (weights_bytes/BW_w_eff)


def decode_run_time(depth_start_toks: int, codec_ratio: float, n_run_toks: int, kv_tok_bytes_f16: float,
                    weights_bytes: float, BW_kv: float, BW_w_eff: float) -> float:
    avg_depth = depth_start_toks + ((n_run_toks - 1) / 2)
    return n_run_toks * single_decode_time(depth_toks=avg_depth, codec_ratio=codec_ratio, kv_tok_bytes_f16=kv_tok_bytes_f16,
                                           weights_bytes=weights_bytes, BW_kv=BW_kv, BW_w_eff=BW_w_eff)


def prefill_time(ref_model: ModelConfig, n_new_toks: int, depth_toks: int, rate0_tok_s: float,
                 n_ref_toks: int = 512, BW_w_eff: float | None = None,
                 codec_ratio: float = 1.0, n_ubatch: int = DEFAULT_N_UBATCH) -> float:
    """Seconds to prefill n_new_toks on top of depth_toks cached tokens.

    Compute side: rate0_tok_s is a measured prefill rate (llama-bench pp<n_ref_toks> at depth 0).
    It fixes the platform's effective compute C_eff = rate0 * FLOPs(n_ref_toks, 0) / n_ref_toks,
    and t_compute = FLOPs(N, d) / C_eff. This equals the plan's rate(d) = rate(0) * F_w / (F_w + F_attn(d)).

    Memory side (optional): if BW_w_eff is given, t_memory = prefill DRAM bytes / BW_w_eff, and the
    result is the roofline max of the two. With n_ubatch=512 prefill is compute-bound on all three
    platforms; the memory side only matters for small ubatches.
    rate0 should come from a run with the same n_ubatch."""
    if n_new_toks <= 0:
        return 0.0
    C_eff = rate0_tok_s * prefill_flops(ref_model, n_ref_toks, 0) / n_ref_toks
    t_compute = prefill_flops(ref_model, n_new_toks, depth_toks) / C_eff
    if BW_w_eff is None:
        return t_compute
    t_memory = prefill_dram_count(ref_model, n_new_toks, depth_toks, codec_ratio, n_ubatch) / BW_w_eff
    return max(t_compute, t_memory)


###################################################################################
#   ENERGY MODEL
#   E = DRAM bytes * e_DRAM + SRAM bytes * e_SRAM + SSD bytes * e_SSD + FLOPs * e_op + P_static * t
#   All constants are assumptions from Horowitz ISSCC'14 (see constants.py). Results are in joules.
###################################################################################
def energy_breakdown(dram_bytes: float, flops: float, time_s: float, platform: str,
                     scenario: str = "e_nominal", sram_bytes: float = 0.0,
                     sram_size_bytes: float | None = None, ssd_bytes: float = 0.0) -> dict[str, float]:
    """Energy split into its parts. Returns {dram, sram, ssd, compute, static, total} in J.
    sram_size_bytes (the scratchpad size) is required when sram_bytes > 0."""
    if sram_bytes > 0 and sram_size_bytes is None:
        raise ValueError("sram_size_bytes is required when sram_bytes > 0")
    parts = {
        "dram": dram_bytes * energy_const("e_dram", scenario),
        "sram": sram_bytes * e_sram_per_byte(sram_size_bytes, scenario) if sram_bytes > 0 else 0.0,
        "ssd": ssd_bytes * energy_const("e_ssd", scenario),
        "compute": flops * e_per_flop(platform, scenario),
        "static": p_static(platform, scenario) * time_s,
    }
    parts["total"] = sum(parts.values())
    return parts


def decode_run_energy(ref_model: ModelConfig, depth_start_toks: int, codec_ratio: float, n_run_toks: int,
                      time_s: float, platform: str, scenario: str = "e_nominal") -> dict[str, float]:
    """Energy for a decode run. time_s comes from decode_run_time (or a measurement).
    No dequant op cost yet: that is the M2 term, fitted in Week 2."""
    dram_bytes = decode_run_count(ref_model, depth_start_toks, codec_ratio, n_run_toks)
    flops = decode_run_flops(ref_model, depth_start_toks, n_run_toks)
    return energy_breakdown(dram_bytes, flops, time_s, platform, scenario)


def prefill_energy(ref_model: ModelConfig, n_new_toks: int, depth_toks: int, time_s: float, platform: str,
                   codec_ratio: float = 1.0, n_ubatch: int = DEFAULT_N_UBATCH,
                   scenario: str = "e_nominal") -> dict[str, float]:
    """Energy for one prefill. time_s comes from prefill_time (or a measurement)."""
    dram_bytes = prefill_dram_count(ref_model, n_new_toks, depth_toks, codec_ratio, n_ubatch)
    flops = prefill_flops(ref_model, n_new_toks, depth_toks)
    return energy_breakdown(dram_bytes, flops, time_s, platform, scenario)