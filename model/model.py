# Abbreviations explained:  agn = agent | toks = tokens | tot = total | d_ = dimension | n_<name> = number of <name>

from dataclasses import dataclass
import numpy as np


# general constants
WEIGHTS_SIZE_BYTES = 2011539712
DEFAULT_PREFIX_TOKS = 3000
DEFAULT_AGN_STEP_TOKS = 400
NUM_AGN_STEPS = 12
MEBI = 2**20


# KV size relative to f16, from llama.cpp block formats (32 values per block):
# f16 = 64 B, q8_0 = 32 B + 2 B scale = 34 B, q4_0 = 16 B + 2 B scale = 18 B.
CODEC_RATIO = {
    "f16": 1.0,
    "q8_0": 34 / 64,
    "q4_0": 18 / 64,
}


@dataclass(frozen=True)
class ModelConfig:
    """Static description of a model. Immutable after creation."""
    name: str
    n_layers: int               # number of layers in the model
    n_kv: int                   # number of KV heads
    d_head: int                 # dimensions per head
    weights_bytes: int          # total bytes of weights (file size)
    #kv_format: string          Need to determine if I make subclass or something else to handle all the codec_ratio's in my code

    @property
    def kv_tok_bytes_f16(self) -> float:
        """KV bytes stored per token (K and V, all layers) at f16 precision."""
        # first 2 for K and V. Second 2 for 2 bytes in f16 precision
        return 2 * self.n_layers * self.n_kv * self.d_head * 2


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
