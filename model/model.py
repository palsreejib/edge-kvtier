# Abbreviations explained:  agn = agent | toks = tokens | tot = total | d_ = dimension | n_<name> = number of <name>

from dataclasses import dataclass


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
    bytes_per_kv_elem: float    # bytes per KV element (f16 = 2)

    @property
    def kv_tok_bytes(self) -> float:
        """KV bytes stored per token (K and V, all layers) at f16 precision."""
        return 2 * self.n_layers * self.n_kv * self.d_head * self.bytes_per_kv_elem



def prefill_count(ref_model: ModelConfig,
                  prefix_toks: int = DEFAULT_PREFIX_TOKS,
                  agn_step_toks: int = DEFAULT_AGN_STEP_TOKS,
                  n_agn_steps: int = NUM_AGN_STEPS) -> tuple[int, int, float]:
    """Prefill tokens and KV cache size for a fixed-size agent loop.
    Returns (reuse_toks, naive_toks, kv_cache_mebibytes)."""
    naive_tok_tot = 0
    for i in range(1, n_agn_steps + 1):
        naive_tok_tot += prefix_toks + agn_step_toks * i
 
    reuse_tok_tot = prefix_toks + agn_step_toks * n_agn_steps
    kv_cache_mebibytes = reuse_tok_tot * ref_model.kv_tok_bytes / MEBI
    return reuse_tok_tot, naive_tok_tot, kv_cache_mebibytes


def single_decode_count(depth_toks: int, ref_model: ModelConfig, codec_ratio: float = 1.0) -> float:
    """Bytes read to generate one token."""
    kv_bytes_read = depth_toks * ref_model.kv_tok_bytes * codec_ratio
    return ref_model.weights_bytes + kv_bytes_read


def decode_run_count(depth_start_toks: int, ref_model: ModelConfig,
                     codec_ratio: float, n_run_toks: int) -> float:
    """Total bytes read to generate n_run_toks tokens, starting at depth_start_toks.
    Depth grows by one per generated token."""
    total_bytes = 0.0
    # L is the number of KV entries read when generating that token
    for L in range(depth_start_toks, depth_start_toks + n_run_toks):
        total_bytes += single_decode_count(depth_toks=L, ref_model=ref_model, codec_ratio=codec_ratio)
    return total_bytes


if __name__ == "__main__":
    print("model.py main")