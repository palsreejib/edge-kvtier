# Abbreviations explained:  agn = agent | toks = tokens | tot = total | d_ = dimension |n_<name> = number of <name> | <name>_<unit> = unit is unit for name

from dataclasses import dataclass, field

DEFAULT_PREFIX_TOKS = 3000
DEFAULT_AGN_STEP_TOKS = 400
NUM_AGN_STEPS = 12
MEBI = 2**20
MEGA = 1e6

@dataclass(frozen=True)
class AI_Model:
    name: str
    n_layers: float
    n_kv: float
    d_head: float
    kv_bits: float
    kv_bytes: float = field(init=False)
    kv_tok_bytes: float = field(init=False)
    
    # runs automatically after the object is created
    def __post_init__(self):
        object.__setattr__(self, "kv_bytes", self.kv_bits / 8)
        object.__setattr__(self, "kv_tok_bytes", 2 * self.n_layers * self.n_kv * self.d_head * self.kv_bytes)
    
    

def prefill(ref_model: AI_Model,
            prefix_toks: float = DEFAULT_PREFIX_TOKS,
            agn_step_toks: float = DEFAULT_AGN_STEP_TOKS,
            n_agn_step: float = NUM_AGN_STEPS
            ):
    naive_tok_tot = 0
    
    for i in range(1,n_agn_step+1):
        naive_tok_tot = naive_tok_tot + (prefix_toks + agn_step_toks*i)
        
    reuse_tok_tot = prefix_toks + (agn_step_toks*n_agn_step)

    kv_tot_bytes = reuse_tok_tot * ref_model.kv_tok_bytes
    
    print(f"Number of reuse tokens = {reuse_tok_tot}")
    print(f"Number of naive tokens = {naive_tok_tot}")
    print(f"Size of KV token = {ref_model.kv_tok_bytes} B")
    print(f"Size of KV cache needed = {kv_tot_bytes / MEBI} MiB")


if __name__ == "__main__":
    llama = AI_Model(name="llama", n_layers=28, n_kv=8, d_head=128, kv_bits=16)
    prefill(ref_model=llama)