import pytest

from model.model import prefill_count, decode_run_count, single_decode_count, ModelConfig, CODEC_RATIO


WEIGHTS_SIZE_BYTES = 2011539712


@pytest.fixture
def llama_3b():
    return ModelConfig(name="llama", n_layers=28, n_kv=8, d_head=128, weights_bytes=WEIGHTS_SIZE_BYTES, bytes_per_kv_elem=2)


def test_prefill_count(llama_3b):
    reuse_tok_tot, naive_tok_tot, kv_cache_mebibytes = prefill_count(ref_model=llama_3b)
    assert reuse_tok_tot == 7800
    assert naive_tok_tot == 67200
    assert kv_cache_mebibytes == pytest.approx(853, rel=0.01)   # within 1%
    

def test_decode_count(llama_3b):
    depth_zero = single_decode_count(0, llama_3b, CODEC_RATIO["f16"])
    assert depth_zero == WEIGHTS_SIZE_BYTES
    assert decode_run_count(0, llama_3b, CODEC_RATIO["f16"], 1) == depth_zero
    assert decode_run_count(0, llama_3b, CODEC_RATIO["f16"], 2) == depth_zero + single_decode_count(1, llama_3b, CODEC_RATIO["f16"])
    
    