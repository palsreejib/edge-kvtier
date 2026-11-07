import pytest
import pandas as pd

from model.model import (prefill_count, decode_run_count, single_decode_count, ModelConfig,
                         CODEC_RATIO, fit_decode_time, single_decode_time, decode_run_time, WEIGHTS_SIZE_BYTES)
from model.calibration import add_sec_per_tok, f16_decode_samples, platform_from_filename


@pytest.fixture
def llama_3b():
    return ModelConfig(name="llama", n_layers=28, n_kv=8, d_head=128, weights_bytes=WEIGHTS_SIZE_BYTES)

@pytest.fixture
def loaded_csv_df():
    return pd.read_csv("bench/results/raw/2026-10-27_vulkan-rtx4060_depth_kvf16.csv", encoding="utf-8-sig")


###################################################################################
#   model.py tests
###################################################################################
def test_prefill_count(llama_3b):
    reuse_tok_tot, naive_tok_tot, kv_cache_mebibytes = prefill_count(ref_model=llama_3b)
    assert reuse_tok_tot == 7800
    assert naive_tok_tot == 67200
    assert kv_cache_mebibytes == pytest.approx(853, rel=0.01)   # within 1%


def test_decode_count(llama_3b):
    depth_zero = single_decode_count(llama_3b, 0, CODEC_RATIO["f16"])
    assert depth_zero == WEIGHTS_SIZE_BYTES
    assert decode_run_count(llama_3b, 0, CODEC_RATIO["f16"], 1) == depth_zero
    assert decode_run_count(llama_3b, 0, CODEC_RATIO["f16"], 2) == depth_zero + single_decode_count(llama_3b, 1, CODEC_RATIO["f16"])


def test_fit_decode_time_exact_line():
    # avg_st = 2e-6 * depth + 0.05 exactly, so the fit must recover it
    samples = [(d, 2e-6 * d + 0.05) for d in (0, 1024, 2048, 4096)]
    slope, intercept, BW_kv, BW_w_eff = fit_decode_time(samples=samples, kv_tok_bytes_f16=1000.0, weights_bytes=100)
    assert slope == pytest.approx(2e-6)
    assert intercept == pytest.approx(0.05)
    assert BW_kv == pytest.approx(1000.0 / 2e-6)
    assert BW_w_eff == pytest.approx(100 / 0.05)


def test_decode_time(llama_3b, loaded_csv_df):
    df_decode = add_sec_per_tok(loaded_csv_df[loaded_csv_df["n_gen"] != 0])
    samples = f16_decode_samples(df_decode)
    _, _, BW_kv, BW_w_eff = fit_decode_time(samples=samples, kv_tok_bytes_f16=llama_3b.kv_tok_bytes_f16, weights_bytes=llama_3b.weights_bytes)
    depth_zero_time = single_decode_time(depth_toks=0, codec_ratio=CODEC_RATIO["f16"], kv_tok_bytes_f16=llama_3b.kv_tok_bytes_f16,
                              weights_bytes=llama_3b.weights_bytes, BW_kv=BW_kv, BW_w_eff=BW_w_eff)

    depth_zero_df = df_decode[df_decode["n_depth"] == 0]
    assert depth_zero_time == pytest.approx(depth_zero_df["avg_st"].iloc[0], rel=0.05)

    depth_run_zero_time = decode_run_time(depth_start_toks=0, codec_ratio=CODEC_RATIO["f16"], n_run_toks=1, kv_tok_bytes_f16=llama_3b.kv_tok_bytes_f16,
                                          weights_bytes=llama_3b.weights_bytes, BW_kv=BW_kv, BW_w_eff=BW_w_eff)
    assert depth_run_zero_time == pytest.approx(depth_zero_time, rel=0.05)

    num_toks_gen = depth_zero_df["n_gen"].iloc[0]
    depth_full_run_time = decode_run_time(depth_start_toks=0, codec_ratio=CODEC_RATIO["f16"], n_run_toks=num_toks_gen, kv_tok_bytes_f16=llama_3b.kv_tok_bytes_f16,
                                              weights_bytes=llama_3b.weights_bytes, BW_kv=BW_kv, BW_w_eff=BW_w_eff)
    assert depth_full_run_time == pytest.approx((depth_zero_df["avg_ns"].iloc[0])/(1e9), rel=0.05)


###################################################################################
#   calibration.py tests
###################################################################################
def test_f16_decode_samples_filters_rows():
    df = pd.DataFrame({
        "n_depth":[0,     2048,   0,      0],
        "type_k": ["f16", "f16",  "q8_0", "f16"],
        "type_v": ["f16", "f16",  "q8_0", "q4_0"],
        "avg_st": [0.05,  0.10,   0.04,   0.045],
    })
    assert f16_decode_samples(df) == [(0, 0.05), (2048, 0.10)]


def test_platform_from_filename():
    assert platform_from_filename("bench/results/raw/2026-10-27_cpu_depth_kvq8.csv") == "cpu"
    assert platform_from_filename("2026-10-27_vulkan-igpu_depth_kvf16.csv") == "igpu"
    assert platform_from_filename("2026-10-27_vulkan-rtx4060_depth_kvq4.csv") == "4060"
    with pytest.raises(ValueError):
        platform_from_filename("2026-10-27_unknown.csv")