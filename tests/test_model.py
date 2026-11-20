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
        
        
###################################################################################
#   energy / prefill tests  (constants.py + new model.py functions)
###################################################################################
from model.constants import (E_CONST, SCENARIOS, energy_const, e_per_flop, p_static,
                             e_sram_per_byte, SRAM_KEYS, KIB, MIB)
from model.model import (LLAMA_3B, DEFAULT_N_UBATCH, prefill_dram_count, prefill_flops, single_decode_flops,
                         decode_run_flops, prefill_time, energy_breakdown, decode_run_energy, prefill_energy)

PLATFORMS = ("cpu", "igpu", "4060")


def test_constants_schema():
    required = {"unit", "raw", "low", "nom", "high", "source", "label", "group"}
    labels = {"published", "scaled from published", "pure assumption"}
    groups = {g for cols in SCENARIOS.values() for g in cols}
    for name, c in E_CONST.items():
        assert required <= c.keys(), name
        assert c["label"] in labels, name
        assert c["group"] in groups, name
        assert 0 < c["low"] <= c["nom"] <= c["high"], name
        assert c["raw"] > 0, name


def test_energy_const_scenarios():
    c = E_CONST["e_dram"]
    assert energy_const("e_dram", "e_nominal") == pytest.approx(c["raw"] * c["nom"])
    assert energy_const("e_dram", "e_pessimistic") == pytest.approx(c["raw"] * c["low"])
    assert energy_const("e_dram", "e_optimistic") == pytest.approx(c["raw"] * c["high"])
    # pessimistic for the tier = cheap DRAM, expensive ops
    assert energy_const("e_dram", "e_pessimistic") < energy_const("e_dram", "e_optimistic")
    assert e_per_flop("4060", "e_pessimistic") > e_per_flop("4060", "e_optimistic")


def test_e_per_flop_uses_platform_op():
    s = "e_nominal"
    assert e_per_flop("cpu", s) == pytest.approx((energy_const("int8_mul", s) + energy_const("int8_add", s)) / 2)
    assert e_per_flop("4060", s) == pytest.approx((energy_const("f16_mul", s) + energy_const("f16_add", s)) / 2)


def test_sram_interpolation():
    for k in SRAM_KEYS:   # hits the table points exactly
        assert e_sram_per_byte(E_CONST[k]["size_bytes"]) == pytest.approx(energy_const(k))
    mid = e_sram_per_byte(256 * KIB)
    assert energy_const("sram_32KiB") < mid < energy_const("sram_1MiB")
    assert e_sram_per_byte(16 * MIB) > energy_const("sram_1MiB")   # extrapolates upward


def test_prefill_dram_count_ubatch():
    m, W, K = LLAMA_3B, LLAMA_3B.weights_bytes, LLAMA_3B.kv_tok_bytes_f16
    # one full chunk at depth 0: weights once, write 512 KV, read 512 KV
    assert prefill_dram_count(m, 512, 0) == pytest.approx(W + 512 * K + 512 * K)
    # one token over a chunk -> weights streamed twice
    assert prefill_dram_count(m, 513, 0) == pytest.approx(2 * W + 513 * K + (512 + 513) * K)
    # 3000-token prefix: ceil(3000/512) = 6 weight passes
    assert prefill_dram_count(m, 3000, 0) > 6 * W
    assert prefill_dram_count(m, 3000, 0) < 7 * W
    # smaller ubatch -> more weight traffic; depth adds KV reads; codec shrinks KV terms
    assert prefill_dram_count(m, 3000, 0, n_ubatch=256) > prefill_dram_count(m, 3000, 0)
    assert prefill_dram_count(m, 512, 4096) == pytest.approx(W + 512 * K + (4096 + 512) * K)
    assert prefill_dram_count(m, 512, 0, codec_ratio=0.5) == pytest.approx(W + 512 * K)
    assert prefill_dram_count(m, 0, 4096) == 0.0


def test_flop_counts():
    m = LLAMA_3B
    a = 4 * 128 * 24 * 28
    assert m.attn_flops_coeff == a
    assert prefill_flops(m, 512, 0) == pytest.approx(2 * m.n_params * 512 + a * 512 * 256)
    assert single_decode_flops(m, 1000) == pytest.approx(2 * m.n_params + a * 1000)
    # closed-form decode run equals the per-token sum
    brute = sum(single_decode_flops(m, L) for L in range(100, 100 + 37))
    assert decode_run_flops(m, 100, 37) == pytest.approx(brute)
    # prefill split in two equals one prefill (chunking does not change FLOPs)
    assert prefill_flops(m, 1000, 0) == pytest.approx(prefill_flops(m, 400, 0) + prefill_flops(m, 600, 400))


def test_prefill_time():
    m = LLAMA_3B
    # at the calibration point the model returns the measured rate exactly
    assert prefill_time(m, 512, 0, rate0_tok_s=352.8) == pytest.approx(512 / 352.8)
    # iGPU: Nov 4 review says the FLOP formula predicts ~242.9 tok/s at 8k (measured 76.9)
    rate_8k = 512 / prefill_time(m, 512, 8192, rate0_tok_s=352.8)
    assert rate_8k == pytest.approx(242.9, rel=0.02)
    # roofline: a tiny bandwidth makes it memory-bound
    t_mem = prefill_time(m, 512, 0, rate0_tok_s=352.8, BW_w_eff=1e9)
    assert t_mem == pytest.approx(prefill_dram_count(m, 512, 0) / 1e9)
    # realistic bandwidth: still compute-bound
    assert prefill_time(m, 512, 0, rate0_tok_s=352.8, BW_w_eff=44e9) == pytest.approx(512 / 352.8)
    assert prefill_time(m, 0, 100, rate0_tok_s=352.8) == 0.0


def test_energy_units():
    e = energy_breakdown(dram_bytes=1e9, flops=0, time_s=0, platform="cpu")
    assert e["dram"] == pytest.approx(1e9 * energy_const("e_dram"))
    assert e["total"] == pytest.approx(e["dram"])
    e = energy_breakdown(dram_bytes=0, flops=1e12, time_s=0, platform="4060")
    assert e["compute"] == pytest.approx(1e12 * e_per_flop("4060"))


def test_energy_static_only():
    for p in PLATFORMS:
        e = energy_breakdown(dram_bytes=0, flops=0, time_s=2.0, platform=p)
        assert e["total"] == pytest.approx(2.0 * p_static(p))
        assert e["dram"] == e["compute"] == e["sram"] == e["ssd"] == 0


def test_energy_total_is_sum():
    e = energy_breakdown(1e9, 1e12, 0.5, "igpu", sram_bytes=1e6, sram_size_bytes=MIB, ssd_bytes=1e6)
    assert e["total"] == pytest.approx(sum(v for k, v in e.items() if k != "total"))
    with pytest.raises(ValueError):
        energy_breakdown(1e9, 0, 0, "cpu", sram_bytes=1e6)   # missing scratchpad size


@pytest.mark.parametrize("scenario", list(SCENARIOS))
@pytest.mark.parametrize("platform, tok_s", [("cpu", 24.2), ("igpu", 22.0), ("4060", 105.0)])
def test_implied_power_decode(platform, tok_s, scenario):
    # depth-0 decode rates from the Nov 4 benchmark review; power must be laptop-sized
    t = 1 / tok_s
    e = decode_run_energy(LLAMA_3B, 0, 1.0, 1, time_s=t, platform=platform, scenario=scenario)
    assert 1 < e["total"] / t < 150


@pytest.mark.parametrize("scenario", list(SCENARIOS))
def test_implied_power_prefill(scenario):
    t = prefill_time(LLAMA_3B, 512, 0, rate0_tok_s=352.8)
    e = prefill_energy(LLAMA_3B, 512, 0, time_s=t, platform="igpu", scenario=scenario)
    assert 1 < e["total"] / t < 150