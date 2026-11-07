"""Calibration: read llama-bench CSVs, fit the time model per platform, and write the
decode error table. Everything that knows about CSV files and file names lives here.

Run from the repo root with either:
    python -m model.calibration
    python model/calibration.py
"""

import sys
from pathlib import Path
import pandas as pd

try:
    from model.model import LLAMA_3B, CODEC_RATIO, fit_decode_time, decode_run_time
except ModuleNotFoundError:   # run as a plain script: model/ is on sys.path, not the repo root
    from model import LLAMA_3B, CODEC_RATIO, fit_decode_time, decode_run_time


REPO_ROOT = Path(__file__).resolve().parent.parent   # adjust to your top-level directory
RAW_DIR = REPO_ROOT / "bench" / "results" / "raw"
RESULTS_DIR = REPO_ROOT / "results"

PLATFORMS = ("cpu", "igpu", "4060")   # substrings that appear in your CSV file names
DEDUPE_COLS = ["type_k", "type_v", "n_depth", "n_gen", "n_prompt", "n_threads"]

# decimal places per column in the written error table
CSV_DECIMALS = {
    "measured_tok_s": 3, "predicted_tok_s": 3,
    "signed_error": 4, "signed_error_pct": 4, "abs_error": 4,
    "BW_kv_GBps": 3, "BW_w_eff_GBps": 3,
}

# key columns: model_type, model_size, model_n_params, n_threads, type_k, type_v, n_prompt, n_gen, n_depth,
#              devices, avg_ns, avg_ts, stddev_ts


def add_sec_per_tok(df: pd.DataFrame) -> pd.DataFrame:
    df_new = df.copy()
    # Calculate seconds per token (st)
    df_new["avg_st"] = 1 / df_new["avg_ts"]
    return df_new


def platform_from_filename(csv_path: str) -> str:
    """Pick the platform from the benchmark file name. Raises if it's ambiguous."""
    stem = Path(csv_path).stem.lower()
    matches = [p for p in PLATFORMS if p in stem]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one platform in '{Path(csv_path).name}', found {matches}")
    return matches[0]


def f16_decode_samples(df: pd.DataFrame) -> list[tuple[int, float]]:
    """(depth, avg_st) pairs from the f16 rows of one platform's decode rows.
    df must already be decode-only with avg_st added. q8/q4 stay held-out."""
    df_fit = df[(df["type_k"] == "f16") & (df["type_v"] == "f16")]
    return list(zip(df_fit["n_depth"], df_fit["avg_st"]))


def single_run_decode_error(df: pd.DataFrame, platform: str) -> pd.DataFrame:
    """Error table for ONE platform. df holds all of that platform's benchmark rows
    (f16 plus q8/q4). The fit uses f16 rows only; every row is then predicted."""
    llama = LLAMA_3B
    df_decode = add_sec_per_tok(df[(df["n_prompt"] == 0) & (df["n_gen"] != 0)])   # pure decode rows only

    _, _, BW_kv, BW_w_eff = fit_decode_time(samples=f16_decode_samples(df_decode),
                                            kv_tok_bytes_f16=llama.kv_tok_bytes_f16,
                                            weights_bytes=llama.weights_bytes)

    rows = []
    for row in df_decode.itertuples(index=False):
        k_format, v_format = row.type_k, row.type_v
        if k_format != v_format:
            codec_ratio = (CODEC_RATIO[k_format] + CODEC_RATIO[v_format]) / 2
            codec_label = f"{k_format}/{v_format}"
        else:
            codec_ratio = CODEC_RATIO[k_format]
            codec_label = k_format

        predicted_total_time = decode_run_time(depth_start_toks=row.n_depth, codec_ratio=codec_ratio,
                                               n_run_toks=row.n_gen, kv_tok_bytes_f16=llama.kv_tok_bytes_f16,
                                               weights_bytes=llama.weights_bytes, BW_kv=BW_kv, BW_w_eff=BW_w_eff)
        predicted_tok_p_sec = row.n_gen / predicted_total_time
        signed_error = (predicted_tok_p_sec - row.avg_ts) / row.avg_ts   # > 0: model too fast

        rows.append({
            "platform": platform,
            "codec": codec_label,
            "depth": row.n_depth,
            "n_gen": row.n_gen,
            "measured_tok_s": row.avg_ts,
            "predicted_tok_s": predicted_tok_p_sec,
            "signed_error": signed_error,
            "signed_error_pct": signed_error * 100,
            "abs_error": abs(signed_error),
            "fit_or_heldout": "fit" if codec_label == "f16" else "held-out",
            "BW_kv_GBps": BW_kv / 1e9,           # from this platform's f16 fit
            "BW_w_eff_GBps": BW_w_eff / 1e9,
        })
    return pd.DataFrame(rows)


def build_decode_error_table(csv_paths: list[str],
                             out_name: str = "decode_error_table.csv") -> pd.DataFrame:
    """Read every listed CSV, label each row with its platform, then run the
    per-platform baseline fit and prediction. Writes results/<out_name>."""
    frames = []
    for path in csv_paths:
        df = pd.read_csv(path, encoding="utf-8-sig")
        df["platform"] = platform_from_filename(path)
        df["source_file"] = Path(path).name
        frames.append(df)
    all_data = pd.concat(frames, ignore_index=True)

    tables = []
    for platform, df_platform in all_data.groupby("platform", sort=False):
        keys = [c for c in DEDUPE_COLS if c in df_platform.columns]
        # paths are sorted, so a *_rerun file comes after its original and replaces it
        df_platform = df_platform.drop_duplicates(subset=keys, keep="last")
        tables.append(single_run_decode_error(df_platform, platform))

    result = pd.concat(tables, ignore_index=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    result.round(CSV_DECIMALS).to_csv(RESULTS_DIR / out_name, index=False)
    return result


if __name__ == "__main__":
    # Paths from the command line, else every depth sweep (f16/q8/q4 per platform).
    # Sorted so a *_rerun file comes after its original and wins the dedupe.
    csv_files = sorted(sys.argv[1:] or (str(p) for p in RAW_DIR.glob("*_depth_*.csv")))
    print("Input files:", *(Path(f).name for f in csv_files), sep="\n  ")

    table = build_decode_error_table(csv_files)
    print(f"Wrote {len(table)} rows to {RESULTS_DIR / 'decode_error_table.csv'}\n")

    held_out = table[table["fit_or_heldout"] == "held-out"]
    print("Held-out mean signed error (> 0: model too fast):")
    print(held_out.groupby(["platform", "codec", "depth"])["signed_error"].mean().unstack("depth").round(3))
