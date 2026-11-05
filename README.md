# edge-kvtier

**An Agent-Aware Memory Tier for Edge SoCs.** 

LLM agents run in a loop: read a long, mostly unchanged context, emit a short action, wait for a tool, repeat. On an edge SoC the KV cache for that context grows every step and competes for one shared DRAM pool. This project asks how much DRAM traffic, latency and energy an agent-aware memory tier can remove beyond software prefix caching: M1, a pinned on-chip prefix scratchpad; M2, a KV quantize/dequantize codec in the memory path; and M3, a tier controller that migrates KV across SRAM, DRAM and flash between steps. The project also aims to give a sizing rule for building one.

This README records the **preliminary workload characterization** (October 2026) that the architecture work builds on.

---

## Repository layout

```
bench/results/raw/   llama-bench CSV output, one file per run: <date>_<engine>_<test>.csv
model/model.py       analytical model: agent-loop token counts and KV-cache size
tests/               pytest tests (placeholder)
results/             (empty) reserved for model and simulator outputs
pyproject.toml, uv.lock, .python-version   Python environment (uv)
```

Model weights (`*.gguf`) and llama.cpp binaries are not in the repo (see `.gitignore`). The section [Reproducing](#reproducing) lists where to get them.

---

## Preliminary benchmark: workload characterization

### Question

Before designing the tier, we needed to know three things on real edge hardware:

1. How much does a growing context (KV cache) slow decode?
2. Is that slowdown caused by **memory traffic** (which a memory tier can attack) or by **compute** (which it cannot)?
3. How much does compressing the KV cache (a software preview of M2) actually recover?

### Setup

| Item | Value |
|---|---|
| Machine | Laptop, Windows, AC power, "Best performance" power mode |
| CPU | Intel Core Ultra 9 185H (Meteor Lake-H): 6 P-cores + 8 E-cores + 2 LP E-cores, 22 threads |
| System memory | 16 GB DDR5-5600 SODIMM, dual-channel, 89.6 GB/s theoretical peak |
| Integrated GPU | Intel Arc Graphics. Shares system DDR5 (Vulkan reports `uma: 1`, `matrix cores: none`) |
| Discrete GPU | NVIDIA GeForce RTX 4060 Laptop GPU. Own 8 GB GDDR6, ~256 GB/s spec (`uma: 0`, `matrix cores: NV_coopmat2`) |
| Inference engine | llama.cpp build **b11889** (commit `d854d98ba`), prebuilt Windows binaries: CPU (auto-selected `ggml-cpu-alderlake` backend with AVX2, AVX-VNNI and FMA) and Vulkan |
| Model | Llama-3.2-3B-Instruct, **Q4_K_M** GGUF. 3.21 B parameters, 2,011,539,712 bytes (2.01 GB = 1.87 GiB) |
| KV cache (f16) | 2 × 28 layers × 8 KV heads × 128 head dim × 2 B = **114,688 B/token** (114.7 MB per 1,000 tokens) |
| KV formats tested | f16; q8_0 (34 B per 32 values, 0.53× f16); q4_0 (18 B per 32 values, 0.28× f16) |
| Fixed settings (Oct 27 runs) | flash attention on (`-fa 1`); CPU: 16 threads; GPUs: all layers offloaded (`-ngl 99`); 3 repetitions (`-r 3`) |

Dual-channel memory is confirmed indirectly: the CPU sustained 50 GB/s, which is more than single-channel DDR5-5600 can deliver (44.8 GB/s).

Unit note: the proposal's end-of-loop KV size of "853 MB" (7,800 tokens × 114,688 B) is 853 **MiB** = 894 MB. `model/model.py` prints it correctly in MiB.

### Method

`llama-bench` reports two tests:

- **`pp512`** (prompt processing, i.e. prefill) times 512 tokens processed as one batch.
- **`tg64`** (text generation, i.e. decode) times 64 tokens generated one at a time.

`-d N` first fills the KV cache with N tokens (untimed), then runs the timed test on top of it. This simulates an agent that already holds N tokens of context. We swept N = 0, 2,048, 4,096 and 8,192.

The analysis rests on four rules:

- **Work in time per token, not tokens per second.** Time components add; rates do not. Decode time per token = time to read the weights + time spent on the context. Each rate is converted to ms per token (1000 ÷ tok/s).
- **Context cost** is the slope of ms per token against context length, in **ms per 1,000 tokens of context**. It is fitted between 2,048 and 8,192 tokens so that one noisy depth-0 point cannot skew it. Segment slopes are reported where the curve is not linear.
- **Effective bandwidth** is decode tok/s at depth 0 × 2.01 GB, since each decoded token reads every weight once. The **KV read rate** is the KV bytes per 1,000 tokens divided by the context cost. Comparing the two shows whether the engine reads the KV cache as efficiently as it reads weights.
- **Controls and sanity checks.** At depth 0 the cache is empty, so runs that differ only in KV format must agree. Any result implying more bandwidth than the hardware can deliver means the wrong device was measured. That check caught one mislabelled run (see [Data files](#data-files)).

---

## Results

### 1. CPU thread sweep (depth 0, f16 KV)

| Threads | Prefill pp512 (tok/s) | Decode tg64 (tok/s) | Effective bandwidth |
|---:|---:|---:|---:|
| 4 | 86.8 | 17.0 | 34 GB/s |
| 6 | 114.8 | 19.5 | 39 GB/s |
| 8 | 129.0 | 20.8 | 42 GB/s |
| 12 | 154.1 | 22.8 | 46 GB/s |
| 16 | 175.1 | 24.9 | 50 GB/s |

From 4 to 16 threads, prefill rose **102%** and decode rose only **47%**. Decode had not fully plateaued at 16 threads. All later CPU runs use 16 threads.

### 2. Decode speed against context depth (tok/s)

| Engine | KV | d = 0 | 2,048 | 4,096 | 8,192 | Drop 0 → 8k |
|---|---|---:|---:|---:|---:|---:|
| CPU | f16 | 24.17 | 18.12 | 14.33 | 8.61 | 64% |
| CPU | q8_0 | 24.26 | 17.55 | 13.63 | 9.18 | 62% |
| CPU | q4_0 | 23.83 | 18.18 | 14.65 | 10.53 | 56% |
| iGPU | f16 | 21.99 | 18.89 | 16.36 | 13.18 | 40% |
| iGPU | q4_0 | 22.81 | 20.01 | 18.22 | 15.08 | 34% |
| RTX 4060 | f16 | 104.98 | 94.99 | 86.24 | 73.22 | 30% |
| RTX 4060 | q4_0 | 100.52 | 94.66 | 87.71 | 77.32 | 23% |

At 8k tokens of context, f16 decode is **2.8×** slower than at depth 0 on the CPU, **1.7×** on the iGPU and **1.4×** on the RTX 4060. A bandwidth-only model predicts 1.47× on every engine (2.95 GB read per token ÷ 2.01 GB).

### 3. Cost of context and how efficiently each engine reads KV

| Engine | Effective bandwidth | f16 context cost (ms per 1k tokens) | q4_0 context cost | q4_0 ÷ f16 (bytes: 0.28) | f16 KV read rate | KV rate ÷ weight rate |
|---|---:|---:|---:|---:|---:|---:|
| CPU | 48.6 GB/s | 6.7 (0–2k), 7.1 (2–4k), **11.3** (4–8k) | 6.4–6.5 | 0.94, 0.91, 0.57 | 17 → 10 GB/s | 0.35 → 0.21 |
| iGPU | 44.2 GB/s | 3.73 | 2.66 | 0.71 | 31 GB/s | 0.70 |
| RTX 4060 | 211 GB/s | 0.51 | 0.39 | 0.76 | 225 GB/s | 1.07 |

On the CPU, q8_0 costs **7.7–8.7 ms per 1k tokens**. Up to 4k tokens that is more than f16, despite having half the bytes.

### 4. Accuracy of a bandwidth-only model (f16 KV)

The model is calibrated on each engine's depth-0 decode and predicts deeper contexts from KV bytes alone.

| Engine | 2,048 | 4,096 | 8,192 |
|---|---:|---:|---:|
| CPU | −16% | −27% | **−48%** |
| iGPU | −4% | −8% | **−12%** |
| RTX 4060 | +1.0% | +1.3% | **+2.3%** |

### 5. Prefill speed against context depth (GPUs, pp512, tok/s)

| Engine | KV | d = 0 | 2,048 | 4,096 | 8,192 | Drop 0 → 8k |
|---|---|---:|---:|---:|---:|---:|
| iGPU | f16 | 352.8 | 183.8 | 124.9 | 76.9 | 78% |
| iGPU | q4_0 | 338.5 | 205.4 | 143.6 | 89.7 | 73% |
| RTX 4060 | f16 | 5,236 | 4,810 | 4,445 | 3,586 | 32% |
| RTX 4060 | q4_0 | 4,898 | 4,661 | 4,092 | 3,278 | 33% |

Effect of q4_0 KV on prefill compared with f16:
- **iGPU:** −4% at depth 0, then **+12%, +15%, +17%** at 2k, 4k and 8k.
- **RTX 4060:** −3% to −9% at every depth.

CPU prefill at depth has not been measured yet. At depth 0 with 16 threads it is 175 tok/s, so the iGPU is 2.0× faster and the RTX 4060 is 30× faster.

---

## Findings

1. **Context growth is a large cost on every engine.** f16 decode at 8k context is 1.4–2.8× slower than at an empty context. Agent loops reach this range: the proposal's 12-step loop ends near 7,800 tokens.

2. **Decode is memory-bound on the RTX 4060. On the CPU and iGPU, weight streaming is consistent with memory-bound but not yet proven.**
   - **RTX 4060:** a model calibrated only at depth 0 predicts the slowdown at 2k–8k from KV bytes within 2.3%, and the 211 GB/s it reaches is ~82% of the ~256 GB/s GDDR6 spec.
   - **CPU and iGPU:** the iGPU has 2× the CPU's prefill throughput but slightly lower decode (22.0 versus 24.2 tok/s), so decode on these two engines is not limited by compute. They also reach similar effective bandwidth (44–50 GB/s) from the same DDR5.
   - **What's still unknown:** that is only 49–56% of the DDR5 theoretical peak, and without a dedicated memory-bandwidth test we cannot tell whether it is the achievable ceiling.

3. **The context part of decode (attention over the KV cache) behaves differently on each engine.**
   - **RTX 4060:** it reads KV at full bandwidth (KV rate ≈ weight rate), so it is memory-bound.
   - **iGPU:** it reads KV at 70% of the weight rate, so it is mostly memory-bound with some compute overhead.
   - **CPU:** it reads KV at 21–35% of the weight rate. Below 4k tokens, q4_0 (72% fewer bytes) costs almost the same time as f16, so the cost is **attention compute, not memory traffic**.
   - The accuracy of a bandwidth-only model follows the same order: 2% error, then 12%, then 48%.

4. **Software KV compression recovers only part of its byte savings.** Even on the GPUs, where KV reads are largely memory-bound, q4_0 cuts context cost by 24–29% while cutting bytes by 72%. The difference is the compute cost of quantizing on write and dequantizing on read.

5. **On the iGPU, prefill at depth dominates late agent steps.** iGPU prefill drops 78% from 0 to 8k context. A step with 400 new tokens and 50 generated tokens takes about 1.1 s of prefill and 2.3 s of decode at depth 0, but about **5.2 s of prefill and 3.8 s of decode at 8k**. Prefix caching (B1) avoids recomputing the prefix, but every new token still attends to all of it.

6. **Whether KV compression pays off depends on the platform.** On the iGPU, where bandwidth is scarce, q4_0 makes prefill at depth 12–17% faster. On the RTX 4060, where bandwidth is plentiful, it makes prefill 3–9% slower.

---

## What this means for the architecture plan

### Supports the plan

- The problem is real on edge hardware: KV growth across an agent loop slows decode by 1.4–2.8× by 8k tokens.
- On the shared-memory iGPU, which is the edge-SoC case the tier targets, KV reads are mostly memory-bound. That is the regime where M1 (keep KV on chip) and M2 (move fewer bytes) can help.
- **A first measured number for M2.** At 8k context on the iGPU:
  - Software q4_0 KV makes decode **14%** faster than f16 (15.08 versus 13.18 tok/s).
  - If the compressed KV were read as efficiently as f16 KV, with a zero-overhead decompressor, decode would be up to **~40%** faster (≈18.5 tok/s).
  - The gap between the two is the most an in-path hardware codec could recover over software. It is an upper-bound estimate that assumes compressed KV is read at f16's KV read rate.
  - The same estimate on the RTX 4060 is 6% today against up to ~28%.
- The bandwidth-based traffic model is validated on a memory-bound engine (≤ 2.3% error on the RTX 4060), so it can serve as the core of the simulator where it applies.

### Complicates the plan

- **CPU path:** the context cost is attention compute, so a memory tier alone does little there. The tier's case depends on attention running on a GPU or NPU.
- **Software dequantization overhead** eats most of the compression benefit even on memory-bound engines. M2's value therefore depends on the codec's own cost, which these benchmarks cannot measure; it needs an RTL or literature estimate.
- **Prefill at depth**, not only decode, dominates late steps on the iGPU. The B1 versus P comparison has to account for KV reads during prefill of new tokens.
- **Measured DDR5 bandwidth is lower than assumed:** 44–50 GB/s achieved against the 65 GB/s used in early estimates.
- **M3** (migration across tool-call gaps) is **not tested** by these benchmarks. We have no evidence on it yet, for or against.

### Proposed decisions for the architecture phase (for team review)

1. **Primary target:** the shared-memory iGPU path. The RTX 4060 serves as the memory-bound reference and the CPU as the compute-bound counter-example.
2. **Model form.** Calibrate per engine and KV format from the constants below:
   - decode time per token = W ÷ BW + (d ÷ 1000) × c
   - prefill time for n new tokens ≈ n × (a + b × d ÷ 1000)

   A bandwidth-only model is acceptable only on the dGPU.
3. **M1 sizing starting point.** The proposal's 3,000-token prefix is 344 MB of KV in f16, 183 MB in q8_0 and 97 MB in q4_0. That is far above typical on-chip SRAM (tens of MB), so M1 must hold a compressed or hot subset. The sizing rule has to be stated in those terms.
4. **Report M2 as an upper bound** together with a codec cost model.
5. **Include prefill at depth** in the agent workload and in the simulator.

### Calibrated constants (Core Ultra 9 185H laptop, llama.cpp b11889, Llama-3.2-3B Q4_K_M)

| Engine | KV | BW (GB/s) | Decode c (ms per token per 1k context) | Prefill a (ms per token) | Prefill b (ms per token per 1k depth) |
|---|---|---:|---:|---:|---:|
| CPU, 16 threads | f16 | 48.6 | 6.7–7.1 below 4k; 11.3 from 4k to 8k | 5.71 | not measured |
| CPU, 16 threads | q8_0 | 48.8 | 7.7–8.7 | — | — |
| CPU, 16 threads | q4_0 | 47.9 | 6.4–6.5 | — | — |
| iGPU (Vulkan) | f16 | 44.2 | 3.73 | 2.83 | 1.24–1.27 |
| iGPU (Vulkan) | q4_0 | 45.9 | 2.66 | 2.95 | 0.94–1.00 |
| RTX 4060 (Vulkan) | f16 | 211.2 | 0.51 | 0.19 | ~0.01 |
| RTX 4060 (Vulkan) | q4_0 | 202.2 | 0.39 | 0.20 | ~0.01 |

W = 2.01 GB. The prefill constants are fitted from pp512 runs.

---

## Limitations and open measurements

- **Achievable DDR5 bandwidth is unknown.** Measure it with Intel Memory Latency Checker to tell whether the CPU and iGPU decode at the memory ceiling.
- **No energy data yet.** Energy per task is a primary metric in the proposal. Next step: HWiNFO package power for the CPU and iGPU, and NVML for the RTX 4060.
- **Run-to-run variation is larger than the within-run ±.** Depth-0 controls differ by 3.7% (iGPU) and 4.4% (RTX 4060) between runs whose own ± is mostly below 1%. Slopes cancel constant offsets, but key configurations should be repeated on another day, after a restart.
- **The CPU f16 jump between 4k and 8k** (7.1 to 11.3 ms per 1k tokens) is unexplained. Extra depths (6k, 10k, 12k) would tell whether it is a threshold or a curve.
- **CPU prefill at depth** has not been measured. Command below.
- **Synthetic context.** `-d` fills the cache with synthetic tokens. Real ReAct traces (token counts, prefix reuse, tool-call gaps) are the next workload step.
- **One machine.** This is one Intel CPU and iGPU and one dGPU. The second platform per architecture is still to do.
- **Vulkan only on the GPUs.** CUDA (NVIDIA) and SYCL (Intel) backends may differ and are worth one comparison run.
- **Unrecorded metadata:** GPU driver versions and Windows build. Record them with `Get-CimInstance Win32_VideoController | Select-Object Name, DriverVersion`.

---

## Data files

All files are in `bench/results/raw/`. They were written by PowerShell 5.1 as UTF-8 with a byte-order mark.

| File | Contents | Status |
|---|---|---|
| `2026-10-25_cpu_t6.csv` | First run: 6 threads, flash attention auto, pp512 and tg128 | Exploratory, superseded |
| `2026-10-27_cpu_threadsweep.csv` | 4/6/8/12/16 threads, pp512 and tg64 | Used |
| `2026-10-27_cpu_depth_kvf16.csv` | CPU decode at depth 0–8k, f16 KV | Used |
| `2026-10-27_cpu_depth_kvq8.csv` | CPU decode at depth, q8_0 KV | **Discarded:** run was disturbed (depth-0 control 20.4 ± 3.0 against ~24) |
| `2026-10-27_cpu_depth_kvq8_rerun.csv` | CPU decode at depth, q8_0 KV | Used |
| `2026-10-27_cpu_depth_kvq4.csv` | CPU decode at depth, q4_0 KV | Used |
| `2026-10-27_vulkan-igpu_depth_kvf16.csv` | iGPU (`-dev Vulkan1`), pp512 and tg64 at depth, f16 | Used |
| `2026-10-27_vulkan-igpu_depth_kvq4.csv` | iGPU (`-dev Vulkan1`), q4_0 | Used |
| `2026-10-27_vulkan-rtx4060_depth_kvf16.csv` | RTX 4060, f16 | Used. Run with default device selection (`devices = auto`); identified as the RTX 4060 because 105 tok/s implies 211 GB/s, above the DDR5 ceiling. Re-run with `-dev Vulkan0` for clean provenance |
| `2026-10-27_vulkan-rtx4060_depth_kvq4.csv` | RTX 4060, q4_0 | Same as above |

Reading the files in Python:

```python
import pandas as pd
df = pd.read_csv("bench/results/raw/2026-10-27_vulkan-igpu_depth_kvf16.csv", encoding="utf-8-sig")
# key columns: n_threads, type_k, type_v, n_prompt, n_gen, n_depth, devices, avg_ts, stddev_ts
```

---

## Reproducing

1. Download llama.cpp release **b11889** from https://github.com/ggml-org/llama.cpp/releases/tag/b11889: the Windows x64 CPU zip and the Vulkan zip.
2. Download `Llama-3.2-3B-Instruct-Q4_K_M.gguf` from `bartowski/Llama-3.2-3B-Instruct-GGUF` on Hugging Face.
3. Before running, plug in, set Best performance, close background apps, and confirm idle CPU is below 3% in Task Manager.
4. In PowerShell, from the extracted llama.cpp folder:

```powershell
$M = "<path>\Llama-3.2-3B-Instruct-Q4_K_M.gguf"
$R = "<path>\edge-kvtier\bench\results\raw"

# CPU thread sweep
.\llama-bench.exe -m $M -p 512 -n 64 -t 4,6,8,12,16 -fa 1 -r 3 -o csv -oe md | Out-File -Encoding utf8 "$R\<date>_cpu_threadsweep.csv"

# CPU decode vs depth (repeat with -ctk/-ctv q8_0 and q4_0)
.\llama-bench.exe -m $M -p 0 -n 64 -d 0,2048,4096,8192 -t 16 -fa 1 -ctk f16 -ctv f16 -r 3 -o csv -oe md | Out-File -Encoding utf8 "$R\<date>_cpu_depth_kvf16.csv"

# CPU prefill vs depth (not yet run)
.\llama-bench.exe -m $M -p 512 -n 0 -d 0,2048,4096,8192 -t 16 -fa 1 -ctk f16 -ctv f16 -r 3 -o csv -oe md | Out-File -Encoding utf8 "$R\<date>_cpu_prefill-depth_kvf16.csv"

# GPU (Vulkan build): Vulkan0 = RTX 4060, Vulkan1 = Intel iGPU on this machine; check the startup device list
.\llama-bench.exe -m $M -ngl 99 -dev Vulkan1 -p 512 -n 64 -d 0,2048,4096,8192 -fa 1 -ctk f16 -ctv f16 -r 3 -o csv -oe md | Out-File -Encoding utf8 "$R\<date>_vulkan-igpu_depth_kvf16.csv"
```

`Out-File` does not create folders, so create `bench\results\raw` first.

Python environment: `uv sync`, then `uv run python model/model.py` and `uv run pytest`.
