"""Energy constants for the energy model, plus the helpers that look them up.

Every constant is a plain dict with the same keys:
    unit   : string label of the unit (all values are SI: J/byte, J/op, W)
    raw    : the source value (Horowitz's number converted to the unit, or our guess)
    low, nom, high : dimensionless scale factors. The value actually used is raw * scale.
    source : where raw came from
    label  : "published", "scaled from published", or "pure assumption"
    group  : which scenario column applies ("op", "sram", "dram", "ssd", "static")

Horowitz numbers are 45 nm, 0.9 V, DDR3-era. Scale factors map them to our hardware
(Intel 4 CPU/iGPU with LPDDR5, RTX 4060 with GDDR6). All scale factors are assumptions.
TODO: verify every raw value against the paper and fill in the figure/table number.
"""

import math

KIB = 1024
MIB = 1024 ** 2

HOROWITZ = "Horowitz, 'Computing's Energy Problem', ISSCC 2014, energy table (45 nm, 0.9 V). TODO: fig/table #"

# Shared scale columns. value used = raw * scale.
# Logic: dynamic energy ~ C*V^2. 45 nm/0.9 V -> Intel 4-class at ~0.7 V gives roughly 0.1x to 0.4x.
OP_SCALE = {"low": 0.1, "nom": 0.2, "high": 0.4}
# SRAM access energy is dominated by bitlines and wires, which scale worse than logic.
SRAM_SCALE = {"low": 0.25, "nom": 0.4, "high": 0.7}
# DRAM: raw is ~30 pJ/bit. LPDDR5 / GDDR6 are roughly 4 to 12 pJ/bit.
DRAM_SCALE = {"low": 0.12, "nom": 0.2, "high": 0.4}
# Pure assumptions: a factor of 2 either way.
GUESS_SCALE = {"low": 0.5, "nom": 1.0, "high": 2.0}


E_CONST = {
    # ---- Arithmetic (per op). Horowitz gives add and mul separately; e_per_flop() combines them.
    "f16_add":  {"unit": "J/op", "raw": 0.4e-12,  **OP_SCALE, "source": HOROWITZ + ", 16b FP add",
                 "label": "scaled from published", "group": "op"},
    "f16_mul":  {"unit": "J/op", "raw": 1.1e-12,  **OP_SCALE, "source": HOROWITZ + ", 16b FP mult",
                 "label": "scaled from published", "group": "op"},
    "f32_add":  {"unit": "J/op", "raw": 0.9e-12,  **OP_SCALE, "source": HOROWITZ + ", 32b FP add",
                 "label": "scaled from published", "group": "op"},
    "f32_mul":  {"unit": "J/op", "raw": 3.7e-12,  **OP_SCALE, "source": HOROWITZ + ", 32b FP mult",
                 "label": "scaled from published", "group": "op"},
    "int8_add": {"unit": "J/op", "raw": 0.03e-12, **OP_SCALE, "source": HOROWITZ + ", 8b int add",
                 "label": "scaled from published", "group": "op"},
    "int8_mul": {"unit": "J/op", "raw": 0.2e-12,  **OP_SCALE, "source": HOROWITZ + ", 8b int mult",
                 "label": "scaled from published", "group": "op"},

    # ---- SRAM (per byte). Horowitz gives energy per 64-bit (8-byte) access, so divide by 8.
    "sram_8KiB":  {"unit": "J/byte", "raw": 10e-12 / 8,  "size_bytes": 8 * KIB,  **SRAM_SCALE,
                   "source": HOROWITZ + ", 8 KB SRAM, 64b access", "label": "scaled from published",
                   "group": "sram"},
    "sram_32KiB": {"unit": "J/byte", "raw": 20e-12 / 8,  "size_bytes": 32 * KIB, **SRAM_SCALE,
                   "source": HOROWITZ + ", 32 KB SRAM, 64b access", "label": "scaled from published",
                   "group": "sram"},
    "sram_1MiB":  {"unit": "J/byte", "raw": 100e-12 / 8, "size_bytes": 1 * MIB,  **SRAM_SCALE,
                   "source": HOROWITZ + ", 1 MB SRAM, 64b access", "label": "scaled from published",
                   "group": "sram"},

    # ---- DRAM (per byte). Horowitz: 1.3 to 2.6 nJ per 64-bit access; raw is the midpoint / 8 bytes.
    "e_dram": {"unit": "J/byte", "raw": (1.3e-9 + 2.6e-9) / 2 / 8, **DRAM_SCALE,
               "source": HOROWITZ + ", DRAM 64b access (midpoint of 1.3-2.6 nJ)",
               "label": "scaled from published", "group": "dram"},

    # ---- SSD (per byte). Placeholder until M3: ~5 W active at ~5 GB/s is about 1 nJ/byte.
    "e_ssd": {"unit": "J/byte", "raw": 1e-9, **GUESS_SCALE,
              "source": "assumed: NVMe active power / sequential read BW", "label": "pure assumption",
              "group": "ssd"},

    # ---- Static power (W) per platform: everything not counted as DRAM, SRAM or ops.
    "p_static_cpu":  {"unit": "W", "raw": 4.0,  **GUESS_SCALE,
                      "source": "assumed: 185H package uncore + leakage", "label": "pure assumption",
                      "group": "static"},
    "p_static_igpu": {"unit": "W", "raw": 4.0,  **GUESS_SCALE,
                      "source": "assumed: 185H package uncore + leakage (same package as CPU)",
                      "label": "pure assumption", "group": "static"},
    "p_static_4060": {"unit": "W", "raw": 12.0, **GUESS_SCALE,
                      "source": "assumed: RTX 4060 laptop idle-while-active + host package",
                      "label": "pure assumption", "group": "static"},
}


# Which scale column each group uses in each scenario.
# e_pessimistic hurts the tier (cheap DRAM, expensive SRAM/ops/SSD, low static), e_optimistic is the reverse.
SCENARIOS = {
    "e_nominal":     {"op": "nom",  "sram": "nom",  "dram": "nom",  "ssd": "nom",  "static": "nom"},
    "e_pessimistic": {"op": "high", "sram": "high", "dram": "low",  "ssd": "high", "static": "low"},
    "e_optimistic":  {"op": "low",  "sram": "low",  "dram": "high", "ssd": "low",  "static": "high"},
}

# Arithmetic type each platform's matmuls run in.
# CPU: llama.cpp runs Q4_K x Q8_K dot products in int8. GPUs: f16.
PLATFORM_OP_TYPE = {"cpu": "int8", "igpu": "f16", "4060": "f16"}

SRAM_KEYS = ("sram_8KiB", "sram_32KiB", "sram_1MiB")


###################################################################################
#   LOOKUP HELPERS
###################################################################################
def energy_const(name: str, scenario: str = "e_nominal") -> float:
    """Value of one constant under a scenario: raw * (that group's scale column)."""
    c = E_CONST[name]
    column = SCENARIOS[scenario][c["group"]]
    return c["raw"] * c[column]


def e_per_flop(platform: str, scenario: str = "e_nominal") -> float:
    """Energy per FLOP (J). Industry convention: one multiply-add = 2 FLOPs,
    so one FLOP costs (e_mul + e_add) / 2."""
    op = PLATFORM_OP_TYPE[platform]
    return (energy_const(f"{op}_mul", scenario) + energy_const(f"{op}_add", scenario)) / 2


def p_static(platform: str, scenario: str = "e_nominal") -> float:
    """Static power (W) for a platform."""
    return energy_const(f"p_static_{platform}", scenario)


def e_sram_per_byte(size_bytes: float, scenario: str = "e_nominal") -> float:
    """SRAM energy per byte (J) for an array of size_bytes. Log-log interpolation between
    the Horowitz points; outside them, extrapolates the nearest segment's slope.
    Caveat: extrapolating far past 1 MiB treats it as one monolithic array, which is pessimistic.
    Real large SRAMs are banked."""
    pts = sorted((E_CONST[k]["size_bytes"], energy_const(k, scenario)) for k in SRAM_KEYS)
    xs = [math.log(s) for s, _ in pts]
    ys = [math.log(e) for _, e in pts]
    x = math.log(size_bytes)
    # segment index: the last point at or below x, clamped so i and i+1 both exist
    i = max(0, min(len(xs) - 2, sum(1 for xi in xs if xi <= x) - 1))
    slope = (ys[i + 1] - ys[i]) / (xs[i + 1] - xs[i])
    return math.exp(ys[i] + slope * (x - xs[i]))