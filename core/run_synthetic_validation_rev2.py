# run_synthetic_validation.py  (rev 2 -- assertion-based)
# End-to-end pipeline validation against ground-truth agents (no LLM).
# rev 2 corrections after PI review of the 2026-07-12_1445 report:
#   - V2b: explicit closed-form assertion  q = (r_n - r_y)(p_A - p_B)
#   - V4 : boundary case m = eps with atol; corrected structural-bound wording
#   - V5 : Wilson vs exact Clopper-Pearson calibration; CP is the registered
#          method per SRS v0.7, Wilson retained for comparison only
#   - all checks are hard assertions; nonzero exit on failure
#   - output path relative to the script directory
# ASCII only.

import os
import sys
import numpy as np

from metrics_ex18 import (q_qq, order_effects, q_qq_envelope, envelope_decision,
                          aligned_distances, consistency_check, EPSILON_QQ,
                          _adjusted_joint)
from simulators import (ProjectiveAgent, POVMAgent, AnchoringAgent,
                        AsymmetricAnchoringAgent, AsymmetricPOVMAgent,
                        inject_other_mass, sample_joint, RNG)

rng = RNG(20260712)
report = []
failures = []


def log(line):
    print(line)
    report.append(line)


def check(name, cond):
    status = "PASS" if cond else "FAIL"
    log(f"    [{status}] {name}")
    if not cond:
        failures.append(name)


# --- V1: projective agent must satisfy QQ exactly ---------------------------
log("V1  ProjectiveAgent: q_QQ == 0 exactly, OSS > 0")
max_abs_q, min_oss, max_oss = 0.0, 1.0, 0.0
for _ in range(2000):
    psi, a, b = rng.uniform(0, np.pi, size=3)
    j_ab, j_ba = ProjectiveAgent(psi, a, b).joints()
    max_abs_q = max(max_abs_q, abs(q_qq(j_ab, j_ba)))
    oss = order_effects(j_ab, j_ba)["OSS"]
    min_oss, max_oss = min(min_oss, oss), max(max_oss, oss)
log(f"    max |q_QQ| = {max_abs_q:.3e}; OSS range [{min_oss:.4f}, {max_oss:.4f}]")
check("projective q_QQ at machine precision", max_abs_q < 1e-12)
check("order effects present (max OSS > 0.1)", max_oss > 0.1)

# --- V2: classical models -----------------------------------------------------
log("V2  Classical agents vs QQ")
# V2a symmetric repetition: satisfies QQ identically (interpretive limit of H-A)
max_q_sym = 0.0
for _ in range(2000):
    pa, pb = rng.uniform(0.1, 0.9, size=2)
    r = rng.uniform(0.1, 0.6)
    j_ab, j_ba = AnchoringAgent(pa, pb, r).joints()
    max_q_sym = max(max_q_sym, abs(q_qq(j_ab, j_ba)))
log(f"    V2a symmetric repetition: max |q_QQ| = {max_q_sym:.3e}")
check("symmetric repetition satisfies QQ identically", max_q_sym < 1e-12)

# V2b polarity-dependent repetition: violates QQ; closed form asserted
#     q_QQ = (r_n - r_y) * (p_A - p_B)
violations, max_q, max_closed_err = 0, 0.0, 0.0
for _ in range(2000):
    pa, pb = rng.uniform(0.1, 0.9, size=2)
    ry, rn = rng.uniform(0.0, 0.8, size=2)
    j_ab, j_ba = AsymmetricAnchoringAgent(pa, pb, ry, rn).joints()
    q = q_qq(j_ab, j_ba)
    q_closed = (rn - ry) * (pa - pb)
    max_closed_err = max(max_closed_err, abs(q - q_closed))
    if abs(q) > EPSILON_QQ:
        violations += 1
    max_q = max(max_q, abs(q))
log(f"    V2b polarity-dependent repetition: |q|>eps in {violations}/2000, "
    f"max |q| = {max_q:.4f}, max closed-form error = {max_closed_err:.2e}")
check("closed form q=(r_n-r_y)(p_A-p_B) to machine precision",
      max_closed_err < 1e-12)
check("polarity-dependent repetition violates QQ generically", violations > 500)

# --- V3: POVM agents -----------------------------------------------------------
log("V3  POVM agents vs QQ (violation possibility within asymmetric Lueders class)")
max_q_povm = 0.0
for _ in range(2000):
    psi, a, b = rng.uniform(0, np.pi, size=3)
    j_ab, j_ba = POVMAgent(psi, a, b, eta=0.3).joints()
    max_q_povm = max(max_q_povm, abs(q_qq(j_ab, j_ba)))
log(f"    V3a symmetric unsharp (eta=0.3): max |q_QQ| = {max_q_povm:.3e}")
check("symmetric-noise sqrt-POVM preserves QQ numerically", max_q_povm < 1e-12)
max_q_apovm = 0.0
for _ in range(2000):
    psi, a, b = rng.uniform(0, np.pi, size=3)
    alpha, beta = rng.uniform(0.6, 1.0), rng.uniform(0.0, 0.4)
    j_ab, j_ba = AsymmetricPOVMAgent(psi, a, b, alpha, beta).joints()
    max_q_apovm = max(max_q_apovm, abs(q_qq(j_ab, j_ba)))
log(f"    V3b asymmetric unsharp: max |q_QQ| = {max_q_apovm:.4f}")
check("QQ violation attainable within asymmetric Lueders POVM class",
      max_q_apovm > EPSILON_QQ)

# --- V4: envelope correctness ---------------------------------------------------
log("V4  64-vertex envelope: containment, collapse, boundary, structural bound")
fails = 0
for _ in range(300):
    psi, a, b = rng.uniform(0, np.pi, size=3)
    j_ab, j_ba = ProjectiveAgent(psi, a, b).joints()
    m = rng.uniform(0.0, 0.05, size=6)
    raw_ab = inject_other_mass(j_ab, m[0], m[1], m[2])
    raw_ba = inject_other_mass(j_ba, m[3], m[4], m[5])
    q_min, q_max = q_qq_envelope(raw_ab, raw_ba)
    for _ in range(50):
        t = rng.uniform(0, 1, size=6)
        qi = q_qq(_adjusted_joint(raw_ab, *t[:3]), _adjusted_joint(raw_ba, *t[3:]))
        if not (q_min - 1e-12 <= qi <= q_max + 1e-12):
            fails += 1
log(f"    interior-point violations: {fails}/15000")
check("envelope contains all interior allocation points", fails == 0)

j_ab, j_ba = ProjectiveAgent(0.3, 0.9, 1.7).joints()
raw_ab = inject_other_mass(j_ab, 0.0, 0.0, 0.0)
raw_ba = inject_other_mass(j_ba, 0.0, 0.0, 0.0)
q_min, q_max = q_qq_envelope(raw_ab, raw_ba)
check("zero-mass envelope collapses to a point", abs(q_max - q_min) < 1e-12)

# symmetric equal-mass injection on a projective (q_point = 0) config:
# envelope ~ [-m, +m]; structural bound is m > eps (= 0.02), NOT 1 percent.
j_ab, j_ba = ProjectiveAgent(0.5, 0.4, 1.1).joints()
for m, expect in ((0.01, "SATISFIED"), (0.02, "SATISFIED"),
                  (0.03, "INDETERMINATE"), (0.05, "INDETERMINATE")):
    raw_ab = inject_other_mass(j_ab, m, m, m)
    raw_ba = inject_other_mass(j_ba, m, m, m)
    q_min, q_max = q_qq_envelope(raw_ab, raw_ba)
    dec = envelope_decision(q_min, q_max)
    log(f"    m={m:.2f}: envelope=[{q_min:+.4f},{q_max:+.4f}] -> {dec}")
    check(f"decision at m={m:.2f} equals {expect} (boundary atol active)",
          dec == expect)
log("    NOTE: under symmetric equal-mass injection with q_point=0 the")
log("    envelope is ~[-m,+m]; SATISFIED becomes structurally impossible only")
log("    for m > eps = 0.02. No universal threshold exists for unequal masses.")

# --- V5: consistency-check calibration --------------------------------------------
log("V5  Simultaneous check calibration (alpha=0.01, n_rej=200, 20000 trials)")
trials = 20000
pass_w, pass_cp, detect_cp = 0, 0, 0
for _ in range(trials):
    psi, a, b = rng.uniform(0, np.pi, size=3)
    j_ab, _ = ProjectiveAgent(psi, a, b).joints()
    counts = sample_joint(j_ab, 200, rng)
    ok_w, _, _ = consistency_check(j_ab, counts, method="wilson")
    ok_cp, _, _ = consistency_check(j_ab, counts, method="clopper_pearson")
    pass_w += int(ok_w)
    pass_cp += int(ok_cp)
    shifted = {k: 0.85 * v + 0.15 * 0.25 for k, v in j_ab.items()}
    counts2 = sample_joint(shifted, 200, rng)
    ok2, _, _ = consistency_check(j_ab, counts2, method="clopper_pearson")
    detect_cp += int(not ok2)
rate_w, rate_cp = pass_w / trials, pass_cp / trials
log(f"    Wilson pass rate           = {rate_w:.4f}  (undercoverage vs 0.99)")
log(f"    Clopper-Pearson pass rate  = {rate_cp:.4f}  (registered method, SRS v0.7)")
log(f"    CP detection of 15pct-uniform shift = {detect_cp / trials:.3f}")
check("Clopper-Pearson simultaneous coverage >= 0.99", rate_cp >= 0.99)
check("Wilson undercoverage documented (rate < 0.99)", rate_w < 0.99)

# --- Layer 3 sanity -------------------------------------------------------------
j_ab, j_ba = ProjectiveAgent(0.5, 0.4, 1.1).joints()
d = aligned_distances(j_ab, j_ba)
log(f"L3  aligned distances (projective config): TV={d['TV']:.4f} "
    f"JSD={d['JSD']:.4f} with q_QQ=0 (signed vs unsigned summary)")
check("TV positive while q_QQ = 0 on a projective config", d["TV"] > 0.01)

# --- summary ---------------------------------------------------------------------
log("")
if failures:
    log(f"RESULT: {len(failures)} FAILURE(S): " + "; ".join(failures))
else:
    log("RESULT: all assertions passed")

out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "validation_output.txt")
with open(out_path, "w") as f:
    f.write("\n".join(report) + "\n")

sys.exit(1 if failures else 0)
