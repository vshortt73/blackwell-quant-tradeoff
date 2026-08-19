"""Tests for curve-existence detection. Run: python harness/test_apex_curve.py"""
from __future__ import annotations
import math, random, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from apex_curve import detect_curve  # noqa: E402

POSITIONS = [0.02,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95,0.98]

def gen(fn, reps=5, noise=0.0, seed=0, n_probes=1, probe_spread=0.0):
    """(probe_id, position, score). probe_spread gives each probe its own
    difficulty offset -- the nuisance term the detector must cancel."""
    rng = random.Random(seed)
    offs = [rng.uniform(-probe_spread, probe_spread) for _ in range(n_probes)]
    out=[]
    for pi in range(n_probes):
        for p in POSITIONS:
            for _ in range(reps):
                v = fn(p) + offs[pi] + (rng.gauss(0, noise) if noise else 0.0)
                out.append((f"P{pi:03d}", p, min(1.0, max(0.0, v))))
    return out

def main() -> int:
    fails=[]
    def check(name, cond):
        print(f"  [{'ok  ' if cond else 'FAIL'}] {name}")
        if not cond: fails.append(name)

    # flat + noise -> position explains nothing
    r = detect_curve(gen(lambda p: 0.7, noise=0.15, seed=1), n_perm=2000)
    check("flat noisy signal -> no curve", not r.curve_exists)
    print(f"         eta2={r.curve_strength:.4f} p={r.p_value:.4f}")

    # U-shape (lost-in-the-middle) -- the case a correlation would MISS
    u = gen(lambda p: 0.4 + 0.5*(2*p-1)**2, noise=0.05, seed=2)
    r = detect_curve(u, n_perm=2000)
    check("U-shaped curve detected", r.curve_exists)
    print(f"         eta2={r.curve_strength:.4f} p={r.p_value:.4f}")
    # confirm a monotonic measure would have failed here
    xs=[p for _,p,_ in u]; ys=[s for _,_,s in u]
    mx,my=sum(xs)/len(xs),sum(ys)/len(ys)
    num=sum((a-mx)*(b-my) for a,b in zip(xs,ys))
    den=math.sqrt(sum((a-mx)**2 for a in xs)*sum((b-my)**2 for b in ys))
    pear=abs(num/den) if den else 0.0
    print(f"         |pearson| on same data = {pear:.4f}  <- a correlation-based test would call this noise")
    check("pearson would miss the U-shape", pear < 0.2)

    # monotonic decay -> detected too
    r = detect_curve(gen(lambda p: 1.0-0.6*p, noise=0.05, seed=3), n_perm=2000)
    check("monotonic decay detected", r.curve_exists)

    # ceiling -> reported as no_variance, NOT as "no curve"
    r = detect_curve(gen(lambda p: 1.0, noise=0.0), n_perm=500)
    check("ceiling -> reason=no_variance", r.reason == "no_variance" and not r.curve_exists)

    # degenerate inputs
    r = detect_curve([("P0", 0.5, 1.0)], n_perm=100)
    check("single position -> insufficient_positions", r.reason == "insufficient_positions")
    r = detect_curve([("P0",0.1,1.0),("P0",0.9,0.0)], n_perm=100)
    check("too few observations -> insufficient_observations", r.reason == "insufficient_observations")

    # determinism
    d = gen(lambda p: 0.4+0.5*(2*p-1)**2, noise=0.05, seed=4)
    check("deterministic", detect_curve(d, n_perm=1000).as_dict() == detect_curve(d, n_perm=1000).as_dict())

    # p can never be exactly 0
    r = detect_curve(gen(lambda p: p, noise=0.001, seed=5), n_perm=1000)
    check("p-value never zero", r.p_value > 0.0)

    # strength ordering: stronger signal -> larger eta-squared
    weak = detect_curve(gen(lambda p: 0.7+0.05*(2*p-1)**2, noise=0.15, seed=6), n_perm=1500)
    strong = detect_curve(gen(lambda p: 0.3+0.6*(2*p-1)**2, noise=0.05, seed=6), n_perm=1500)
    check("strength ordering sane", strong.curve_strength > weak.curve_strength)
    print(f"         weak eta2={weak.curve_strength:.4f}  strong eta2={strong.curve_strength:.4f}")

    # THE POINT OF WITHIN-PROBE ANALYSIS: a modest positional effect buried
    # under large probe-to-probe difficulty differences. Measured on real APEX
    # data, probe identity explains 20x more variance than position.
    buried = gen(lambda p: 0.55 + 0.12*(2*p-1)**2, reps=3, noise=0.03,
                 seed=21, n_probes=20, probe_spread=0.35)
    r = detect_curve(buried, n_perm=3000)
    # same data, pooled (probe difficulty NOT removed)
    from apex_curve import _eta_squared
    g={}
    for _pid,p,s_ in buried: g.setdefault(p,[]).append(s_)
    pooled = _eta_squared(list(g.values()))
    print(f"\n  buried effect: pooled eta2={pooled:.4f}  within-probe eta2={r.curve_strength:.4f} "
          f"p={r.p_value:.4f}")
    check("within-probe recovers a probe-buried effect", r.curve_exists)
    check("within-probe beats pooled on the same data", r.curve_strength > pooled * 2)
    check("n_probes recorded", r.n_probes == 20)

    print(f"\n{'PASS' if not fails else 'FAIL: '+', '.join(fails)}")
    return 1 if fails else 0

if __name__ == "__main__":
    raise SystemExit(main())
