"""Run approach 3 end to end on synthetic layers, with the recipes in config/recipes.

    uv run python scripts/demo_suitability.py [--nbs agroforestry --family cross_family] [--png out.png]

The layers are made up (smooth gradients plus noise) so the pipeline can be tested without Earth
Engine. Their ranges are chosen to overlap the recipes' nodes only partly, so the "outside the
evidence" state shows up. Nothing here is a real place.
"""
from __future__ import annotations

import argparse

import numpy as np

from nbs_runtime.recipes import load_family
from nbs_runtime.sensitivity import dirichlet, equal_weights, leave_one_out
from nbs_runtime.suitability import STATUS, classify, compute

RANGES = {                                   # synthetic range of each variable
    "annual_precipitation": (500, 1600),
    "elevation": (300, 2200),
    "slope": (1, 15),
    "ndvi": (0.15, 0.6),
    "mean_annual_temperature": (12, 28),
    "soil_ph": (4.5, 7.5),
    "soil_organic_carbon": (5, 60),
}


def synthetic_layers(variables, shape=(120, 160), seed=0):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:1:complex(shape[0]), 0:1:complex(shape[1])]
    layers = {}
    for k, v in enumerate(variables):
        lo, hi = RANGES.get(v, (0, 1))
        a = np.sin(3 * xx + k) * np.cos(2 * yy + 2 * k)
        f = (a - a.min()) / (a.max() - a.min())
        layers[v] = lo + (hi - lo) * np.clip(f + rng.normal(0, 0.05, shape), 0, 1)
    eligible = rng.random(shape) > 0.15        # 15% of the pixels: land cover not eligible
    return layers, eligible


def pct(x):
    return f"{100 * x:5.1f}%"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--nbs", default="agroforestry")
    ap.add_argument("--family", default="cross_family")
    ap.add_argument("--png", help="save maps of L, M, S and the classes to this file")
    args = ap.parse_args(argv)

    fam = load_family(args.nbs, args.family)
    print(f"\n{fam.nbs}/{fam.family}")
    print("  limiting:  ", ", ".join(r.variable for r in fam.limiting))
    print("  modulating:", ", ".join(f"{v} {w:.3f}" for v, w in fam.modulating_weights().items()))

    layers, eligible = synthetic_layers(fam.variables)
    base = compute(fam, layers, eligible)
    n = base.S.size

    print("\nPixel states")
    for name, c in base.status_counts().items():
        print(f"  {pct(c / n)}  {name}")
    print("\nShare of eligible pixels outside each recipe's evidence")
    for v, s in base.outside_share(layers).items():
        r = fam.recipe(v)
        print(f"  {pct(s)}  {v:22s} nodes {r.x[0]:g} to {r.x[-1]:g} {r.unit}")
    print("\nWhich limiting variable sets L (scored pixels)")
    for v, c in base.driver_counts().items():
        print(f"  {c:8,d}  {v}")

    cls = classify(base.S)
    print(f"\nClasses ({cls.k} above 0{'; ' + cls.note if cls.note else ''})")
    print("  absolute S over positive scores: "
          + ", ".join(f"{k} {v:.3f}" for k, v in cls.summary.items()))
    for k in range(cls.k + 1):
        print(f"  class {k}: {int((cls.classes == k).sum()):,} pixels")

    print("\nSensitivity")
    eq = equal_weights(fam, layers, eligible, base=base)
    print(f"  1. equal weights: Spearman {eq['spearman']:.3f}, top quintile retained "
          f"{pct(eq['top_retained'])}, mean |change| {eq['mean_abs_change']:.3f}")
    d = dirichlet(fam, layers, eligible, n_draws=300, concentration=50, base=base)
    s = d.summary()
    print(f"  2. Dirichlet (300 draws, concentration 50): Spearman median {s['spearman_median']:.3f} "
          f"(p05 {s['spearman_p05']:.3f}); base top quintile robust (>=80% of draws) "
          f"{pct(s['base_top_that_is_robust'])}; mean 5-95% range {s['mean_range_5_95']:.3f}")
    print("  3. leave one out:")
    for v, o in leave_one_out(fam, layers, eligible, base=base).items():
        print(f"     without {v:22s} ({o['role']:10s}) Spearman {o['spearman']:.3f}, "
              f"top retained {pct(o['top_retained'])}, newly scored {o['newly_scored']:,}")

    if args.png:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 4, figsize=(16, 4), constrained_layout=True)
        for a, (arr, title) in zip(ax, [(base.L, "L (limiting)"), (base.M, "M (modulating)"),
                                        (base.S, "S = L·G·M"), (np.where(cls.classes < 0, np.nan, cls.classes), "classes")]):
            im = a.imshow(arr, cmap="YlGn")
            a.set_title(title)
            a.set_axis_off()
            fig.colorbar(im, ax=a, shrink=0.7)
        fig.suptitle(f"SYNTHETIC TEST: {fam.nbs}/{fam.family}")
        fig.savefig(args.png, dpi=120)
        print(f"\nsaved {args.png}")
    print("\nStatus codes:", STATUS)


if __name__ == "__main__":
    main()
