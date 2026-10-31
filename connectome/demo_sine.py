"""Smoke test: can the connectome ESN reconstruct a sine wave?

Setup (differs from the GeeksforGeeks example, which feeds noise in and fits the
training set): the input is the sine itself, the target is the next sample, and
scores come from held-out segments that come after the training segment.

  one-step : predict s[t+1] from s[..t] on a held-out segment
  free-run : teacher-force on a context, then let the network feed its own
             predictions back and generate the continuation

The ridge penalty is chosen on a validation segment, using input-weight seeds
that are different from the seeds reported on the test segment.

Usage:  python demo_sine.py [--seeds 5] [--plot sine_demo.png]
"""
import argparse

import numpy as np

from connectome_esn import (
    ConnectomeESN,
    build_weight_matrix,
    load_connectome,
    neuron_indices,
    nrmse,
)

DT = 0.1                              # ~63 samples per period
N_WASHOUT, N_TRAIN = 200, 2000
ONE_STEP = (2000, 3000)               # held-out one-step segment, right after training
VAL_START, TEST_START = 2500, 3200    # where free-running generation begins
CONTEXT, HORIZONS = 800, (100, 300)
RIDGES = (1e-8, 1e-6, 1e-4, 1e-2, 1.0)
SELECT_SEEDS = range(100, 105)        # never reported; used only to pick the ridge

S = np.sin(np.arange(4000) * DT)

CONFIGS = [
    ("chem, all-excitatory", "excitatory", 0.0),
    ("chem, NT signs", "neurotransmitter", 0.0),
    ("chem+gap, NT signs", "neurotransmitter", 1.0),
]


def fit_esn(conn, signs, gap_scale, seed, ridge, rho=0.9, input_scaling=1.0):
    W = build_weight_matrix(conn, signs=signs, gap_scale=gap_scale)
    esn = ConnectomeESN(W, input_idx=neuron_indices(conn, "sensory"), spectral_radius=rho,
                        input_scaling=input_scaling, seed=seed)
    return esn.fit(S[:N_TRAIN], S[1:N_TRAIN + 1], washout=N_WASHOUT, ridge=ridge)


def free_run_nrmse(esn, start, horizon):
    gen = esn.free_run(S[start - CONTEXT:start], horizon)
    return nrmse(S[start:start + horizon], gen)


def pick_ridge(conn, signs, gap_scale):
    scores = {
        r: float(np.mean([free_run_nrmse(fit_esn(conn, signs, gap_scale, sd, r), VAL_START, HORIZONS[0])
                          for sd in SELECT_SEEDS]))
        for r in RIDGES
    }
    best = min(scores, key=scores.get)
    return best, scores[best]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--plot", default=None, help="optional PNG of true vs predicted")
    args = ap.parse_args()

    conn = load_connectome()
    a, b = ONE_STEP
    cols = ["one_step"] + [f"free_{h}" for h in HORIZONS]
    print(f"Test NRMSE (lower is better; 1.0 = no better than the mean), "
          f"mean ± std over {args.seeds} input-weight seeds\n")
    print(f"{'config':22s}{'ridge':>8s}{'val_free100':>13s}" + "".join(f"{c:>20s}" for c in cols))

    first = None
    for label, signs, gap in CONFIGS:
        ridge, val = pick_ridge(conn, signs, gap)
        rows = []
        for seed in range(args.seeds):
            esn = fit_esn(conn, signs, gap, seed, ridge)
            one = esn.predict(S[a:b], x0=esn.final_state)
            res = {"one_step": nrmse(S[a + 1:b + 1], one)}
            for h in HORIZONS:
                res[f"free_{h}"] = free_run_nrmse(esn, TEST_START, h)
            rows.append(res)
            if first is None:
                first = (S, one, esn.free_run(S[TEST_START - CONTEXT:TEST_START], max(HORIZONS)))
        line = f"{label:22s}{ridge:>8g}{val:>13.3g}"
        for c in cols:
            v = np.array([r[c] for r in rows])
            line += f"{v.mean():>12.4g} ± {v.std():<5.2g}"
        print(line)

    if args.plot and first is not None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        s, one, gen = first
        fig, ax = plt.subplots(2, 1, figsize=(10, 6))
        ax[0].plot(s[a + 1:a + 301], "--", label="true")
        ax[0].plot(one[:300, 0], label="one-step")
        ax[0].set_title("one-step prediction (held-out)")
        ax[1].plot(s[TEST_START:TEST_START + 300], "--", label="true")
        ax[1].plot(gen[:300, 0], label="free-run")
        ax[1].set_title("free-running generation (first config, seed 0)")
        for x in ax:
            x.legend()
        fig.tight_layout()
        fig.savefig(args.plot, dpi=120)
        print(f"plot -> {args.plot}")


if __name__ == "__main__":
    main()
