"""Extract a neuron-only C. elegans connectome from the `cect` package.

Source: Cook et al. 2019 (hermaphrodite) for synapse counts, and the Wang et al.
2024 neurotransmitter atlas (both bundled with `cect`) for excitatory/inhibitory
signs.

Writes data/celegans_connectome.npz with:
  names       (N,)    neuron names
  cell_class  (N,)    'sensory' | 'inter' | 'motor' | 'other'
  W_chem      (N, N)  chemical synapse counts, indexed [post, pre]
  W_gap       (N, N)  gap-junction counts, indexed [post, pre] (symmetric)
  nt_sign     (N,)    +1 / -1 for each presynaptic neuron
  nt_label    (N,)    neurotransmitter(s) found for each neuron

Usage:  python prepare_connectome.py [--out data/celegans_connectome.npz]
"""
import argparse
import contextlib
import io
import os

import numpy as np


def load_datasets():
    """Load the two cect datasets; cect prints a lot, so silence it."""
    sink = io.StringIO()
    with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
        from cect.readers.Cook2019HermReader import get_instance as get_cook
        from cect.readers.Wang2024HermReader import get_instance as get_wang

        cook = get_cook()
        wang = get_wang()
    return cook, wang


def classify(name):
    from cect.Cells import (
        INTERNEURONS_NONPHARYNGEAL_COOK,
        MOTORNEURONS_NONPHARYNGEAL_COOK,
        SENSORY_NEURONS_NONPHARYNGEAL_COOK,
    )

    if name in SENSORY_NEURONS_NONPHARYNGEAL_COOK:
        return "sensory"
    if name in INTERNEURONS_NONPHARYNGEAL_COOK:
        return "inter"
    if name in MOTORNEURONS_NONPHARYNGEAL_COOK:
        return "motor"
    return "other"  # mostly pharyngeal neurons


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(here, "data", "celegans_connectome.npz"))
    args = ap.parse_args()

    cook, wang = load_datasets()
    from cect.Cells import is_any_neuron

    # Neurons only (drop muscles, gonad, and other non-neuronal cells).
    keep = [i for i, n in enumerate(cook.nodes) if is_any_neuron(n)]
    names = [cook.nodes[i] for i in keep]
    N = len(names)
    ix = np.ix_(keep, keep)

    # cect stores connections as [pre, post]; the ESN wants [post, pre].
    W_chem = np.asarray(cook.connections["Generic_CS"])[ix].T.copy()
    W_gap = np.asarray(cook.connections["Generic_GJ"])[ix].T.copy()

    # Sanity check our indexing against the package's own accessor.
    rng = np.random.default_rng(0)
    for _ in range(200):
        a, b = rng.integers(0, N, size=2)
        expected = cook.get_connection_weight(names[a], names[b], "Generic_CS")
        assert W_chem[b, a] == expected, (names[a], names[b])

    # Signs from the neurotransmitter atlas: a neuron whose only fast transmitter
    # is GABA is inhibitory; everything else (ACh, glutamate, mixed, unknown) is
    # treated as excitatory. Glutamate can be inhibitory in C. elegans (GluCl
    # receptors), so this is a simplification.
    wang_idx = {n: i for i, n in enumerate(wang.nodes)}
    nt_mats = {k: np.asarray(wang.connections[k]) for k in ("Acetylcholine", "Glutamate", "GABA")}
    nt_label, nt_sign = [], []
    for n in names:
        found = []
        if n in wang_idx:
            r = wang_idx[n]
            found = [k for k, m in nt_mats.items() if m[r].sum() > 0]
        nt_label.append("+".join(found) if found else "unknown")
        nt_sign.append(-1 if found == ["GABA"] else 1)
    nt_sign = np.array(nt_sign)

    cell_class = np.array([classify(n) for n in names])

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez_compressed(
        args.out,
        names=np.array(names),
        cell_class=cell_class,
        W_chem=W_chem,
        W_gap=W_gap,
        nt_sign=nt_sign,
        nt_label=np.array(nt_label),
    )

    # Summary
    print(f"neurons: {N}")
    for c in ("sensory", "inter", "motor", "other"):
        print(f"  {c:8s} {int((cell_class == c).sum())}")
    print(f"chemical: {int((W_chem > 0).sum())} connections, {int(W_chem.sum())} synapses")
    print(f"gap:      {int((W_gap > 0).sum())} connections, symmetric error {np.abs(W_gap - W_gap.T).max():.3g}")
    print(f"inhibitory presynaptic neurons (GABA-only): {int((nt_sign < 0).sum())}")
    print(f"neurons with no neurotransmitter info: {sum(l == 'unknown' for l in nt_label)}")
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
