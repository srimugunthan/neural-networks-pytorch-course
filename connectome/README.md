# Connectome Echo State Network

An echo state network (ESN) whose recurrent reservoir is the real **C. elegans connectome** (302 neurons, Cook et al. 2019). The connectome weights are never trained; only a linear (ridge) readout is fit. A sine-wave demo checks that the network works.

## What has been implemented

| File | Purpose |
|---|---|
| `prepare_connectome.py` | Extracts a neuron-only connectome from the `cect` package and saves it to `data/celegans_connectome.npz`. |
| `connectome_esn.py` | The ESN library: weight-matrix construction, the `ConnectomeESN` class, and metrics. |
| `demo_sine.py` | Sine-wave reconstruction demo with a held-out test and validation-based ridge selection. |
| `test_connectome_esn.py` | Structural tests (orientation, scaling, input placement, determinism, state continuity). |
| `data/celegans_connectome.npz` | The extracted data (already included, so you can skip the prepare step). |

### 1. Connectome data (`prepare_connectome.py`)

- Loads the **Cook et al. 2019 hermaphrodite** connectome through the [OpenWorm Connectome Toolbox](https://github.com/openworm/ConnectomeToolbox) (`cect` on PyPI) and keeps the **302 neurons** (83 sensory, 81 inter, 116 motor, 22 other, mostly pharyngeal). Muscles and other non-neuronal cells are dropped.
- Stores chemical-synapse weights and gap-junction weights, both indexed `[post, pre]`. Cook's weights are the **number of EM serial sections of connectivity**, reflecting both the number and the size of synapses. They are not raw synapse counts.
- Assigns each presynaptic neuron a sign using the **Wang et al. 2024 neurotransmitter atlas** (also bundled in `cect`): neurons whose only fast transmitter is GABA are inhibitory (31 neurons), everything else is excitatory.
- Checks its own indexing against the package's accessor on 200 random neuron pairs.

### 2. The ESN (`connectome_esn.py`)

State update and readout:

```
x[t] = (1 - leak) * x[t-1] + leak * tanh(W x[t-1] + W_in u[t] + b)
y[t] = W_out . [x[t][readout_idx], 1]
```

- `build_weight_matrix(conn, signs, gap_scale, transform)` builds `W`:
  - `signs`: `"excitatory"` (all positive) or `"neurotransmitter"` (GABA-only neurons inhibitory).
  - `gap_scale`: `0` for chemical synapses only, `>0` to add gap junctions scaled by that factor.
  - `transform`: `"raw"`, `"log1p"` or `"binary"` weights.
- `ConnectomeESN(W, input_idx, ...)` rescales `W` once so its spectral radius equals `spectral_radius`. Inputs enter only through the neurons in `input_idx` (the demo uses the sensory neurons). The readout can use all neurons or a chosen subset (`readout_idx`).
- `fit()` trains the readout by ridge regression with a washout period (the bias term is not penalized). `predict()` gives outputs, and `free_run()` generates autonomously by feeding the network's own predictions back as input.
- `neuron_indices(conn, "sensory" | "inter" | "motor" | "other" | "all" | [names])` selects neurons by class or name.
- `nrmse()` is the metric (1.0 means no better than predicting the mean). `effective_rank()` is a diagnostic of how many independent directions the states use.

### 3. Sine demo (`demo_sine.py`)

The input is a sine wave and the target is the next sample. The network is trained on the first 2000 samples and scored on held-out segments after that. Two tasks:

- **One-step prediction** on a held-out segment.
- **Free-running generation**: the network is driven by a context, then feeds its own predictions back to generate the continuation.

The ridge penalty is chosen on a separate validation segment using input-weight seeds that are never reported. Three connectome variants are compared.

Results from one run (mean ± std over 5 input-weight seeds, test NRMSE, lower is better):

| Config | Ridge | One-step | Free-run, 100 steps | Free-run, 300 steps |
|---|---|---|---|---|
| chemical, all-excitatory | 0.01 | 0.0016 | 0.028 ± 0.044 | 0.19 ± 0.36 |
| chemical, neurotransmitter signs | 0.01 | 0.0016 | 0.080 ± 0.15 | 0.27 ± 0.53 |
| chemical + gap junctions | 0.01 | 0.0042 | 0.034 ± 0.006 | 0.093 ± 0.018 |

Read these as a smoke test, not a study:

- A sine is an easy task, so everything scores well.
- Free-running generation is **seed-dependent** for the chemical-only variants: some input-weight seeds generate the sine almost perfectly and others fail, which is why the standard deviations are large.
- With only 5 seeds, the differences between the three variants are within noise.
- At a very small ridge (1e-6), free-running generation failed for most seeds. Regularization matters here.

## Steps to run

Tested with Python 3.13.

### 1. Install dependencies

```bash
pip install numpy matplotlib
```

`matplotlib` is only needed for the optional plot. If you want to regenerate the data yourself (step 3), also install `cect`; this pulls in many dependencies and can take several minutes:

```bash
pip install cect
```

### 2. Run the tests

```bash
python test_connectome_esn.py
```

Expected: 7 checks reported as `ok`, then `7 passed`.

### 3. (Optional) Regenerate the connectome data

The included `data/celegans_connectome.npz` is enough to run everything. To rebuild it from the source datasets:

```bash
python prepare_connectome.py
```

This prints the neuron counts per class and the number of connections, then saves the `.npz` file.

### 4. Run the sine demo

```bash
python demo_sine.py --seeds 5
```

It takes under a minute on a laptop. Add `--plot sine_demo.png` to also save a figure of true vs predicted values.

### 5. Use the ESN in your own code

```python
import numpy as np
from connectome_esn import (ConnectomeESN, build_weight_matrix, load_connectome,
                            neuron_indices, nrmse)

conn = load_connectome()
W = build_weight_matrix(conn, signs="neurotransmitter", gap_scale=0.0)
esn = ConnectomeESN(W, input_idx=neuron_indices(conn, "sensory"),
                    spectral_radius=0.9, input_scaling=1.0, seed=0)

s = np.sin(np.arange(4000) * 0.1)
esn.fit(s[:2000], s[1:2001], washout=200, ridge=1e-2)            # train: input -> next sample
pred = esn.predict(s[2000:3000], x0=esn.final_state)             # held-out one-step prediction
print(nrmse(s[2001:3001], pred))

gen = esn.free_run(s[2400:3200], 100)                            # autonomous generation
print(nrmse(s[3200:3300], gen))
```

## Design choices and limitations

- **Signs are a simplification.** Only GABA-only neurons are inhibitory. Glutamate is treated as excitatory, although it can act as inhibitory in C. elegans (GluCl receptors). Neurons with no neurotransmitter information are treated as excitatory.
- **Gap junctions** are symmetric and added as positive weights, multiplied by `gap_scale`.
- **Real C. elegans neurons** mostly show graded potentials rather than classic spikes, so a rate-based `tanh` reservoir is a reasonable but simplified model.
- **The connectome gives wiring only.** Neuron dynamics, synaptic strengths in physiological units, neuromodulation and plasticity are not part of it. Results depend on these modeling choices (signs, weight transform, spectral radius, input neurons).
- **A sine wave is periodic**, so held-out segments resemble the training data. It is a sanity check, not a benchmark of generalization.

## References

- Cook, S. J., Jarrell, T. A., Brittin, C. A., et al. **Whole-animal connectomes of both *Caenorhabditis elegans* sexes.** *Nature* 571, 63-71 (2019). https://doi.org/10.1038/s41586-019-1352-7 (source of the synapse and gap-junction weights)
- Wang, C., Vidal, B., Sural, S., et al. **A neurotransmitter atlas of *C. elegans* males and hermaphrodites.** *eLife* (2024), reviewed preprint 95402. https://doi.org/10.7554/eLife.95402 (source of the neurotransmitter identities used for signs)
- Jaeger, H. **The "echo state" approach to analysing and training recurrent neural networks.** GMD Report 148, German National Research Center for Information Technology (2001). (the original echo state network report)
- OpenWorm **Connectome Toolbox** (`cect`): https://github.com/openworm/ConnectomeToolbox (the package used to load both datasets)

Please verify the author lists and citation details against the publisher pages before citing them in other work.
