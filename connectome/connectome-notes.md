An artificial neural network (ANN) is a function built from many simple units arranged in layers. Each unit takes a weighted sum of its inputs, passes it through a nonlinearity such as ReLU or tanh, and sends the result onward. The weights are the parameters, and they are learned by gradient descent: compute an error, use backpropagation to find how each weight contributed to it, and nudge the weights. The idea comes from the McCulloch-Pitts neuron (1943) and Rosenblatt's perceptron (1958), but in practice an ANN is best seen as a trainable function approximator that borrowed a biological metaphor.

## What is "artificial" about it

- **The neuron is a single number.** A real neuron is an extended cell with dendrites that compute nonlinearly, many ion channel types, and spiking dynamics. Even in the worm, where many neurons use graded potentials instead of spikes, each neuron is a cell with its own dynamics. An ANN unit has no internal state.
- **The synapse is one real number.** Biological synapses are stochastic, change over time, and adapt on several timescales. Their sign depends on the receptor, not only the transmitter. We ran into this when assigning signs: glutamate can be excitatory or inhibitory in C. elegans, which is why my sign assignment was a simplification. There are also gap junctions, which are electrical connections with no counterpart in a standard ANN.
- **Time is discretized.** Layers compute in one synchronous sweep, while brains run continuously and asynchronously.
- **The architecture is designed.** Engineers choose dense layers, convolutions or attention, and the layout is regular. Brain wiring is set by genes, development and experience, and it is sparse, recurrent and heterogeneous.
- **Learning is different.** Backpropagation needs precise error signals and symmetric weights, and there is no established evidence the brain does exactly this. Brains use local plasticity rules and neuromodulators, and evolution acts as a long "pre-training" that sets the wiring. Whether the brain approximates backprop is still an open research question.
- **Chemical context is missing.** Neuropeptides and monoamines act outside the wired synapses. The `cect` package we used contains such extrasynaptic datasets alongside the synaptic one.

## Brain, connectome and ANN side by side

| | Biological brain | Connectome | ANN |
|---|---|---|---|
| What it is | The living, running system | A static wiring map of one or a few animals | A designed, trained function |
| Unit | Cell with dynamics | A named node | A scalar function |
| Connections | Dynamic, plastic synapses | Which neuron connects to which, plus contact strength | A dense matrix of learned weights |
| Sparsity | Very sparse | The worm's chemical network has about 4% of possible connections | Usually dense (or sparse by design) |
| Origin | Genes, development, experience | Same as the brain, since it is the brain's wiring | Chosen by engineers, weights set by gradient descent |
| Dynamics | Continuous, asynchronous | None; the diagram alone doesn't run | Discrete layers or steps |

## Where they genuinely overlap

Both are networks of simple nonlinear units where connection strength matters, and both can form distributed representations. Changing connection strength is also the central idea of learning in both.

## How this relates to what you just built

The connectome ESN sits in between. The topology is biological (sparse, recurrent, 302 specific neurons), but the neurons are still `tanh` units with one weight per connection, and the weights are not learned. It is an ANN built on a biological prior, still artificial in every item above. In my quick check on the sine task, the random reservoir did at least as well as the connectome, which fits a recurring lesson: biological wiring is not automatically a better computing structure on generic tasks. That check was on one validation segment, so the shuffled-wiring comparison we planned is the proper test of whether the specific wiring matters.

--
I'm not aware of a complete dragonfly connectome, so I'll check what exists now before answering.

The results show no dragonfly connectome, only fruit fly (*Drosophila*) work. I'll read the current fly ecosystem overview so I can say what is realistically usable.

No, and I couldn't find a dragonfly connectome. Searches turn up only fruit fly (*Drosophila*) wiring diagrams, so I'll assume you may have meant the fly. As far as I recall, dragonfly interception has been studied through recordings from target-tracking neurons and one hand-built interception circuit model (Chance, 2020), not through an EM wiring map, but please verify that.

If you meant the fruit fly, it's the same idea but not nearly as simple as the worm, mainly because of scale and data access.

## Scale

| Dataset | Region | Approx. neurons |
|---|---|---|
| C. elegans (what we used) | Whole nervous system | 302 |
| Hemibrain | Part of the central brain | ~25,000 |
| FlyWire (FAFB) | Whole brain | ~140,000 |
| Male CNS | Whole central nervous system | ~166,000 |

This breaks parts of my current code:

- **Dense matrices don't fit.** A 140,000 × 140,000 float64 matrix is about 157 GB. The reservoir would have to be a sparse matrix, which is fine for the state update.
- **Dense eigenvalue rescaling is out.** `np.linalg.eigvals` is impractical at this size; you'd use a sparse solver for just the largest eigenvalue (`scipy.sparse.linalg.eigs`).
- **The ridge readout over all neurons is out.** Its normal equations would need an n × n matrix. You'd read out from a subset (for example descending or output neurons, or a random few thousand).

## Data access

The worm came from a pip package, which I could install directly. The fly data is served through web platforms (Codex, neuPrint with a free account token, Coda) with Python tools such as `fafbseg-py`, `connecto` and `navis`, and the overview I read doesn't say what file formats they export. My sandbox can only reach package registries and GitHub, so you would need to download connection tables and attach them, or I'd need to find a package that bundles them.

## Biology and trust in the weights

- **Signs.** For the worm we got neurotransmitter identities from a curated atlas. For the fly I believe they are mostly predicted from EM images by a classifier (from memory, so check), so the sign problem is bigger.
- **Data quality.** Fly synapses are detected automatically and the wiring is crowd-proofread, so errors are more likely than in the hand-curated worm.
- **Same missing physiology.** Dynamics, plasticity and neuromodulation are still absent.

## A practical route

I'd start with a sub-circuit in the low thousands of neurons, such as the mushroom body (about 2,000 Kenyon cells per hemisphere). It's small enough to keep dense, it's a circuit people already model, and it doubles as a natural "expansion layer" test. A whole-brain fly reservoir is a bigger project: sparse code, subsampled readout, and the attached data.

If you want, I can adapt `connectome_esn.py` for a sparse reservoir and a subsampled readout, so it is ready when you have fly connection tables. I'd still finish the worm baselines first, since the shuffled control tells you whether wiring matters before you scale up.

Sources:
- [The Adult Drosophila Connectome Ecosystem](https://flyconnecto.me/2026/09/04/the-adult-drosophila-connectome-ecosystem/)
- [Neuronal wiring diagram of an adult brain (FlyWire)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10327113/)
- [Drosophila connectome (Wikipedia)](https://en.wikipedia.org/wiki/Drosophila_connectome)


---