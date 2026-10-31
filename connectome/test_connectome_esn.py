"""Structural checks for the connectome ESN. Run:  python test_connectome_esn.py
(also works under pytest)."""
import numpy as np

from connectome_esn import (
    ConnectomeESN,
    build_weight_matrix,
    load_connectome,
    max_abs_eig,
    neuron_indices,
)

conn = load_connectome()


def test_connectome_basics():
    assert len(conn["names"]) == 302
    assert set(conn["cell_class"]) == {"sensory", "inter", "motor", "other"}
    assert np.abs(conn["W_gap"] - conn["W_gap"].T).max() == 0  # gap junctions are symmetric
    assert conn["W_chem"].min() >= 0


def test_orientation_is_post_by_pre():
    # Sensory neurons should send more than they receive, motor neurons the reverse
    # (within the neuron-only graph). A transposed matrix would flip both.
    W = conn["W_chem"]
    out_w, in_w = W.sum(axis=0), W.sum(axis=1)  # column = presynaptic, row = postsynaptic
    sens = conn["cell_class"] == "sensory"
    motor = conn["cell_class"] == "motor"
    assert out_w[sens].sum() > in_w[sens].sum()
    assert in_w[motor].sum() > out_w[motor].sum()


def test_signs_only_flip_inhibitory_columns():
    Wx = build_weight_matrix(conn, signs="excitatory")
    Wn = build_weight_matrix(conn, signs="neurotransmitter")
    assert (Wx >= 0).all()
    neg_cols = np.flatnonzero((Wn < 0).any(axis=0))
    assert set(neg_cols) <= set(np.flatnonzero(conn["nt_sign"] < 0))
    assert np.array_equal(np.abs(Wn), Wx)


def test_spectral_radius_is_rescaled():
    sens = neuron_indices(conn, "sensory")
    for signs in ("excitatory", "neurotransmitter"):
        for gap in (0.0, 1.0):
            W = build_weight_matrix(conn, signs=signs, gap_scale=gap)
            esn = ConnectomeESN(W, input_idx=sens, spectral_radius=0.9)
            assert abs(max_abs_eig(esn.W) - 0.9) < 1e-9, (signs, gap)


def test_input_only_reaches_chosen_neurons():
    sens = neuron_indices(conn, "sensory")
    esn = ConnectomeESN(build_weight_matrix(conn), input_idx=sens)
    assert set(np.flatnonzero(esn.W_in[:, 0])) == set(sens)


def test_seed_determinism():
    sens = neuron_indices(conn, "sensory")
    W = build_weight_matrix(conn)
    u = np.sin(np.arange(50) * 0.1)
    a = ConnectomeESN(W, sens, seed=1).run(u)
    b = ConnectomeESN(W, sens, seed=1).run(u)
    c = ConnectomeESN(W, sens, seed=2).run(u)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_predict_continues_from_training_state():
    sens = neuron_indices(conn, "sensory")
    esn = ConnectomeESN(build_weight_matrix(conn), sens)
    s = np.sin(np.arange(600) * 0.1)
    esn.fit(s[:400], s[1:401], washout=50)
    # Running train+test in one go must equal training run then continuing from its state.
    whole = esn.run(s[:600])[400:]
    cont = esn.run(s[400:600], x0=esn.final_state)
    assert np.allclose(whole, cont)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print("ok  ", t.__name__)
    print(f"{len(tests)} passed")
