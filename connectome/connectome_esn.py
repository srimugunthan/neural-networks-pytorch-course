"""Echo state network (ESN) whose recurrent weights are the C. elegans connectome.

Only the linear readout is trained. The connectome never changes; it is only
rescaled once so its largest eigenvalue magnitude equals `spectral_radius`.

    conn = load_connectome()
    W = build_weight_matrix(conn, signs="neurotransmitter")
    esn = ConnectomeESN(W, input_idx=neuron_indices(conn, "sensory"), spectral_radius=0.9)
    esn.fit(U_train, Y_train, washout=200, ridge=1e-6)
    Y_hat = esn.predict(U_test, x0=esn.final_state)

State update:  x[t] = (1 - leak) * x[t-1] + leak * tanh(W x[t-1] + W_in u[t] + b)
Readout:       y[t] = W_out . [x[t][readout_idx], 1]
Row i of W is the set of inputs to neuron i, i.e. W is indexed [post, pre].
"""
import os

import numpy as np

DEFAULT_DATA = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "data", "celegans_connectome.npz"
)


# --------------------------------------------------------------------------
# Connectome loading and weight-matrix construction
# --------------------------------------------------------------------------
def load_connectome(path=DEFAULT_DATA):
    with np.load(path, allow_pickle=False) as d:
        return {k: d[k] for k in d.files}


def neuron_indices(conn, which):
    """Indices for 'all', a cell class ('sensory', 'inter', 'motor', 'other'),
    or an explicit list of neuron names."""
    names = list(conn["names"])
    if which == "all":
        return np.arange(len(names))
    if isinstance(which, str):
        idx = np.flatnonzero(conn["cell_class"] == which)
        if idx.size == 0:
            raise ValueError(f"no neurons in class {which!r}")
        return idx
    lookup = {n: i for i, n in enumerate(names)}
    return np.array([lookup[n] for n in which])


def build_weight_matrix(conn, signs="neurotransmitter", gap_scale=0.0, transform="raw"):
    """Turn the stored connectome into a recurrent weight matrix W[post, pre].

    signs:      'excitatory' (all weights positive) or 'neurotransmitter'
                (GABA-only presynaptic neurons are inhibitory, the rest excitatory).
    gap_scale:  0 = chemical synapses only; >0 adds gap junctions (symmetric,
                positive) multiplied by this factor.
    transform:  'raw' (EM-section weights), 'log1p', or 'binary'.
    """
    def f(A):
        A = A.astype(float)
        if transform == "raw":
            return A
        if transform == "log1p":
            return np.log1p(A)
        if transform == "binary":
            return (A > 0).astype(float)
        raise ValueError(f"unknown transform {transform!r}")

    W = f(conn["W_chem"])
    if signs == "neurotransmitter":
        W = W * conn["nt_sign"][None, :]  # sign belongs to the presynaptic neuron (column)
    elif signs != "excitatory":
        raise ValueError(f"unknown signs {signs!r}")
    if gap_scale:
        W = W + gap_scale * f(conn["W_gap"])
    return W


def max_abs_eig(W):
    return float(np.max(np.abs(np.linalg.eigvals(W))))


# --------------------------------------------------------------------------
# The network
# --------------------------------------------------------------------------
def _as_2d(a):
    a = np.asarray(a, dtype=float)
    return a.reshape(-1, 1) if a.ndim == 1 else a


class ConnectomeESN:
    def __init__(self, W, input_idx, readout_idx=None, spectral_radius=0.9,
                 input_scaling=1.0, bias_scaling=0.0, leak=1.0, input_dim=1, seed=0):
        W = np.asarray(W, dtype=float)
        n = W.shape[0]
        if W.shape != (n, n):
            raise ValueError("W must be square")
        rho = max_abs_eig(W)
        if rho < 1e-9:
            raise ValueError("W has spectral radius ~0 (no recurrence); cannot rescale")
        self.n = n
        self.raw_spectral_radius = rho
        self.W = W * (spectral_radius / rho)

        rng = np.random.default_rng(seed)
        input_idx = np.asarray(input_idx)
        self.W_in = np.zeros((n, input_dim))
        self.W_in[input_idx] = rng.uniform(-1, 1, (len(input_idx), input_dim)) * input_scaling
        self.b = rng.uniform(-1, 1, n) * bias_scaling
        self.readout_idx = np.arange(n) if readout_idx is None else np.asarray(readout_idx)
        self.leak = leak
        self.W_out = None
        self.final_state = None

    # -- dynamics ----------------------------------------------------------
    def _step(self, x, u):
        pre = self.W @ x + self.W_in @ u + self.b
        return (1.0 - self.leak) * x + self.leak * np.tanh(pre)

    def run(self, U, x0=None):
        """Drive the reservoir with inputs U (T, d); returns states (T, n).
        states[t] is the state after seeing U[t]."""
        U = _as_2d(U)
        x = np.zeros(self.n) if x0 is None else np.array(x0, dtype=float)
        X = np.empty((len(U), self.n))
        for t, u in enumerate(U):
            x = self._step(x, u)
            X[t] = x
        return X

    def _features(self, X):
        return np.hstack([X[:, self.readout_idx], np.ones((len(X), 1))])

    # -- training and prediction -------------------------------------------
    def fit(self, U, Y, washout=100, ridge=1e-6):
        """Ridge regression from states[t] to Y[t]; the first `washout` steps are skipped."""
        U, Y = _as_2d(U), _as_2d(Y)
        if len(U) != len(Y):
            raise ValueError("U and Y must have the same length")
        X = self.run(U)
        self.final_state = X[-1].copy()
        Z = self._features(X)[washout:]
        reg = ridge * np.eye(Z.shape[1])
        reg[-1, -1] = 0.0  # do not penalise the bias term
        A, B = Z.T @ Z + reg, Z.T @ Y[washout:]
        try:
            self.W_out = np.linalg.solve(A, B)
        except np.linalg.LinAlgError:
            self.W_out = np.linalg.lstsq(A, B, rcond=None)[0]
        return self

    def _check_fitted(self):
        if self.W_out is None:
            raise RuntimeError("call fit() first")

    def predict(self, U, x0=None):
        """Outputs for inputs U. Pass x0=esn.final_state to continue straight after training."""
        self._check_fitted()
        return self._features(self.run(U, x0)) @ self.W_out

    def free_run(self, context, n_steps, x0=None):
        """Teacher-force on `context`, then feed the network's own predictions back
        as its next input for `n_steps` steps (needs output dim == input dim)."""
        self._check_fitted()
        if self.W_out.shape[1] != self.W_in.shape[1]:
            raise ValueError("free_run needs output dim == input dim")
        x = self.run(context, x0)[-1]
        out = np.empty((n_steps, self.W_out.shape[1]))
        u = (self._features(x[None, :]) @ self.W_out)[0]
        for k in range(n_steps):
            out[k] = u
            x = self._step(x, u)
            u = (self._features(x[None, :]) @ self.W_out)[0]
        return out


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def nrmse(y_true, y_pred):
    """Normalised RMSE: 1.0 means no better than predicting the mean."""
    y_true, y_pred = _as_2d(y_true), _as_2d(y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2) / np.var(y_true)))


def effective_rank(X):
    """Participation ratio of the state covariance spectrum: roughly how many
    independent directions the reservoir states actually use."""
    lam = np.linalg.svd(X - X.mean(axis=0), compute_uv=False) ** 2
    return float(lam.sum() ** 2 / (lam ** 2).sum())
