"""
Plaintext Sep-CMA-ES aligned to the ciphertext C++ implementation in src/Server-SGFPT/cma_state.h
and src/Server-SGFPT/update.cu.

Key design choices matching the C++:
- Scalar global sigma (not per-dimension sigma_vec)
- Diagonal C stored as a d-vector (variances)
- p_sigma update uses z (raw N(0,I) noise) = C^{-1/2} y for diagonal C
- p_c update uses y_w (weighted step in y-space)
- C update: decay * C + c1 * p_c^2 + c_mu * sum_i w[i] * y_i^2
- Sigma update: sigma *= exp(b1 * ||p_sigma|| - b2)
  where b1 = c_sigma / (v_sigma * E_N), b2 = c_sigma / v_sigma
"""

import numpy as np


class SepCMAES:
    """Sep-CMA-ES matching src/Server-SGFPT/cma_state.h parameter formulas and src/Server-SGFPT/update.cu logic.

    Interface is compatible with shallow_cma: ask() returns candidate list,
    tell(solutions, fitnesses) updates the state.
    """

    def __init__(self, cfg):
        d = cfg["intrinsic_dim_L"] + cfg["intrinsic_dim_V"]
        lambda_ = int(cfg["popsize"])
        mu = lambda_ // 2
        seed = int(cfg.get("seed", 42))

        self.d = d
        self.lambda_ = lambda_
        self.mu = mu
        self.rng = np.random.default_rng(seed)

        # --- Recombination weights (same as cma_state.h init()) ---
        w_raw = np.array([np.log(mu + 0.5) - np.log(i + 1) for i in range(mu)])
        w_sum = w_raw.sum()
        w = w_raw / w_sum
        delta_eff = 1.0 / np.sum(w ** 2)
        self.w = w
        self.delta_eff = delta_eff

        # --- Step-size adaptation (CSA) ---
        c_sigma = (delta_eff + 2.0) / (d + delta_eff + 5.0)
        v_sigma = 1.0 + c_sigma
        E_N = np.sqrt(d) * (1.0 - 1.0 / (4.0 * d) + 1.0 / (21.0 * d * d))
        self.c_sigma1 = 1.0 - c_sigma              # decay factor for p_sigma
        self.c_sigma2 = np.sqrt(c_sigma * (2.0 - c_sigma) * delta_eff)
        self.b_sigma1 = c_sigma / (v_sigma * E_N)  # sigma *= exp(b1*||p_s|| - b2)
        self.b_sigma2 = c_sigma / v_sigma

        # --- Covariance path (p_c) ---
        c_c = (4.0 + delta_eff / d) / (d + 4.0 + 2.0 * delta_eff / d)
        self.c_c1 = 1.0 - c_c
        self.c_c2 = np.sqrt(c_c * (2.0 - c_c) * delta_eff)

        # --- Rank-1 and rank-mu learning rates ---
        c_1 = 2.0 / ((d + 1.3) ** 2 + delta_eff)
        c_mu = min(1.0 - c_1,
                   2.0 * (delta_eff - 2.0 + 1.0 / delta_eff) / ((d + 2.0) ** 2 + delta_eff))
        self.c_1 = c_1
        self.c_mu = c_mu
        self.c_decay = 1.0 - c_1 - c_mu  # C decay (= c in cma_state.h)

        # --- State ---
        self.m = np.zeros(d)
        self.sigma = float(cfg.get("sigma", 1.0))
        self.C = np.ones(d)       # diagonal covariance, initialised to I
        self.p_sigma = np.zeros(d)
        self.p_c = np.zeros(d)

        # Stash raw z and scaled y from the last ask() for use in tell()
        self._last_z = None
        self._last_y = None

        self._print_params(c_sigma, c_c, v_sigma, E_N)

    def _print_params(self, c_sigma, c_c, v_sigma, E_N):
        print(f"[SepCMAES] d={self.d}  lambda={self.lambda_}  mu={self.mu}")
        print(f"  delta_eff  = {self.delta_eff:.6f}")
        print(f"  c_sigma    = {c_sigma:.6f}   (step-size path lr)")
        print(f"  c_sigma2   = {self.c_sigma2:.6f}   (p_sigma noise coeff)")
        print(f"  b_sigma1   = {self.b_sigma1:.10f}")
        print(f"  b_sigma2   = {self.b_sigma2:.10f}")
        print(f"  v_sigma    = {v_sigma:.6f}   E_N = {E_N:.6f}")
        print(f"  c_c        = {c_c:.6f}   c_c2 = {self.c_c2:.6f}")
        print(f"  c_1        = {self.c_1:.6f}   c_mu = {self.c_mu:.6f}")
        print(f"  c_decay    = {self.c_decay:.6f}")

    # ------------------------------------------------------------------
    # Public interface (matches shallow_cma / pycma)
    # ------------------------------------------------------------------

    def ask(self):
        """Sample lambda_ candidate solutions.

        Returns a list of numpy arrays (same shape as pycma's ask()).
        Internally stores raw z and scaled y for the subsequent tell().
        """
        z = self.rng.standard_normal((self.lambda_, self.d))   # z_k ~ N(0, I)
        y = np.sqrt(self.C) * z                                # y_k = sqrt(C) * z_k
        x = self.m + self.sigma * y                            # x_k = m + sigma * y_k
        self._last_z = z
        self._last_y = y
        return list(x)   # list of 1-D arrays, same as pycma

    def tell(self, solutions, fitnesses):
        """Update state given evaluated candidates.

        solutions : list/array of shape (lambda_, d) — same as returned by ask()
        fitnesses : list/array of length lambda_    — losses to minimise
        """
        assert self._last_z is not None, "call ask() before tell()"

        fitnesses = np.asarray(fitnesses, dtype=np.float64)
        idx = np.argsort(fitnesses)                # ascending: lowest loss first
        z_mu = self._last_z[idx[: self.mu]]        # (mu, d) top-mu raw noise
        y_mu = self._last_y[idx[: self.mu]]        # (mu, d) top-mu scaled noise

        # Weighted mean steps
        y_w = self.w @ y_mu   # (d,)  weighted step in y-space
        z_w = self.w @ z_mu   # (d,)  = C^{-1/2} y_w for diagonal C

        # m-update
        self.m = self.m + self.sigma * y_w

        # p_sigma update: uses z_w (raw noise), same as C++ d_csigma2_w * Z_mu
        self.p_sigma = self.c_sigma1 * self.p_sigma + self.c_sigma2 * z_w

        # sigma-update: sigma *= exp(b1 * ||p_sigma|| - b2)
        ps_norm = float(np.linalg.norm(self.p_sigma))
        self.sigma *= np.exp(self.b_sigma1 * ps_norm - self.b_sigma2)

        # p_c update: uses y_w (scaled step), same as C++ c_c2 * d_yw
        self.p_c = self.c_c1 * self.p_c + self.c_c2 * y_w

        # C-update (diagonal): C = decay*C + c1*p_c^2 + c_mu*sum_i w[i]*y_i^2
        C_rankone = self.c_1 * (self.p_c ** 2)
        C_rankmu = self.c_mu * (self.w[:, None] * (y_mu ** 2)).sum(axis=0)
        self.C = self.c_decay * self.C + C_rankone + C_rankmu

        self._last_z = None
        self._last_y = None

    def stop(self):
        return {}   # no stopping criterion; caller controls loop

    @property
    def sigma_value(self):
        return self.sigma
