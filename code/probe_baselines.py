"""
Ambient baselines fed the same probes as SPSC (sensing control).

ProbeLinUCB plays, on exactly the rounds where SPSC_Algorithm1 probes, the same scaled-sphere probe u_t,
observes the same y_t, is charged the same forgone reward plus the fee c, and adds (u_t, y_t) to its
ambient d-dimensional ridge.  On every other round it is LinUCB (window=None) or SW-LinUCB (window=W),
with the same confidence radius as those baselines.  With the same seed as SPSC_Algorithm1 it draws the
same action sets and the same probe directions, so the comparison is paired round by round.

reset_at_boundaries=True  : resets at the true change points and probes on SPSC_Algorithm1's schedule
                            (first round of each segment, then every probe_every rounds).
reset_at_boundaries=False : never resets and probes every probe_every rounds from t = 0, the control
                            for SPSC-Adaptive.
"""
import numpy as np

from algorithm import RunMetrics


class ProbeLinUCB:
    def __init__(self, env, probe_every=50, probe_cost=0.1, window=None, lam=1.0, delta=0.05, seed=0,
                 reset_at_boundaries=True):
        self.env = env
        self.probe_every = probe_every
        self.c = probe_cost
        self.W = window
        self.lam = lam
        self.delta = delta
        self.rng = np.random.default_rng(seed)
        self.reset = reset_at_boundaries
        self.S = env.S
        self.sigma_eps = env.sigma_eps
        self.L_x = env.L_x
        self.probe_log = []          # (t, u_t) for the identity check against SPSC

    def _beta(self, n):
        # same radius as LinUCB / SW-LinUCB in algorithm.py
        d = self.env.d
        arg = max(1.0 + n * self.L_x ** 2 / self.lam, 1.0 + 1e-12)
        return self.sigma_eps * np.sqrt(d * np.log(arg / self.delta)) + np.sqrt(self.lam) * self.S

    def _is_probe(self, t, k):
        if self.reset:
            seg_start = self.env.tau[k]
            return t == seg_start or ((t - seg_start) % self.probe_every) == 0
        return (t % self.probe_every) == 0

    def run(self):
        env = self.env
        d, T = env.d, env.T
        name = ("Probe-SW-LinUCB" if self.W else "Probe-LinUCB") + ("" if self.reset else " (no reset)")
        metrics = RunMetrics(name=name, T=T)
        V = self.lam * np.eye(d)
        b = np.zeros(d)
        buf = []                     # (x, y, s) for the sliding-window version, probes included
        current_k = -1
        for t in range(T):
            k = env.seg_of[t]
            if self.reset and k != current_k:
                V = self.lam * np.eye(d)
                b = np.zeros(d)
                buf = []
                current_k = k
            action_set = env.get_action_set(t, rng=self.rng)
            r_opt = env.optimal_reward(action_set, t)
            if self.W:
                win = [(xs, ys) for xs, ys, s in buf if s >= t - self.W]
                V = self.lam * np.eye(d)
                b = np.zeros(d)
                for xs, ys in win:
                    V += np.outer(xs, xs)
                    b += xs * ys
                n_obs = len(win)
            else:
                n_obs = t
            if self._is_probe(t, k):
                metrics.probe_flags[t] = True
                z = self.rng.standard_normal(d)
                u_t = np.sqrt(d) * z / (np.linalg.norm(z) + 1e-12)
                y = env.step(u_t, t)
                self.probe_log.append((t, u_t))
                x_dep, cost = u_t, self.c
            else:
                theta_hat = np.linalg.solve(V, b)
                beta_t = self._beta(n_obs)
                V_inv_A = np.linalg.solve(V, action_set.T).T
                ucb = action_set @ theta_hat + beta_t * np.sqrt(np.einsum('ij,ij->i', action_set, V_inv_A))
                x_dep = action_set[int(np.argmax(ucb))]
                y = env.step(x_dep, t)
                cost = 0.0
            if self.W:
                buf.append((x_dep, y, t))
                if len(buf) > self.W + 10:
                    buf = [(xs, ys, s) for xs, ys, s in buf if s >= t - self.W]
            else:
                V += np.outer(x_dep, x_dep)
                b += x_dep * y
            r_t = float(x_dep @ env.theta[t])
            metrics.control_regret[t] = r_opt - r_t
            metrics.costed_regret[t] = r_opt - r_t + cost
        return metrics
