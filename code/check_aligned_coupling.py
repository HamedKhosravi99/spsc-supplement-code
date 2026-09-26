"""Sensitivity of the lifted subspace estimator to a probe-dependent coupling (Table 8 of the paper).

Part 1  Probe-dependent coupling sweep, eps -> eps + xi*(u^T theta), xi in [0,1].
        The observation law is the model of Sec. 2 with theta rescaled, so the quadratic identity
        applies to theta' = (1+xi)theta and E[G_t | H_{t-1}] = (1+xi)^2 Mtilde + Btilde.
        A positive rescaling shifts eigenvalues without rotating eigenvectors, so the
        top-r projector should be unaffected across the sweep.

Part 2  General first-order case: if the odd part of m_t(u) is linear, m^odd(u) = M u,
        then E[(u^T M u) u u^T] = K(sym M), so the bias of the lifted estimator is
        exactly 2*sym(M) -- dimension-free -- and Davis-Kahan gives
        ||Phat - P*|| <= 8||sym M||_op / lambda_min + sampling error.

Both are exact consequences of K^{-1} o K = id; the Monte Carlo below is a check for
sign and constant slips. Fixed seed, ~2 min. Residuals track sqrt(d/n).
"""
import numpy as np

rng = np.random.default_rng(0)
CHUNK = 100_000


def Kinv(N, d):
    return (d + 2) / (2 * d) * N - np.trace(N) / (2 * d) * np.eye(d)


def sphere(d, n, rng):
    g = rng.standard_normal((n, d))
    return np.sqrt(d) * g / np.linalg.norm(g, axis=1, keepdims=True)


def topr(M, r):
    V = np.linalg.eigh(M)[1][:, -r:]
    return V @ V.T


print("=" * 86)
print("PART 1  probe-dependent coupling sweep:  y = (1+xi) u^T theta + eps")
print("        i.e. eps -> eps + xi*(u^T theta), a coupling that depends on the probe direction")
print("        prediction: E[Mhat] = (1+xi)^2 * Mtilde  =>  top-r eigenspace UNCHANGED")
print("=" * 86)
N_TOT = 800_000
for d, r in [(10, 2), (30, 3)]:
    B = np.linalg.qr(rng.standard_normal((d, r)))[0]
    Sig = np.diag(np.linspace(1.0, 3.0, r))
    Mtilde = B @ Sig @ B.T
    P0 = topr(Mtilde, r)
    for xi in [0.0, 0.5, 1.0]:
        acc = np.zeros((d, d))
        done = 0
        while done < N_TOT:
            n = min(CHUNK, N_TOT - done)
            U = sphere(d, n, rng)
            Th = (rng.standard_normal((n, r)) @ np.sqrt(Sig)) @ B.T
            proj = np.einsum("ij,ij->i", U, Th)
            y = (1 + xi) * proj + 0.5 * rng.standard_normal(n)
            s = y**2 - 0.25
            acc += (U * s[:, None]).T @ U
            done += n
        Mhat = Kinv(acc / N_TOT, d)
        pred = (1 + xi) ** 2 * Mtilde
        rel = np.linalg.norm(Mhat - pred, 2) / np.linalg.norm(pred, 2)
        dP = np.linalg.norm(topr(Mhat, r) - P0, 2)
        print(f"  d={d:3d} r={r} xi={xi:4.2f} |  ||Mhat-(1+xi)^2*Mtilde||/||.|| = {rel:.4f}"
              f"  |  ||Phat-P*||_op = {dP:.4f}")

print()
print("=" * 86)
print("PART 2  general first-order case: odd part of m_t(u) linear, m^odd(u) = M u")
print("        prediction: bias of E[Mhat] = 2*sym(M) EXACTLY -- no factor of d")
print("=" * 86)
for d, N_TOT in [(5, 1_500_000), (20, 1_500_000), (60, 1_500_000)]:
    M = rng.standard_normal((d, d)) / np.sqrt(d)
    Msym = (M + M.T) / 2
    acc = np.zeros((d, d))
    done = 0
    while done < N_TOT:
        n = min(CHUNK, N_TOT - done)
        U = sphere(d, n, rng)
        phi = np.einsum("ij,ij->i", U @ M, U)
        acc += (U * phi[:, None]).T @ U
        done += n
    bias = 2 * Kinv(acc / N_TOT, d)
    rel = np.linalg.norm(bias - 2 * Msym, 2) / np.linalg.norm(2 * Msym, 2)
    print(f"  d={d:3d} n={N_TOT:9d} |  ||bias - 2*sym(M)||_op / ||2*sym(M)||_op = {rel:.4f}")
