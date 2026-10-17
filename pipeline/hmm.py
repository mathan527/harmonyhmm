"""Unit 4 algorithms implemented from scratch in NumPy.

- scaled forward / backward (Rabiner scaling factors)  -> log-likelihood without underflow
- Baum-Welch (EM for HMMs) on many sequences, discrete emissions
- Viterbi in log space
- Kalman filter + RTS smoother for a linear dynamical system
"""
import numpy as np


def forward_unscaled(pi, A, B, obs):
    """Plain forward recursion. Underflows to 0 on long sequences (kept to demonstrate why scaling is needed)."""
    alpha = pi * B[:, obs[0]]
    out = [alpha.sum()]
    for o in obs[1:]:
        alpha = (alpha @ A) * B[:, o]
        out.append(alpha.sum())
    return np.array(out)


def forward_scaled(pi, A, B, obs):
    T, N = len(obs), len(pi)
    ah = np.zeros((T, N))
    c = np.zeros(T)
    a = pi * B[:, obs[0]]
    c[0] = a.sum()
    ah[0] = a / c[0]
    for t in range(1, T):
        a = (ah[t - 1] @ A) * B[:, obs[t]]
        c[t] = a.sum()
        ah[t] = a / c[t]
    return ah, c, float(np.log(c).sum())


def backward_scaled(A, B, obs, c):
    T, N = len(obs), A.shape[0]
    bh = np.ones((T, N))
    for t in range(T - 2, -1, -1):
        bh[t] = (A @ (B[:, obs[t + 1]] * bh[t + 1])) / c[t + 1]
    return bh


def posteriors(pi, A, B, obs):
    ah, c, ll = forward_scaled(pi, A, B, obs)
    bh = backward_scaled(A, B, obs, c)
    gamma = ah * bh
    gamma /= gamma.sum(1, keepdims=True)
    return gamma, ah, bh, c, ll


def baum_welch(seqs, N, M, iters=40, seed=0, pi=None, A=None, B=None, floor=1e-6):
    """EM for a discrete HMM over a list of integer sequences. Returns (pi, A, B, loglik_per_iteration)."""
    rng = np.random.RandomState(seed)
    if pi is None:
        pi = rng.dirichlet(np.ones(N))
        A = rng.dirichlet(np.ones(N), size=N)
        B = rng.dirichlet(np.ones(M), size=N)
    curve = []
    for _ in range(iters):
        pi_n = np.zeros(N)
        A_num = np.zeros((N, N))
        B_num = np.zeros((N, M))
        total = 0.0
        for obs in seqs:
            ah, c, ll = forward_scaled(pi, A, B, obs)
            bh = backward_scaled(A, B, obs, c)
            gamma = ah * bh
            gamma /= gamma.sum(1, keepdims=True)
            total += ll
            pi_n += gamma[0]
            if len(obs) > 1:
                W = (B[:, obs[1:]].T * bh[1:]) / c[1:, None]
                A_num += A * (ah[:-1].T @ W)
            np.add.at(B_num.T, obs, gamma)
        curve.append(total)
        pi = np.maximum(pi_n / pi_n.sum(), floor)
        pi /= pi.sum()
        A = np.maximum(A_num / A_num.sum(1, keepdims=True), floor)
        A /= A.sum(1, keepdims=True)
        B = np.maximum(B_num / B_num.sum(1, keepdims=True), floor)
        B /= B.sum(1, keepdims=True)
    return pi, A, B, curve


def viterbi(pi, A, B, obs):
    T, N = len(obs), len(pi)
    lp, lA, lB = np.log(pi), np.log(A), np.log(B)
    d = np.zeros((T, N))
    psi = np.zeros((T, N), dtype=int)
    d[0] = lp + lB[:, obs[0]]
    for t in range(1, T):
        sc = d[t - 1][:, None] + lA
        psi[t] = sc.argmax(0)
        d[t] = sc.max(0) + lB[:, obs[t]]
    path = np.zeros(T, dtype=int)
    path[-1] = d[-1].argmax()
    for t in range(T - 2, -1, -1):
        path[t] = psi[t + 1, path[t + 1]]
    return path, float(d[-1].max())


def supervised_hmm(seqs_obs, seqs_states, N, M, alpha=0.5):
    """Maximum-likelihood HMM from labelled sequences (counts + additive smoothing)."""
    pi = np.full(N, alpha)
    A = np.full((N, N), alpha)
    B = np.full((N, M), alpha)
    for o, s in zip(seqs_obs, seqs_states):
        pi[s[0]] += 1
        for t in range(len(s)):
            B[s[t], o[t]] += 1
            if t:
                A[s[t - 1], s[t]] += 1
    return pi / pi.sum(), A / A.sum(1, keepdims=True), B / B.sum(1, keepdims=True)


def kalman(z, F, H, Q, R, x0, P0):
    """Kalman filter + Rauch-Tung-Striebel smoother. z: (T, m). Returns filtered, smoothed means and filtered variances."""
    T, n = len(z), len(x0)
    xf, Pf, xp, Pp = np.zeros((T, n)), np.zeros((T, n, n)), np.zeros((T, n)), np.zeros((T, n, n))
    x, P = x0, P0
    for t in range(T):
        xp[t] = F @ x if t else x0
        Pp[t] = F @ P @ F.T + Q if t else P0
        S = H @ Pp[t] @ H.T + R
        K = Pp[t] @ H.T @ np.linalg.inv(S)
        x = xp[t] + K @ (z[t] - H @ xp[t])
        P = (np.eye(n) - K @ H) @ Pp[t]
        xf[t], Pf[t] = x, P
    xs = xf.copy()
    for t in range(T - 2, -1, -1):
        G = Pf[t] @ F.T @ np.linalg.inv(Pp[t + 1])
        xs[t] = xf[t] + G @ (xs[t + 1] - xp[t + 1])
    return xf, xs, Pf
