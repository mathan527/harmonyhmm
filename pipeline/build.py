"""Train HarmonyHMM models, evaluate them, verify against hmmlearn, export web/data/model.json."""
import json
import sys
import warnings
from pathlib import Path

import numpy as np
from sklearn.metrics import normalized_mutual_info_score

sys.path.insert(0, str(Path(__file__).parent))
import hmm  # noqa: E402

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
rs = np.random.RandomState(11)
NOTE = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]


def chord_name(i):
    return NOTE[i % 12] + ("m" if i >= 12 else "")


def r(a, n=5):
    return np.round(np.asarray(a, dtype=float), n).tolist()


# ------------------------------------------------------------------ harmonisation (hidden chords)
H = json.loads((ROOT / "data" / "harmony.json").read_text())
idx = rs.permutation(len(H))
cut = int(0.8 * len(H))
train_ids, test_ids = idx[:cut], idx[cut:]
harm_models, harm_eval = {}, {}
for mode, mname in ((0, "major"), (1, "minor")):
    tr = [H[i] for i in train_ids if H[i]["mode"] == mode]
    te = [H[i] for i in test_ids if H[i]["mode"] == mode]
    pi, A, B = hmm.supervised_hmm([c["mel"] for c in tr], [c["chords"] for c in tr], 24, 12)
    harm_models[mname] = {"pi": r(pi), "A": r(A), "B": r(B)}
    prior = np.bincount(np.concatenate([c["chords"] for c in tr]), minlength=24) + 1.0
    prior /= prior.sum()
    ok = ok_root = ok_base = n = 0
    for c in te:
        obs = np.array(c["mel"])
        true = np.array(c["chords"])
        path, _ = hmm.viterbi(pi, A, B, obs)
        base = np.array([(B[:, o] * prior).argmax() for o in obs])
        ok += (path == true).sum()
        ok_root += ((path % 12) == (true % 12)).sum()
        ok_base += (base == true).sum()
        n += len(true)
    harm_eval[mname] = {"train_pieces": len(tr), "test_pieces": len(te), "beats": int(n),
                        "viterbi_chord_acc": float(ok / n), "viterbi_root_acc": float(ok_root / n),
                        "per_beat_baseline_acc": float(ok_base / n)}
    print(mname, harm_eval[mname])

# ------------------------------------------------------------------ unsupervised Baum-Welch on melodies
maj = [H[i] for i in train_ids if H[i]["mode"] == 0]
seqs = [np.array(c["mel"]) for c in maj]
best = None
for seed in range(4):
    out = hmm.baum_welch(seqs, 8, 12, iters=60, seed=seed)
    if best is None or out[3][-1] > best[3][-1]:
        best = out
upi, uA, uB, ucurve = best
# decode held-out major chorales and compare hidden states with the real chords
tmaj = [H[i] for i in test_ids if H[i]["mode"] == 0]
st, ch = [], []
for c in tmaj:
    p, _ = hmm.viterbi(upi, uA, uB, np.array(c["mel"]))
    st += p.tolist()
    ch += c["chords"]
st, ch = np.array(st), np.array(ch)
nmi = float(normalized_mutual_info_score(ch, st))
unsup_states = []
for k in range(8):
    m = st == k
    if m.sum() == 0:
        unsup_states.append({"state": k, "share": 0, "top_chord": "-", "purity": 0})
        continue
    cnt = np.bincount(ch[m], minlength=24)
    unsup_states.append({"state": k, "share": float(m.mean()), "top_chord": chord_name(int(cnt.argmax())),
                         "purity": float(cnt.max() / m.sum()),
                         "top_notes": [NOTE[i] for i in np.argsort(-uB[k])[:3]]})
print("unsupervised NMI", nmi, "LL start/end", ucurve[0], ucurve[-1])

# ------------------------------------------------------------------ verification vs hmmlearn
try:
    from hmmlearn import hmm as hl
    obs = np.array(maj[0]["mel"])
    ref = hl.CategoricalHMM(n_components=8, n_features=12, init_params="", params="", n_iter=1)
    ref.startprob_, ref.transmat_, ref.emissionprob_ = upi, uA, uB
    ll_ref = float(ref.score(obs.reshape(-1, 1)))
    _, _, ll_us = hmm.forward_scaled(upi, uA, uB, obs)
    v_ref = ref.decode(obs.reshape(-1, 1))[1]
    v_us, _ = hmm.viterbi(upi, uA, uB, obs)
    # one EM step from identical init
    p0 = np.random.RandomState(5)
    pi0, A0, B0 = p0.dirichlet(np.ones(8)), p0.dirichlet(np.ones(8), size=8), p0.dirichlet(np.ones(12), size=8)
    s2 = [np.array(c["mel"]) for c in maj[:30]]
    pi1, A1, B1, _ = hmm.baum_welch(s2, 8, 12, iters=1, pi=pi0.copy(), A=A0.copy(), B=B0.copy(), floor=0)
    m = hl.CategoricalHMM(n_components=8, n_features=12, init_params="", params="ste", n_iter=1, tol=0)
    m.startprob_, m.transmat_, m.emissionprob_ = pi0.copy(), A0.copy(), B0.copy()
    m.fit(np.concatenate(s2).reshape(-1, 1), [len(x) for x in s2])
    verification = {"available": True, "loglik_ours": float(ll_us), "loglik_hmmlearn": ll_ref,
                    "loglik_abs_diff": abs(ll_us - ll_ref), "viterbi_agreement": float((v_us == v_ref).mean()),
                    "em_step_max_abs_diff_A": float(np.abs(A1 - m.transmat_).max()),
                    "em_step_max_abs_diff_B": float(np.abs(B1 - m.emissionprob_).max())}
except Exception as e:  # pragma: no cover
    verification = {"available": False, "error": str(e)}
print(verification)

# ------------------------------------------------------------------ scaling demo
long_obs = np.tile(np.array(maj[0]["mel"]), 40)[:1500]
un = hmm.forward_unscaled(upi, uA, uB, long_obs)
under = int(np.argmax(un == 0)) if (un == 0).any() else None
_, c_, ll_long = hmm.forward_scaled(upi, uA, uB, long_obs)
scaling = {"length": int(len(long_obs)), "underflow_step": under, "scaled_loglik": float(ll_long),
           "unscaled_log10": [float(np.log10(x)) if x > 0 else None for x in un[::25]]}

# ------------------------------------------------------------------ style classifier (HMM per style)
S = json.loads((ROOT / "data" / "style.json").read_text())
NS = 30


def symbols(notes):
    out = []
    for i in range(1, len(notes)):
        iv = int(np.clip(notes[i][0] - notes[i - 1][0], -7, 7)) + 7
        out.append(iv * 2 + (0 if notes[i][1] < 1 else 1))
    return np.array(out)


names = list(S.keys())
split = {}
for nm in names:
    seq = [symbols(x["notes"]) for x in S[nm]]
    seq = [s for s in seq if len(s) >= 15]
    p = rs.permutation(len(seq))
    k = int(0.8 * len(seq))
    split[nm] = ([seq[i] for i in p[:k]], [seq[i] for i in p[k:]])
style_models, mc_models = {}, {}
for nm in names:
    tr = split[nm][0]
    best = None
    for seed in range(3):
        out = hmm.baum_welch(tr, 4, NS, iters=30, seed=seed)
        if best is None or out[3][-1] > best[3][-1]:
            best = out
    style_models[nm] = best
    cnt = np.full((NS, NS), 0.5)
    first = np.full(NS, 0.5)
    for s in tr:
        first[s[0]] += 1
        for a, b in zip(s[:-1], s[1:]):
            cnt[a, b] += 1
    mc_models[nm] = (first / first.sum(), cnt / cnt.sum(1, keepdims=True))


def mc_ll(m, s):
    return np.log(m[0][s[0]]) + sum(np.log(m[1][a, b]) for a, b in zip(s[:-1], s[1:]))


conf = np.zeros((3, 3), int)
conf_mc = np.zeros((3, 3), int)
for i, nm in enumerate(names):
    for s in split[nm][1]:
        lls = [hmm.forward_scaled(*style_models[n][:3], s)[2] for n in names]
        conf[i, int(np.argmax(lls))] += 1
        conf_mc[i, int(np.argmax([mc_ll(mc_models[n], s) for n in names]))] += 1
style_eval = {"names": names, "confusion_hmm": conf.tolist(), "confusion_markov": conf_mc.tolist(),
              "acc_hmm": float(np.trace(conf) / conf.sum()), "acc_markov": float(np.trace(conf_mc) / conf_mc.sum()),
              "train": {n: len(split[n][0]) for n in names}, "test": {n: len(split[n][1]) for n in names},
              "chance": 1 / 3}
print(style_eval)

# ------------------------------------------------------------------ Kalman tempo demo
rk = np.random.RandomState(4)
true_p = np.concatenate([np.full(14, 0.60), np.linspace(0.60, 0.46, 10), np.full(14, 0.46), np.linspace(0.46, 0.58, 8), np.full(10, 0.58)])
z = true_p + rk.normal(0, 0.035, len(true_p))
F = np.array([[1., 1.], [0., 1.]])
Hm = np.array([[1., 0.]])
Q = np.array([[1e-5, 0], [0, 4e-6]])
R = np.array([[0.035 ** 2]])
xf, xs, _ = hmm.kalman(z.reshape(-1, 1), F, Hm, Q, R, np.array([z[0], 0.]), np.eye(2) * 0.05)
rmse = lambda a: float(np.sqrt(np.mean((a - true_p) ** 2)))
kalman_demo = {"true": r(true_p, 4), "obs": r(z, 4), "filtered": r(xf[:, 0], 4), "smoothed": r(xs[:, 0], 4),
               "rmse_raw": rmse(z), "rmse_filtered": rmse(xf[:, 0]), "rmse_smoothed": rmse(xs[:, 0]),
               "Q": [1e-5, 4e-6], "R": 0.035 ** 2}
print({k: kalman_demo[k] for k in ("rmse_raw", "rmse_filtered", "rmse_smoothed")})

# ------------------------------------------------------------------ presets from held-out chorales
presets = []
for i in test_ids:
    c = H[i]
    if c["mode"] == 0 and len(c["mel"]) >= 20 and len(presets) < 4:
        presets.append({"title": "Bach chorale " + c["title"], "mel": c["mel"][:16], "chords": c["chords"][:16], "mode": "major"})

out = {
    "meta": {"corpus": "music21 bundled corpus (Bach chorales, Palestrina, folk tunes)", "chorales": len(H),
             "train_chorales": int(len(train_ids)), "test_chorales": int(len(test_ids))},
    "chords": [chord_name(i) for i in range(24)],
    "harmony": {"models": harm_models, "eval": harm_eval},
    "unsup": {"pi": r(upi), "A": r(uA), "B": r(uB), "curve": r(ucurve, 1), "nmi": nmi, "states": unsup_states},
    "verification": verification, "scaling": scaling,
    "style": {"names": names, "n_symbols": NS,
              "models": {n: {"pi": r(style_models[n][0]), "A": r(style_models[n][1]), "B": r(style_models[n][2]), "curve": r(style_models[n][3], 1)} for n in names},
              "eval": style_eval},
    "kalman": kalman_demo, "presets": presets,
}
dst = ROOT / "web" / "data"
dst.mkdir(parents=True, exist_ok=True)
(dst / "model.json").write_text(json.dumps(out, separators=(",", ":")))
print("wrote", dst / "model.json", round((dst / "model.json").stat().st_size / 1e3), "KB")
