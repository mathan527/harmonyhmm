# ♫ HarmonyHMM

**Hidden Markov Models that harmonize a melody, recognize its musical style and track your tempo, with every algorithm written from scratch.**

[**Live demo →**](https://harmonyhmm.vercel.app) &nbsp;·&nbsp; Course: Machine Learning, **Unit 4** (Markov models, HMMs, forward–backward, Viterbi, linear dynamical systems)

Music is a natural hidden-state sequence: you hear the notes, but the chords underneath are hidden. Click a melody into the grid and HarmonyHMM decodes the most probable chords, plays the result, tells you which style it sounds like, and can even follow your tapping tempo.

---

## Try it

| Tab | What happens |
|---|---|
| **Harmonize** | Write a melody (or load Ode to Joy / a held-out Bach chorale). Viterbi decodes the chords; press play to hear them. |
| **Style detector** | One HMM per style (Bach chorales, Palestrina, folk tunes) scores your melody with the scaled forward algorithm. |
| **Tempo tracker** | Tap a beat. A Kalman filter estimates the hidden tempo and its drift live. |
| **Under the hood** | Baum–Welch likelihood curve, learned states, underflow demo, confusion matrices, library cross-check. |

## From scratch, and verified

`pipeline/hmm.py` implements, in plain NumPy:

- **Scaled forward and backward passes** (Rabiner scaling factors)
- **Baum–Welch** (EM for HMMs) over many sequences
- **Viterbi** in log space
- **Kalman filter and RTS smoother** for a linear dynamical system

Each one is checked against `hmmlearn`:

| Check | Result |
|---|---|
| Forward log-likelihood difference | **≈ 4 × 10⁻¹⁴** |
| Viterbi path agreement | **100%** |
| One Baum–Welch step, max |ΔA|, |ΔB| | **≈ 10⁻¹⁵** |

The same Viterbi and forward algorithms also run **in the browser** (JavaScript) on the exported parameters, so the demo needs no server.

## Results (held-out data, 80/20 split)

| Task | HarmonyHMM | Baseline |
|---|---|---|
| Chord accuracy, major keys (40 chorales, 2,137 beats) | **55.9%** | 55.5% (best chord per note) |
| Chord accuracy, minor keys (31 chorales, 1,549 beats) | **48.2%** | 42.7% |
| Style classification (3 classes, chance 33%) | **96.0%** | 96.6% (plain Markov chain) |
| Tempo error (RMSE of beat period) | **0.021 s** filtered / **0.013 s** smoothed | 0.034 s raw taps |

Why the Kalman result is the cleanest win: a 2-state linear dynamical system cuts timing error by about 40% (filter) and 63% (smoother) on a simulated tempo change.

## Honest limitations

- Picking a beat-level chord from the soprano line alone is ambiguous. Exact-chord accuracy is modest, and the HMM's gain over the per-note baseline is small in major keys.
- Chord labels are **template-matched triads** I derived from the four voices, not human annotations.
- A plain Markov chain classifies style about as well as the HMM here, and folk tunes are easy mainly because of their short-note rhythm. Short hand-written melodies are less reliable than full tunes.
- Unsupervised Baum–Welch states line up with real chords only weakly (NMI 0.28).

## Syllabus map (Unit 4)

| Topic | Where |
|---|---|
| Sequential data, Markov models | Markov-chain style baseline |
| HMM and maximum likelihood | supervised chord HMM (counts) and Baum–Welch |
| Forward–backward, sum-product, scaling factors | `forward_scaled`, `backward_scaled`; underflow demo (unscaled probability hits zero at note 462) |
| Viterbi | Harmonize tab |
| Linear dynamical systems | Kalman tempo tracker |
| Practice: implement an HMM to predict sequential data | the whole project |

## Data

Bach chorales (351 parsed), Palestrina and folk tunes (Ryan's Mammoth and Essen collections) from the corpus bundled with [`music21`](https://web.mit.edu/music21/). Pieces are transposed to a common tonic so one model serves every key.

## Run it yourself

```bash
pip install -r requirements.txt
python pipeline/extract.py     # corpus -> data/*.json
python pipeline/build.py       # train, evaluate, verify, write web/data/model.json
python -m http.server -d web   # open http://localhost:8000
```

```
pipeline/hmm.py       from-scratch HMM, Baum-Welch, Viterbi, Kalman
pipeline/extract.py   corpus extraction and chord labelling
pipeline/build.py     training, evaluation, hmmlearn verification
web/                  static demo (Web Audio + Chart.js), deployed on Vercel
```

*Companion project for Unit 3: [WattWise AI](https://github.com/mathan527/wattwise-ai) — unsupervised electricity-usage analysis.*
