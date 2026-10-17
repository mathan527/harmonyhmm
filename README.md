# HarmonyHMM

Hidden Markov Models for music (ML Unit 4): harmonize a melody, recognize its style, track tempo.

| Unit 4 topic | Where |
|---|---|
| Markov models | style baseline (Markov chain), `pipeline/build.py` |
| HMM, maximum likelihood | supervised chord HMM (counts) + Baum-Welch EM (`pipeline/hmm.py`) |
| Forward-backward, scaling factors | `forward_scaled`, `backward_scaled`; underflow demo in "Under the hood" |
| Viterbi | chord decoding, `viterbi` (Python + JavaScript) |
| Linear dynamical systems | Kalman filter + RTS smoother, tempo tracker |

All algorithms are written from scratch in NumPy and checked against `hmmlearn`
(log-likelihood difference ~1e-14, identical Viterbi paths, identical EM step).

Data: Bach chorales, Palestrina and folk tunes bundled with `music21`.
Held-out results (80/20 split): chord accuracy 55.9% major / 48.2% minor (per-note baseline 55.5% / 42.7%),
style accuracy 96% (a plain Markov chain gets 96.6%). Chord labels are template-matched triads, not human annotations.

## Run
```
pip install -r requirements.txt
python pipeline/extract.py   # writes data/*.json
python pipeline/build.py     # trains, evaluates, writes web/data/model.json
python -m http.server -d web
```
`web/` is a static site deployed on Vercel.
