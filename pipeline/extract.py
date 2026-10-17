"""Extract training sequences from the music21 corpus (bundled with `pip install music21`).

Harmony data : Bach chorales, one observation per beat -> (soprano pitch class, chord label), key-normalised.
Style data   : first-voice melodies of Bach chorales, Palestrina and folk tunes -> (interval, duration) symbols.
Output       : data/harmony.json, data/style.json
"""
import json
import random
import warnings
from multiprocessing import Pool
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data"

# 24 triads: index = root + 12*(0 major / 1 minor)
TEMPLATES = []
for q, ivs in enumerate(((0, 4, 7), (0, 3, 7))):
    for r in range(12):
        TEMPLATES.append((r, q, {(r + i) % 12 for i in ivs}))


def label_chord(pcs, bass):
    """Template matching: best triad for a pitch-class set; bass note breaks ties."""
    best, bs = None, -9
    for idx, (r, q, tpl) in enumerate(TEMPLATES):
        s = len(pcs & tpl) - 0.6 * len(tpl - pcs) - 0.4 * len(pcs - tpl) + (0.3 if r == bass else 0) + (0.1 if r in pcs else 0)
        if s > bs:
            best, bs = idx, s
    return best


def part_arrays(part):
    ns = [n for n in part.flatten().notes if n.isNote]
    st = np.array([float(n.offset) for n in ns])
    en = st + np.array([float(n.quarterLength) for n in ns])
    p = np.array([n.pitch.midi for n in ns])
    return st, en, p


def sound_at(arr, t):
    st, en, p = arr
    i = np.searchsorted(st, t, side="right") - 1
    return int(p[i]) if i >= 0 and en[i] > t + 1e-6 else None


def do_chorale(path):
    from music21 import corpus
    try:
        s = corpus.parse(path)
        if len(s.parts) != 4:
            return None
        k = s.analyze("key")
        tonic = k.tonic.pitchClass
        mode = 0 if k.mode == "major" else 1
        arrs = [part_arrays(p) for p in s.parts]
        total = int(max(a[1].max() for a in arrs if len(a[0])))
        mel, chords = [], []
        for t in range(total):
            ps = [sound_at(a, t) for a in arrs]
            if ps[0] is None or ps[3] is None:
                continue
            pcs = {(x - tonic) % 12 for x in ps if x is not None}
            bass = (ps[3] - tonic) % 12
            mel.append((ps[0] - tonic) % 12)
            chords.append(label_chord(pcs, bass))
        notes = [(int(n.pitch.midi), float(n.quarterLength)) for n in s.parts[0].flatten().notes if n.isNote]
        return {"mode": mode, "mel": mel, "chords": chords, "notes": notes, "title": Path(path).stem}
    except Exception:
        return None


def do_melody(path):
    from music21 import converter
    try:
        s = converter.parse(path)
        part = s.parts[0] if len(s.parts) else s
        notes = [(int(n.pitch.midi), float(n.quarterLength)) for n in part.flatten().notes if n.isNote]
        return {"notes": notes} if 20 <= len(notes) <= 600 else None
    except Exception:
        return None


def paths(sub, exts=(".xml", ".mxl", ".krn", ".abc", ".mid", ".musicxml", ".mxl")):
    import music21
    base = Path(music21.__file__).parent / "corpus" / sub
    return sorted(p for p in base.rglob("*") if p.suffix.lower() in exts)


def main():
    import music21
    from music21 import corpus
    random.seed(1)
    chor_paths = [str(p) for p in corpus.chorales.Iterator(returnType="filename")]
    with Pool(6) as pool:
        ch = [c for c in pool.map(do_chorale, chor_paths, chunksize=4) if c and len(c["mel"]) > 16]
        print("chorales", len(ch))
        sets = {}
        for name, subs, cap in (("Palestrina", ["palestrina"], 160), ("Folk tunes", ["ryansMammoth", "essenFolksong"], 360)):
            ps = [str(p) for sub in subs for p in paths(sub)]
            random.shuffle(ps)
            res = [r for r in pool.map(do_melody, ps[:cap * 2], chunksize=4) if r][:cap]
            sets[name] = res
            print(name, len(res))
    sets["Bach chorales"] = [{"notes": c["notes"]} for c in ch if 20 <= len(c["notes"]) <= 600]
    OUT.mkdir(exist_ok=True)
    (OUT / "harmony.json").write_text(json.dumps([{k: c[k] for k in ("mode", "mel", "chords", "title")} for c in ch]))
    (OUT / "style.json").write_text(json.dumps(sets))
    print("done", {k: len(v) for k, v in sets.items()})


if __name__ == "__main__":
    main()
