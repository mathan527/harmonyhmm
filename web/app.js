/* HarmonyHMM: runs the trained HMMs (exported from NumPy) in the browser. */
const $ = id => document.getElementById(id);
const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const NOTE = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"];
const SCALE = { major: [0, 2, 4, 5, 7, 9, 11], minor: [0, 2, 3, 5, 7, 8, 10] };
const ROWS = 15, STEPS = 16;
let D, charts = {}, melody = Array(STEPS).fill(null), shortN = Array(STEPS).fill(false), audio, playing = [];

const semis = (row, mode) => SCALE[mode][row % 7] + 12 * Math.floor(row / 7);
const chordName = (i, tonic) => NOTE[((i % 12) + tonic) % 12] + (i >= 12 ? "m" : "");
const fmt = (x, d = 1) => Number(x).toLocaleString(undefined, { maximumFractionDigits: d });
const pct = x => (x * 100).toFixed(1) + "%";

function mk(id, cfg) {
  if (charts[id]) charts[id].destroy();
  Chart.defaults.color = css("--mut"); Chart.defaults.borderColor = css("--line");
  cfg.options = Object.assign({ responsive: true, maintainAspectRatio: false }, cfg.options || {});
  charts[id] = new Chart($(id), cfg);
}

/* ---------------- HMM math (log domain / scaled) ---------------- */
const lg = m => m.map(r => Array.isArray(r) ? r.map(Math.log) : Math.log(r));
function viterbi(model, obs, noTrans) {
  const lp = lg(model.pi), lA = lg(model.A), lB = lg(model.B), N = lp.length, T = obs.length;
  if (noTrans) return obs.map(o => { let b = 0; for (let i = 1; i < N; i++) if (lp[i] + lB[i][o] > lp[b] + lB[b][o]) b = i; return b; });
  const d = [], psi = [];
  d[0] = lp.map((p, i) => p + lB[i][obs[0]]); psi[0] = Array(N).fill(0);
  for (let t = 1; t < T; t++) {
    d[t] = []; psi[t] = [];
    for (let j = 0; j < N; j++) {
      let best = -Infinity, arg = 0;
      for (let i = 0; i < N; i++) { const s = d[t - 1][i] + lA[i][j]; if (s > best) { best = s; arg = i; } }
      d[t][j] = best + lB[j][obs[t]]; psi[t][j] = arg;
    }
  }
  const path = Array(T); path[T - 1] = d[T - 1].indexOf(Math.max(...d[T - 1]));
  for (let t = T - 2; t >= 0; t--) path[t] = psi[t + 1][path[t + 1]];
  return path;
}
function forwardLL(model, obs) {            // scaled forward algorithm: sum of log scaling factors
  const { pi, A, B } = model, N = pi.length;
  let a = pi.map((p, i) => p * B[i][obs[0]]), ll = 0, c = a.reduce((x, y) => x + y, 0);
  ll += Math.log(c); a = a.map(x => x / c);
  for (let t = 1; t < obs.length; t++) {
    const n = Array(N).fill(0);
    for (let j = 0; j < N; j++) { let s = 0; for (let i = 0; i < N; i++) s += a[i] * A[i][j]; n[j] = s * B[j][obs[t]]; }
    c = n.reduce((x, y) => x + y, 0); ll += Math.log(c); a = n.map(x => x / c);
  }
  return ll;
}

/* ---------------- audio ---------------- */
function ctx() { if (!audio) audio = new (window.AudioContext || window.webkitAudioContext)(); if (audio.state === "suspended") audio.resume(); return audio; }
function tone(midi, t0, dur, type, vol) {
  const a = ctx(), o = a.createOscillator(), g = a.createGain();
  o.type = type; o.frequency.value = 440 * Math.pow(2, (midi - 69) / 12);
  g.gain.setValueAtTime(0, t0); g.gain.linearRampToValueAtTime(vol, t0 + 0.02);
  g.gain.exponentialRampToValueAtTime(0.0001, t0 + dur); o.connect(g).connect(a.destination); o.start(t0); o.stop(t0 + dur + 0.05);
}
function stopPlay() { playing.forEach(clearTimeout); playing = []; document.querySelectorAll(".play").forEach(e => e.classList.remove("play")); }

/* ---------------- harmonizer ---------------- */
const tonic = () => +$("key").value, mode = () => $("mode").value;
const n_ = () => { const i = melody.indexOf(null); return i < 0 ? STEPS : i; };
let lastChords = null;

function buildGrid() {
  const g = $("grid"); g.style.gridTemplateColumns = `30px repeat(${STEPS},34px)`; g.innerHTML = "";
  const deg = ["1", "2", "3", "4", "5", "6", "7"];
  for (let r = ROWS - 1; r >= 0; r--) {
    const l = document.createElement("div"); l.className = "lbl"; l.textContent = deg[r % 7] + (r >= 7 ? "'" : ""); g.appendChild(l);
    for (let s = 0; s < STEPS; s++) {
      const c = document.createElement("div"); c.className = "c" + (melody[s] === r ? " on" : ""); c.dataset.r = r; c.dataset.s = s;
      c.onclick = () => { melody[s] = melody[s] === r ? null : r; if (melody[s] !== null) previewNote(r); changed(); };
      g.appendChild(c);
    }
  }
  const l = document.createElement("div"); l.className = "lbl"; l.textContent = "short"; l.style.fontSize = "10px"; g.appendChild(l);
  for (let s = 0; s < STEPS; s++) {
    const c = document.createElement("div"); c.className = "sh" + (shortN[s] ? " on" : ""); c.textContent = "♪";
    c.onclick = () => { shortN[s] = !shortN[s]; changed(); }; g.appendChild(c);
  }
}
function previewNote(r) { tone(60 + (tonic() > 6 ? tonic() - 12 : tonic()) + semis(r, mode()), ctx().currentTime, 0.35, "triangle", 0.2); }
function obsPcs() { const n = n_(); return melody.slice(0, n).map(r => semis(r, mode()) % 12); }

function changed() {
  stopPlay(); buildGrid();
  const n = n_(), box = $("chordRow"); box.innerHTML = ""; lastChords = null;
  if (n < 2) { $("hInfo").textContent = n ? "Add at least one more note." : "Click the grid to write a melody."; if (D) drawStyle(); return; }
  const m = D.harmony.models[mode()], obs = obsPcs();
  lastChords = viterbi(m, obs, $("nohmm").checked);
  const tn = tonic();
  lastChords.forEach(c => { const d = document.createElement("div"); d.textContent = chordName(c, tn); box.appendChild(d); });
  const ev = D.harmony.eval[mode()];
  $("hInfo").innerHTML = `Chords decoded by ${$("nohmm").checked ? "per-note emission only" : "Viterbi over the HMM"} (${n} notes). On held-out Bach chorales this model gets the exact chord right ${pct(ev.viterbi_chord_acc)} of the time (baseline ${pct(ev.per_beat_baseline_acc)}); many musically valid harmonizations differ from Bach's choice.` + (window.__bach ? `<br>Bach's own chords: <b>${window.__bach.map(c => chordName(c, tn)).slice(0, n).join(" ")}</b>` : "");
  if (D) drawStyle();
}
function melodyMidi() { const t = tonic() > 6 ? tonic() - 12 : tonic(); return melody.slice(0, n_()).map(r => 60 + t + semis(r, mode())); }
function playMelody(withChords) {
  stopPlay(); const a = ctx(), beat = 60 / +$("bpm").value, t0 = a.currentTime + 0.1, mid = melodyMidi(), tn = tonic();
  const cells = [...document.querySelectorAll("#grid .c")], boxes = [...$("chordRow").children];
  mid.forEach((m, i) => {
    tone(m, t0 + i * beat, beat * 0.95, "triangle", 0.22);
    if (withChords && lastChords) {
      const c = lastChords[i], root = 48 + ((c % 12) + tn) % 12, third = c >= 12 ? 3 : 4;
      [0, third, 7].forEach(iv => tone(root + iv, t0 + i * beat, beat * 0.95, "sine", 0.1));
    }
    playing.push(setTimeout(() => {
      document.querySelectorAll(".play").forEach(e => e.classList.remove("play"));
      cells.filter(c => +c.dataset.s === i && +c.dataset.r === melody[i]).forEach(c => c.classList.add("play"));
      if (boxes[i]) boxes[i].classList.add("play");
    }, i * beat * 1000 + 100));
  });
  playing.push(setTimeout(stopPlay, mid.length * beat * 1000 + 200));
}

function loadPreset(v) {
  window.__bach = null;
  if (v === "ode") { $("mode").value = "major"; $("key").value = 0; melody = [2, 2, 3, 4, 4, 3, 2, 1, 0, 0, 1, 2, 2, 1, 1, null]; }
  else if (v === "twinkle") { $("mode").value = "major"; $("key").value = 0; melody = [0, 0, 4, 4, 5, 5, 4, 3, 3, 2, 2, 1, 1, 0, null, null]; }
  else if (v.startsWith("b")) {
    const p = D.presets[+v.slice(1)]; $("mode").value = "major"; $("key").value = 0;
    const maj = SCALE.major;
    melody = p.mel.slice(0, STEPS).map(pc => { let best = 0, bd = 99; for (let r = 0; r < 7; r++) { const d = Math.abs(maj[r] - pc); if (d < bd) { bd = d; best = r; } } return best; });
    window.__bach = p.chords;
  } else melody = Array(STEPS).fill(null);
  shortN = Array(STEPS).fill(false); changed();
}

/* ---------------- style detector ---------------- */
function drawStyle() {
  if (!D) return;
  const mid = melodyMidi(), n = mid.length;
  if (n < 5) { $("styleInfo").textContent = "Write at least 5 notes on the Harmonize tab first."; if (charts.styleChart) { charts.styleChart.destroy(); charts.styleChart = null; } return; }
  const obs = [];
  for (let i = 1; i < n; i++) obs.push((Math.max(-7, Math.min(7, mid[i] - mid[i - 1])) + 7) * 2 + (shortN[i] ? 0 : 1));
  const S = D.style, lls = S.names.map(nm => forwardLL(S.models[nm], obs));
  const mx = Math.max(...lls), ex = lls.map(l => Math.exp(l - mx)), z = ex.reduce((a, b) => a + b, 0), post = ex.map(e => e / z);
  mk("styleChart", { type: "bar", data: { labels: S.names, datasets: [{ data: post.map(p => p * 100), backgroundColor: [css("--acc"), css("--acc2"), css("--good")] }] },
    options: { indexAxis: "y", plugins: { legend: { display: false } }, scales: { x: { max: 100, title: { display: true, text: "Posterior probability (equal priors), %" } } } } });
  const best = S.names[post.indexOf(Math.max(...post))];
  const e = S.eval;
  $("styleInfo").innerHTML = `Best match: <b>${best}</b>. Log-likelihood per interval: ${S.names.map((nm, i) => `${nm} ${(lls[i] / obs.length).toFixed(2)}`).join(" · ")}. Held-out accuracy of this classifier is ${pct(e.acc_hmm)} (chance ${pct(e.chance)}), but short hand-written melodies are far less reliable than full tunes, and rhythm is a big part of the signal.`;
}

/* ---------------- tempo tracker (Kalman filter) ---------------- */
let taps = [], kal = null;
function kalmanStep(z) {
  const Q = D.kalman.Q, R = D.kalman.R;
  if (!kal) { kal = { x: [z, 0], P: [[0.05, 0], [0, 0.05]], out: [z] }; return z; }
  let { x, P } = kal;
  const xp = [x[0] + x[1], x[1]];
  const Pp = [[P[0][0] + P[0][1] + P[1][0] + P[1][1] + Q[0], P[0][1] + P[1][1]], [P[1][0] + P[1][1], P[1][1] + Q[1]]];
  const S = Pp[0][0] + R, K = [Pp[0][0] / S, Pp[1][0] / S], y = z - xp[0];
  kal.x = [xp[0] + K[0] * y, xp[1] + K[1] * y];
  kal.P = [[(1 - K[0]) * Pp[0][0], (1 - K[0]) * Pp[0][1]], [Pp[1][0] - K[1] * Pp[0][0], Pp[1][1] - K[1] * Pp[0][1]]];
  kal.out.push(kal.x[0]); return kal.x[0];
}
function tap() {
  const now = performance.now() / 1000;
  if (taps.length && now - taps[taps.length - 1] > 2.5) { taps = []; kal = null; }
  taps.push(now);
  tone(76, ctx().currentTime, 0.08, "square", 0.08);
  if (taps.length >= 2) kalmanStep(taps[taps.length - 1] - taps[taps.length - 2]);
  drawTaps();
}
function drawTaps() {
  const iv = taps.slice(1).map((t, i) => t - taps[i]), bpm = x => 60 / x;
  if (iv.length < 3) { $("tapInfo").textContent = "Tap at least 4 times (a pause over 2.5 s starts a new run)."; return; }
  $("tapInfo").innerHTML = `Raw last interval: <b>${fmt(bpm(iv[iv.length - 1]), 1)} BPM</b> · Kalman estimate: <b>${fmt(bpm(kal.x[0]), 1)} BPM</b> · tempo drift: ${(kal.x[1] * 1000).toFixed(1)} ms/tap`;
  mk("tapChart", { type: "line", data: { labels: iv.map((_, i) => i + 1), datasets: [
    { label: "Raw tap tempo", data: iv.map(bpm), borderColor: css("--mut"), pointRadius: 3, borderWidth: 1 },
    { label: "Kalman filtered", data: kal.out.map(bpm), borderColor: css("--acc"), pointRadius: 0, borderWidth: 2.5 }] },
    options: { animation: false, scales: { y: { title: { display: true, text: "BPM" } }, x: { title: { display: true, text: "Interval #" } } } } });
}
function drawKalman() {
  const K = D.kalman;
  mk("kalmanChart", { type: "line", data: { labels: K.true.map((_, i) => i + 1), datasets: [
    { label: "True period", data: K.true, borderColor: css("--good"), pointRadius: 0, borderWidth: 2 },
    { label: "Noisy taps", data: K.obs, borderColor: css("--mut"), pointRadius: 2, showLine: false },
    { label: "Filtered", data: K.filtered, borderColor: css("--acc"), pointRadius: 0 },
    { label: "Smoothed (RTS)", data: K.smoothed, borderColor: css("--acc2"), pointRadius: 0 }] },
    options: { scales: { y: { title: { display: true, text: "Beat period (s)" } } } } });
  $("kalmanInfo").textContent = `RMSE vs true period: raw ${K.rmse_raw.toFixed(4)} s, Kalman filter ${K.rmse_filtered.toFixed(4)} s, RTS smoother ${K.rmse_smoothed.toFixed(4)} s.`;
}

/* ---------------- under the hood ---------------- */
function lab() {
  const v = D.verification, he = D.harmony.eval, se = D.style.eval, U = D.unsup, sc = D.scaling;
  const heat = (M, size, lbl) => {
    const mx = Math.max(...M.flat());
    return `<div class="heat" style="grid-template-columns:30px repeat(${size},1fr)"><div></div>${[...Array(size).keys()].map(i => `<div style="color:var(--mut)">${lbl(i)}</div>`).join("")}` +
      M.map((r, i) => `<div style="color:var(--mut)">${lbl(i)}</div>` + r.map(x => `<div style="background:rgba(107,63,209,${(x / mx).toFixed(2)});color:${x / mx > .5 ? "#fff" : "inherit"}">${x >= 0.01 ? x.toFixed(2) : ""}</div>`).join("")).join("") + "</div>";
  };
  const conf = (M, names) => `<table><tr><th>true \\ predicted</th>${names.map(n => `<th class="n">${n}</th>`).join("")}</tr>${M.map((r, i) => `<tr><td>${names[i]}</td>${r.map((x, j) => `<td class="n" style="${i === j ? "font-weight:600" : ""}">${x}</td>`).join("")}</tr>`).join("")}</table>`;
  $("lab").innerHTML = `<h2>Under the hood</h2>
  <p class="lead">Every algorithm is written from scratch in NumPy (<code>pipeline/hmm.py</code>) and checked against the hmmlearn library. The browser runs the exported parameters, so Viterbi and the forward algorithm you used above are also implemented in JavaScript.</p>
  <div class="kpis">
    <div class="kpi"><div class="v">${pct(he.major.viterbi_chord_acc)}</div><div class="l">chord accuracy, major (baseline ${pct(he.major.per_beat_baseline_acc)})</div></div>
    <div class="kpi"><div class="v">${pct(he.minor.viterbi_chord_acc)}</div><div class="l">chord accuracy, minor (baseline ${pct(he.minor.per_beat_baseline_acc)})</div></div>
    <div class="kpi"><div class="v">${pct(se.acc_hmm)}</div><div class="l">style accuracy (Markov chain ${pct(se.acc_markov)})</div></div>
    <div class="kpi"><div class="v">${v.available ? v.loglik_abs_diff.toExponential(0) : "n/a"}</div><div class="l">|loglik difference| vs hmmlearn</div></div>
  </div>
  <div class="warn"><b>Honest reading.</b> Choosing a beat-level chord from the soprano line alone is ambiguous, so exact-chord accuracy is modest and the HMM's gain over the per-note baseline is small for major keys and larger for minor keys. For style, a plain Markov chain does about as well as the HMM on this data, and the folk-tune class is easy mainly because of rhythm. Test sets are chorales/tunes held out from training (80/20 split).</div>
  <div class="grid">
    <div class="card"><h3>Baum-Welch (EM) log-likelihood</h3><div class="cv"><canvas id="bwChart"></canvas></div><p class="note">Unsupervised 8-state HMM on melodies; EM never decreases the likelihood. Decoded hidden states agree with the real chords at NMI ${U.nmi.toFixed(2)}.</p></div>
    <div class="card"><h3>Why scaling factors matter</h3><div class="cv"><canvas id="scChart"></canvas></div><p class="note">Unscaled forward probability of a ${sc.length}-note sequence reaches 0 (floating-point underflow) at step ${sc.underflow_step}. The scaled forward pass still gives log-likelihood ${fmt(sc.scaled_loglik, 1)}.</p></div>
    <div class="card"><h3>Learned hidden states (unsupervised)</h3><table><tr><th>State</th><th class="n">Share</th><th>Likeliest notes</th><th>Most common real chord</th><th class="n">Purity</th></tr>
      ${U.states.map(s => `<tr><td>S${s.state}</td><td class="n">${pct(s.share)}</td><td>${(s.top_notes || []).join(", ")}</td><td>${s.top_chord}</td><td class="n">${pct(s.purity)}</td></tr>`).join("")}</table></div>
    <div class="card"><h3>Learned transition matrix</h3>${heat(U.A, 8, i => "S" + i)}<p class="note">Row = from state, column = to state.</p></div>
    <div class="card"><h3>Style classifier confusion (HMM)</h3>${conf(se.confusion_hmm, se.names)}<p class="note">Markov-chain baseline:</p>${conf(se.confusion_markov, se.names)}</div>
    <div class="card"><h3>Verification against hmmlearn</h3>${v.available ? `<table>
      <tr><td>Forward log-likelihood (ours vs library)</td><td class="n">${v.loglik_ours.toFixed(6)} / ${v.loglik_hmmlearn.toFixed(6)}</td></tr>
      <tr><td>Viterbi path agreement</td><td class="n">${pct(v.viterbi_agreement)}</td></tr>
      <tr><td>One Baum-Welch step: max |ΔA|, max |ΔB|</td><td class="n">${v.em_step_max_abs_diff_A.toExponential(1)}, ${v.em_step_max_abs_diff_B.toExponential(1)}</td></tr></table>` : "hmmlearn unavailable when the models were built."}
      <p class="note">Data: ${D.meta.corpus}. ${D.meta.chorales} Bach chorales (${D.meta.train_chorales} train / ${D.meta.test_chorales} test).</p></div>
  </div>`;
  mk("bwChart", { type: "line", data: { labels: U.curve.map((_, i) => i + 1), datasets: [{ data: U.curve, borderColor: css("--acc"), pointRadius: 0 }] }, options: { plugins: { legend: { display: false } }, scales: { x: { title: { display: true, text: "EM iteration" } } } } });
  mk("scChart", { type: "line", data: { labels: sc.unscaled_log10.map((_, i) => i * 25), datasets: [{ label: "log10 unscaled forward prob.", data: sc.unscaled_log10, borderColor: css("--bad"), spanGaps: false, pointRadius: 0 }] },
    options: { scales: { x: { title: { display: true, text: "Sequence position" } }, y: { title: { display: true, text: "log10 P(obs[1..t])" } } } } });
}

/* ---------------- boot ---------------- */
const done = {};
function show(t) {
  document.querySelectorAll(".tab").forEach(e => e.classList.toggle("on", e.id === t));
  document.querySelectorAll("#nav button").forEach(b => b.classList.toggle("on", b.dataset.tab === t));
  if (t === "style") drawStyle();
  if (t === "tempo" && !done.tempo) { drawKalman(); done.tempo = true; }
  if (t === "lab" && !done.lab) { lab(); done.lab = true; }
}
fetch("data/model.json").then(r => r.json()).then(d => {
  D = d;
  $("key").innerHTML = NOTE.map((n, i) => `<option value="${i}">${n}</option>`).join("");
  $("preset").innerHTML = `<option value="">Blank</option><option value="ode">Ode to Joy</option><option value="twinkle">Twinkle Twinkle</option>` + d.presets.map((p, i) => `<option value="b${i}">${p.title}</option>`).join("");
  $("preset").onchange = e => loadPreset(e.target.value);
  $("key").onchange = () => { changed(); }; $("mode").onchange = changed; $("nohmm").onchange = changed;
  $("bpm").oninput = () => $("bpmv").textContent = $("bpm").value;
  $("playH").onclick = () => { if (!lastChords) return; playMelody(true); };
  $("playM").onclick = () => { if (n_() < 1) return; playMelody(false); };
  $("clear").onclick = () => { $("preset").value = ""; loadPreset(""); };
  $("tap").onclick = tap; $("tapReset").onclick = () => { taps = []; kal = null; $("tapInfo").textContent = "Tap at least 4 times."; if (charts.tapChart) { charts.tapChart.destroy(); charts.tapChart = null; } };
  document.querySelectorAll("#nav button").forEach(b => b.onclick = () => show(b.dataset.tab));
  $("preset").value = "ode"; loadPreset("ode");
}).catch(e => { document.querySelector("main").innerHTML = "<p>Could not load model.json: " + e + "</p>"; });
