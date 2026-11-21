const MODEL_INFO = {
  eegnet: { desc: "Small neural network · learns over many epochs", files: ["live.py", "train.py", "eegnet.py"] },
  fbcsp: { desc: "Classic filter-bank method · trains in one go", files: ["live.py", "baselines.py"] },
  csp_lda: { desc: "Simplest baseline · trains in seconds", files: ["live.py", "baselines.py"] },
};
const CLASS_NAMES = { left_hand: "left hand", right_hand: "right hand", feet: "feet", tongue: "tongue" };
const MARKERS = {
  loading: { file: "live.py", text: "data = load_dataset(" },
  loaded: { file: "live.py", text: "split = within_subject_splits(" },
  eegnet: { file: "live.py", text: "model = EEGNet(" },
  improved: { file: "train.py", text: "best_state = {name: value" },
  backprop: { file: "train.py", text: "loss = F.cross_entropy(model(X[batch]), y[batch])" },
  forward: { file: "eegnet.py", text: "x = self.activation(self.spatial_norm(self.spatial(x)))" },
  validate: { file: "train.py", text: "val_loss = _mean_loss(" },
  kept: { file: "train.py", text: "model.load_state_dict(best_state)" },
  filter: { file: "baselines.py", text: "bands.append(filter_data(" },
  fbcspCsp: { file: "baselines.py", text: "csps = [self._new_csp().fit(" },
  csp: { file: "baselines.py", text: "csp.fit(wide, y_train)" },
  lda: { file: "baselines.py", text: "lda.fit(csp.transform(wide), y_train)" },
  tune: { file: "baselines.py", text: "accuracy = (predictions == y_val).mean()" },
  result: { file: "live.py", text: 'accuracy = float((model.predict(X["test"])' },
};

const $ = (id) => document.getElementById(id);
const pct = (a) => `${Math.round(a * 100)}%`;
const esc = (s) => String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
const clockTime = (s) => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`;
const shortTime = (s) => (s < 60 ? `${s.toFixed(s < 10 ? 1 : 0)} s` : clockTime(s));

const state = {
  mode: "live",
  options: null,
  recordings: [],
  sources: {},
  running: false,
  run: null,
  points: [],
  chance: 0.25,
  classes: [],
  limit: 120,
  epoch: null,
  activeFile: "live.py",
  marker: null,
  speed: 4,
  stream: null,
  replayTimer: null,
  bestVal: Infinity,
};

async function init() {
  try {
    const response = await fetch("api/options");
    if (!response.ok) throw new Error("no server");
    state.options = await response.json();
  } catch {
    state.mode = "replay";
    const recorded = await fetch("replay/runs.json").then((r) => r.json());
    state.options = recorded.options;
    state.recordings = recorded.runs;
    state.sources = recorded.sources;
  }
  setUpControls();
  if (state.mode === "replay") showReplayBanner();
  renderFiles();
  idleTerminal();
  draw();
  if (state.mode === "replay" && location.hash === "#play") {
    state.speed = 16;
    for (const b of $("speeds").children) b.setAttribute("aria-pressed", String(b.dataset.speed === "16"));
    start();
  }
}

function showReplayBanner() {
  const banner = $("banner");
  banner.className = "banner replay";
  banner.innerHTML = "<b>Recorded runs.</b> This page plays back real training runs recorded on a laptop, at real speed or faster. To train models live on any person, download the project and run <code>python app.py</code>.";
  banner.hidden = false;
  $("speed-field").hidden = false;
  $("train").textContent = "Play recorded run";
}

function setUpControls() {
  for (const d of state.options.datasets) $("dataset").add(new Option(d.title, d.key));
  for (const m of state.options.models) {
    $("models").insertAdjacentHTML("beforeend",
      `<label class="model"><input type="radio" name="model" id="model-${m.key}" value="${m.key}"${m.key === "eegnet" ? " checked" : ""}><b>${esc(m.name)}</b><span>${esc(MODEL_INFO[m.key]?.desc ?? "")}</span></label>`);
  }
  $("dataset").addEventListener("change", () => { fillPeople(); syncAvailability(); resetView(); });
  $("person").addEventListener("change", () => { syncAvailability(); resetView(); });
  $("models").addEventListener("change", () => { syncAvailability(); resetView(); });
  $("limit").addEventListener("input", () => {
    state.limit = +$("limit").value;
    $("limit-out").textContent = clockTime(state.limit);
    if (!state.running) idleTerminal();
    draw();
  });
  $("form").addEventListener("submit", (event) => { event.preventDefault(); start(); });
  $("stop").addEventListener("click", stop);
  $("chart").addEventListener("pointermove", hoverAt);
  $("chart").addEventListener("pointerdown", hoverAt);
  $("chart").addEventListener("pointerleave", () => setHover(null));
  $("chart").addEventListener("blur", () => setHover(null));
  $("chart").addEventListener("keydown", stepHover);
  $("tab-train").addEventListener("click", () => selectTab("train"));
  $("tab-code").addEventListener("click", () => selectTab("code"));
  $("files").addEventListener("click", (event) => {
    const button = event.target.closest(".file");
    if (button) showFile(button.dataset.file, false);
  });
  $("speeds").addEventListener("click", (event) => {
    const button = event.target.closest("button");
    if (!button) return;
    state.speed = +button.dataset.speed;
    for (const b of $("speeds").children) b.setAttribute("aria-pressed", String(b === button));
  });
  fillPeople();
  syncAvailability();
}

const dataset = () => state.options.datasets.find((d) => d.key === $("dataset").value);
const modelKey = () => document.querySelector('input[name="model"]:checked')?.value;
const modelName = (key) => state.options.models.find((m) => m.key === key)?.name ?? key;
const recording = () => state.recordings.find((r) => r.dataset === $("dataset").value && r.person === +$("person").value && r.model === modelKey());

function fillPeople() {
  const d = dataset();
  const classes = d.classes.map((c) => CLASS_NAMES[c] ?? c).join(", ");
  $("dataset-hint").textContent = `${d.channels} channels · ${classes} · ${d.people.length} people`;
  const people = state.mode === "replay"
    ? [...new Set(state.recordings.filter((r) => r.dataset === d.key).map((r) => r.person))]
    : d.people;
  $("person").innerHTML = "";
  for (const p of people) $("person").add(new Option(`Person ${p}`, p));
}

function syncAvailability() {
  if (state.mode !== "replay") return;
  for (const input of document.querySelectorAll('input[name="model"]')) {
    input.disabled = !state.recordings.some((r) => r.dataset === $("dataset").value && r.person === +$("person").value && r.model === input.value);
  }
  if (document.querySelector('input[name="model"]:checked')?.disabled) {
    document.querySelector('input[name="model"]:not(:disabled)')?.click();
  }
  const r = recording();
  if (r) {
    $("limit").max = r.time_limit;
    if (+$("limit").value > r.time_limit) $("limit").value = r.time_limit;
    state.limit = +$("limit").value;
    $("limit-out").textContent = clockTime(state.limit);
  }
}

function resetView() {
  if (state.running) return;
  state.points = [];
  state.chance = dataset().chance;
  renderFiles();
  idleTerminal();
  draw();
}

function selectTab(which) {
  const code = which === "code";
  $("tab-train").setAttribute("aria-selected", String(!code));
  $("tab-code").setAttribute("aria-selected", String(code));
  $("view-train").hidden = code;
  $("view-code").hidden = !code;
  if (code) markLine(state.marker);
}

function command(limit = state.limit) {
  return `python -m src.live <b>--dataset</b> ${esc($("dataset").value)} <b>--person</b> ${esc($("person").value)} <b>--model</b> ${esc(modelKey())} <b>--time-limit</b> ${limit}`;
}

function idleTerminal() {
  const note = state.mode === "replay" ? "# the command that produced this recording" : "# the command the app runs when you press Train model";
  $("term").innerHTML = `<span class="t-dim">${note}</span>\n<span class="cmd">$ ${command()}</span>\n<span class="cursor"></span>`;
}

function print(html) {
  const box = $("term");
  box.querySelector(".cursor")?.remove();
  box.insertAdjacentHTML("beforeend", `${html}\n`);
  while (box.childNodes.length > 1200) box.firstChild.remove();
  if (state.running) box.insertAdjacentHTML("beforeend", '<span class="cursor"></span>');
  box.scrollTop = box.scrollHeight;
}

const TOKEN = /("[^"\n]*"|'[^'\n]*')|(@[\w.]+)|\b(def|class|return|for|in|if|else|elif|import|from|as|with|not|and|or|is|None|True|False|self|super|break|lambda|try|except|raise|while|yield)\b|\b(\d+(?:\.\d+)?(?:e-?\d+)?)\b|\b([A-Za-z_]\w*)(?=\()/g;

function highlight(line) {
  let out = "";
  let last = 0;
  for (const match of line.matchAll(TOKEN)) {
    out += esc(line.slice(last, match.index));
    const kind = match[1] ? "s" : match[2] ? "d" : match[3] ? "k" : match[4] ? "m" : "f";
    out += `<span class="${kind}">${esc(match[0])}</span>`;
    last = match.index + match[0].length;
  }
  return out + esc(line.slice(last));
}

async function source(file) {
  if (!(file in state.sources)) {
    const response = await fetch(`api/source/${file}`);
    state.sources[file] = response.ok ? await response.text() : `# could not load ${file}`;
  }
  return state.sources[file];
}

function renderFiles() {
  const files = MODEL_INFO[modelKey()]?.files ?? ["live.py"];
  if (!files.includes(state.activeFile)) state.activeFile = files[0];
  $("files").innerHTML = files
    .map((f) => `<button class="file" role="tab" type="button" data-file="${f}" aria-selected="${f === state.activeFile}">src/${f}</button>`)
    .join("");
  showFile(state.activeFile, false);
}

async function showFile(file, scroll = true) {
  state.activeFile = file;
  for (const b of $("files").children) b.setAttribute("aria-selected", String(b.dataset.file === file));
  const text = await source(file);
  if (state.activeFile !== file) return;
  $("code").dataset.file = file;
  $("code").innerHTML = text
    .replace(/\n$/, "")
    .split("\n")
    .map((line, i) => `<div class="ln" data-i="${i}"><span class="n">${i + 1}</span><span>${highlight(line) || " "}</span></div>`)
    .join("");
  markLine(state.marker, scroll);
}

function markLine(marker, scroll = true) {
  state.marker = marker;
  const box = $("code");
  box.querySelectorAll(".ln.hl").forEach((el) => el.classList.remove("hl"));
  if (!marker) return;
  if (box.dataset.file !== marker.file) {
    showFile(marker.file, scroll);
    return;
  }
  const lines = state.sources[marker.file]?.split("\n") ?? [];
  const index = lines.findIndex((line) => line.includes(marker.text));
  if (index < 0) return;
  const el = box.querySelector(`.ln[data-i="${index}"]`);
  el.classList.add("hl");
  if (scroll && !$("view-code").hidden) {
    const top = el.offsetTop - box.offsetTop;
    if (top < box.scrollTop + 40 || top > box.scrollTop + box.clientHeight - 60) {
      box.scrollTop = Math.max(0, top - box.clientHeight / 3);
    }
  }
}

function bestWithin(points, limit) {
  let best = null;
  for (const p of points) if (p.t <= limit && (!best || p.a > best.a)) best = p;
  return best ?? (points.length ? points[0] : null);
}

function niceStep(span) {
  return [5, 10, 15, 30, 60, 120, 300].find((s) => span / s <= 8) ?? 600;
}

function draw() {
  const W = 720, H = 320, L = 48, R = 16, T = 16, B = 36;
  const limit = state.limit;
  const last = state.points.length ? state.points[state.points.length - 1].t : 0;
  const maxT = Math.max(limit, last, 10);
  const x = (t) => L + (t / maxT) * (W - L - R);
  const y = (a) => T + (1 - a) * (H - T - B);
  const label = (attrs, text) => `<text ${attrs} font-size="11" fill="var(--muted)" font-family="IBM Plex Mono, monospace">${text}</text>`;
  let svg = "";
  for (const a of [0, 0.25, 0.5, 0.75, 1]) {
    svg += `<line x1="${L}" x2="${W - R}" y1="${y(a)}" y2="${y(a)}" stroke="var(--line)"/>`;
    svg += label(`x="${L - 8}" y="${y(a) + 4}" text-anchor="end"`, pct(a));
  }
  const step = niceStep(maxT);
  for (let t = 0; t <= maxT + 1e-6; t += step) svg += label(`x="${x(t)}" y="${H - 12}" text-anchor="middle"`, clockTime(t));
  svg += `<line x1="${L}" x2="${W - R}" y1="${y(state.chance)}" y2="${y(state.chance)}" stroke="var(--muted)" stroke-dasharray="4 4"/>`;
  svg += `<text x="${W - R}" y="${y(state.chance) - 6}" text-anchor="end" font-size="11" fill="var(--muted)">Guessing: ${pct(state.chance)}</text>`;
  if (limit < maxT) {
    svg += `<rect x="${x(limit)}" y="${T}" width="${W - R - x(limit)}" height="${H - T - B}" fill="var(--line)" opacity="0.45"/>`;
  }
  svg += `<line x1="${x(limit)}" x2="${x(limit)}" y1="${T}" y2="${H - B}" stroke="var(--ink)"/>`;
  svg += `<text x="${x(limit) - 6}" y="${T + 12}" text-anchor="end" font-size="11" fill="var(--ink)">Your limit ${clockTime(limit)}</text>`;
  const pts = state.points;
  if (pts.length > 1) {
    const path = pts.map((p, i) => `${i ? "L" : "M"}${x(p.t).toFixed(1)},${y(p.a).toFixed(1)}`).join("");
    svg += `<path d="${path}L${x(last).toFixed(1)},${y(0)}L${x(pts[0].t).toFixed(1)},${y(0)}Z" fill="var(--accent)" opacity="0.08"/>`;
    svg += `<path d="${path}" fill="none" stroke="var(--accent)" stroke-width="2.2" stroke-linejoin="round"/>`;
  }
  const best = bestWithin(pts, limit);
  if (best) {
    const anchor = x(best.t) > W - 170 ? "end" : "start";
    svg += `<circle cx="${x(best.t)}" cy="${y(best.a)}" r="6" fill="var(--accent)" stroke="var(--panel)" stroke-width="2"/>`;
    svg += `<text x="${x(best.t) + (anchor === "end" ? -10 : 10)}" y="${y(best.a) - 10}" text-anchor="${anchor}" font-size="12.5" font-weight="600" fill="var(--ink)">Best ${pct(best.a)} at ${shortTime(best.t)}</text>`;
  }
  state.geometry = { W, L, R, maxT };
  const hovered = pts[state.hover];
  if (hovered) {
    const hx = x(hovered.t), hy = y(hovered.a);
    const lines = [`${pct(hovered.a)} accuracy`, `at ${shortTime(hovered.t)}`, hovered.epoch == null ? "final result" : `epoch ${hovered.epoch}`];
    const boxW = 128, boxH = 58;
    const bx = hx + 12 + boxW > W - R ? hx - 12 - boxW : hx + 12;
    const by = Math.min(Math.max(hy - boxH / 2, T), H - B - boxH);
    svg += `<line x1="${hx}" x2="${hx}" y1="${T}" y2="${H - B}" stroke="var(--ink)" stroke-dasharray="2 3" opacity="0.6"/>`;
    svg += `<circle cx="${hx}" cy="${hy}" r="4.5" fill="var(--panel)" stroke="var(--accent)" stroke-width="2.2"/>`;
    svg += `<rect x="${bx}" y="${by}" width="${boxW}" height="${boxH}" rx="6" fill="var(--panel)" stroke="var(--line)"/>`;
    lines.forEach((line, i) => {
      svg += `<text x="${bx + 10}" y="${by + 19 + i * 15}" font-size="${i ? 11.5 : 13}" font-weight="${i ? 400 : 600}" fill="${i ? "var(--muted)" : "var(--ink)"}" font-family="IBM Plex Mono, monospace">${line}</text>`;
    });
  }
  $("chart").innerHTML = svg;
  const now = pts.length ? pts[pts.length - 1] : null;
  $("s-now").textContent = now ? pct(now.a) : "–";
  $("s-best").textContent = best ? pct(best.a) : "–";
  $("s-when").textContent = best ? shortTime(best.t) : "–";
  $("s-epoch").textContent = state.epoch ?? "–";
}

function hoverAt(event) {
  const pts = state.points;
  if (!pts.length || !state.geometry) return;
  const { W, L, R, maxT } = state.geometry;
  const box = $("chart").getBoundingClientRect();
  const t = (((event.clientX - box.left) * W) / box.width - L) / (W - L - R) * maxT;
  let nearest = 0;
  pts.forEach((p, i) => { if (Math.abs(p.t - t) < Math.abs(pts[nearest].t - t)) nearest = i; });
  setHover(nearest);
}

function stepHover(event) {
  const pts = state.points;
  if (!pts.length) return;
  const moves = { ArrowLeft: -1, ArrowRight: 1, Home: -Infinity, End: Infinity };
  if (!(event.key in moves)) return;
  event.preventDefault();
  const from = state.hover ?? (moves[event.key] > 0 ? -1 : pts.length);
  setHover(Math.min(pts.length - 1, Math.max(0, from + moves[event.key])));
}

function setHover(index) {
  if (state.hover === index) return;
  state.hover = index;
  draw();
}

function setRunning(running) {
  state.running = running;
  $("train").disabled = running;
  $("stop").hidden = !running;
  for (const el of $("form").querySelectorAll("select, input")) el.disabled = running;
  if (!running) syncAvailability();
  $("dot").className = running ? "dot live" : "dot";
}

async function start() {
  if (state.running) return;
  state.points = [];
  state.hover = null;
  state.epoch = null;
  state.bestVal = Infinity;
  state.limit = +$("limit").value;
  state.run = { dataset: $("dataset").value, person: +$("person").value, model: modelKey(), limit: state.limit };
  $("note").hidden = true;
  $("summary").textContent = "Accuracy is measured on trials the model never trained on.";
  $("banner").hidden = state.mode !== "replay";
  $("term").innerHTML = "";
  setRunning(true);
  draw();
  if (state.mode === "replay") return replay(recording());
  try {
    const response = await fetch("api/runs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ dataset: state.run.dataset, person: state.run.person, model: state.run.model, time_limit: state.limit }),
    });
    const body = await response.json();
    if (!response.ok) throw new Error(body.error ?? "the server refused the run");
    state.run.id = body.id;
    state.stream = new EventSource(`api/runs/${body.id}/events`);
    state.stream.onmessage = (message) => handle(JSON.parse(message.data));
    state.stream.onerror = () => {
      if (state.running) handle({ type: "error", message: "lost the connection to the app. Is python app.py still running?" });
    };
  } catch (err) {
    handle({ type: "error", message: err.message });
  }
}

async function stop() {
  if (!state.running) return;
  if (state.mode === "replay") {
    clearTimeout(state.replayTimer);
    finish(true);
    return;
  }
  $("stop").disabled = true;
  await fetch(`api/runs/${state.run.id}/stop`, { method: "POST" }).catch(() => {});
}

function replay(run) {
  const events = run.events;
  let index = 0;
  let previous = 0;
  const next = () => {
    if (!state.running) return;
    if (index >= events.length) return finish(false);
    const event = events[index++];
    if (event.type === "epoch" && event.t > state.limit && state.limit < run.time_limit) {
      return finish(false, `stopped at your ${clockTime(state.limit)} limit`);
    }
    handle(event);
    if (event.type === "done" || event.type === "error") return;
    const upcoming = events[index];
    const wait = upcoming?.t !== undefined ? Math.max(0, upcoming.t - (event.t ?? previous)) : 0.35;
    if (event.t !== undefined) previous = event.t;
    state.replayTimer = setTimeout(next, (wait * 1000) / state.speed);
  };
  next();
}

function handle(event) {
  const run = state.run;
  switch (event.type) {
    case "start": {
      state.chance = event.chance;
      state.classes = event.classes;
      state.activeFile = "live.py";
      renderFiles();
      print(`<span class="cmd">$ ${command(state.run.limit)}</span>`);
      status(`Starting ${modelName(run.model)} on ${event.title}, person ${event.person}`);
      break;
    }
    case "loading": {
      const d = dataset();
      const first = state.mode === "live" && !d.downloaded;
      print(`<span class="t-dim">load</span>    ${esc(event.dataset)}, person ${run.person}${first ? " · downloading for the first time, this can take a few minutes" : ""}`);
      step(MARKERS.loading, first ? `Downloading <b>${esc(event.dataset)}</b> for the first time` : `Loading <b>${esc(event.dataset)}</b> recordings`);
      status(first ? "Downloading the dataset (first time only)…" : "Loading recordings…");
      break;
    }
    case "loaded": {
      if (state.mode === "live") dataset().downloaded = true;
      print(`<span class="t-dim">data</span>    ${event.trials} trials · ${event.channels} channels · ${event.samples} samples each`);
      print(`<span class="t-dim">split</span>   train ${event.train} · validate ${event.val} · test ${event.test}`);
      step(MARKERS.loaded, `Splitting trials: <b>${event.train}</b> to learn from, <b>${event.val}</b> to check progress, <b>${event.test}</b> kept unseen`);
      if (run.model === "eegnet") markLine(MARKERS.eegnet);
      break;
    }
    case "epoch": {
      state.epoch = event.epoch;
      state.points.push({ t: event.t, a: event.test_accuracy, epoch: event.epoch });
      const marker = event.improved ? MARKERS.improved : [MARKERS.backprop, MARKERS.forward, MARKERS.validate][event.epoch % 3];
      const text = event.improved
        ? `Epoch <b>${event.epoch}</b>: validation loss improved, so these weights are kept`
        : event.epoch % 3 === 1 ? `Epoch <b>${event.epoch}</b>: EEG flows through the network's layers` : `Epoch <b>${event.epoch}</b>: adjusting weights to reduce the loss`;
      print(`<span class="t-dim">epoch</span> ${String(event.epoch).padStart(4)}  train loss ${event.train_loss.toFixed(3)}  val loss ${event.val_loss.toFixed(3)}  test <span class="t-acc">${pct(event.test_accuracy)}</span>${event.improved ? '  <span class="t-good">↑ best so far, weights saved</span>' : ""}`);
      step(marker, text);
      status(`Training ${modelName(run.model)} · epoch ${event.epoch} · ${clockTime(event.t)} elapsed`);
      draw();
      break;
    }
    case "kept": {
      print(`<span class="t-good">kept</span>    weights from epoch ${event.epoch} (lowest validation loss): test <span class="t-acc">${pct(event.test_accuracy)}</span>`);
      step(MARKERS.kept, `Restoring the weights from epoch <b>${event.epoch}</b>, the one with the lowest validation loss`);
      $("note").textContent = `The app keeps the weights from epoch ${event.epoch}, picked by the lowest loss on the validation trials. They score ${pct(event.test_accuracy)} on the unseen test trials. The curve shows every epoch's test accuracy so you can watch it learn; it isn't used to pick the model.`;
      $("note").hidden = false;
      break;
    }
    case "step": {
      const name = (label) => CLASS_NAMES[state.classes[label]] ?? `class ${label + 1}`;
      if (event.stage === "filter") {
        print(`<span class="t-dim">filter</span>  band ${String(event.band[0]).padStart(2)}–${event.band[1]} Hz`);
        step(MARKERS.filter, `Filtering the EEG into the <b>${event.band[0]}–${event.band[1]} Hz</b> band`);
      } else if (event.stage === "csp" && event.label !== undefined) {
        print(`<span class="t-dim">csp</span>     ${name(event.label)} vs the rest: 9 bands × 4 filters`);
        step(MARKERS.fbcspCsp, `Learning <b>spatial filters</b> that pick out ${name(event.label)}`);
      } else if (event.stage === "csp") {
        print(`<span class="t-dim">csp</span>     ${event.n_filters} spatial filters`);
        step(MARKERS.csp, `Learning <b>${event.n_filters} spatial filters</b> that separate the movements`);
      } else if (event.stage === "lda") {
        print(`<span class="t-dim">lda</span>     linear classifier on log-power features`);
        step(MARKERS.lda, "Fitting a <b>linear classifier</b> on the filtered signal power");
      } else if (event.stage === "tune") {
        print(`<span class="t-dim">tune</span>    C = ${String(event.c).padEnd(6)} validation accuracy ${pct(event.val_accuracy)}`);
        step(MARKERS.tune, `Trying regularization <b>C = ${event.c}</b> on the validation trials`);
      }
      status(`Training ${modelName(run.model)} · ${shortTime(event.t)} elapsed`);
      break;
    }
    case "result": {
      state.points.push({ t: event.t, a: event.test_accuracy, epoch: null });
      print(`<span class="t-good">test</span>    accuracy on unseen trials: <span class="t-acc">${pct(event.test_accuracy)}</span>`);
      step(MARKERS.result, "Scoring the model on the <b>unseen test trials</b>");
      draw();
      break;
    }
    case "done":
      finish(event.stopped);
      break;
    case "error":
      print(`<span class="t-bad">error</span>   ${esc(event.message)}`);
      $("banner").className = "banner error";
      $("banner").textContent = `Training failed: ${event.message}`;
      $("banner").hidden = false;
      finish(true, null, true);
      break;
  }
}

function step(marker, text) {
  $("now").innerHTML = text;
  markLine(marker);
}

function status(text) {
  $("status").textContent = text;
}

function finish(stopped, reason = null, failed = false) {
  state.stream?.close();
  state.stream = null;
  $("stop").disabled = false;
  setRunning(false);
  if (failed) {
    $("dot").className = "dot failed";
    status("Stopped because of an error.");
    return;
  }
  const run = state.run;
  const best = bestWithin(state.points, run.limit);
  const elapsed = state.points.length ? state.points[state.points.length - 1].t : 0;
  print(`<span class="t-good">done</span>    ${best ? `best ${pct(best.a)} at ${shortTime(best.t)}` : "no results"}${reason ? ` · ${reason}` : stopped ? " · stopped early" : ""}`);
  status(stopped ? `Stopped after ${shortTime(elapsed)}.` : `Finished after ${shortTime(elapsed)}.`);
  draw();
  if (!best) return;
  $("summary").innerHTML = `Within your ${clockTime(run.limit)} limit, ${esc(modelName(run.model))} reached <strong>${pct(best.a)}</strong>, first hitting it at <strong>${shortTime(best.t)}</strong>. Guessing would score ${pct(state.chance)}.`;
  $("now").innerHTML = `Finished. Best accuracy <b>${pct(best.a)}</b>, reached at <b>${shortTime(best.t)}</b>.`;
  const title = state.options.datasets.find((d) => d.key === run.dataset).title;
  const row = `<tr><td>${esc(title)}</td><td>${run.person}</td><td>${esc(modelName(run.model))}</td><td class="num">${pct(best.a)}</td><td class="num">${shortTime(best.t)}</td><td class="num">${clockTime(run.limit)}</td></tr>`;
  const body = $("runs");
  if (body.querySelector(".empty")) body.innerHTML = "";
  body.insertAdjacentHTML("afterbegin", row);
}

init().catch((err) => {
  const banner = $("banner");
  banner.className = "banner error";
  banner.textContent = `The page couldn't start: ${err.message}. Start the app with python app.py, or open the hosted page.`;
  banner.hidden = false;
});
