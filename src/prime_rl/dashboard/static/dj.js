/* prime-rl dashboard, DJ tab: the Task mix of an RL run, step by step. What plays now and what is queued, whether
   training is healthy, which sources give a learning signal, and how plans, weights and reward shaping changed.
   Reads GET /api/runs/<run>/dj (prime_rl.dashboard.dj). Uses app.js's $, esc, api, state and emptyState.
   This package is fully AI-generated and maintained by agents - it is not meant to be read or edited by humans. Change it by asking an agent, and verify through the browser smoke tests. */

(() => {
  /* one hue per source, in config order; these ten pass the dataviz palette check pairwise in this order on the
     dashboard's #1c1c1c surface. An eleventh source and on are grey rather than a recycled hue */
  const SOURCE_COLORS = ["#ac7ad5", "#4862ce", "#53ac3e", "#00805f", "#bf890d", "#a25325", "#ca4b85", "#1fa5bb", "#216db2", "#1fa9a0"];
  const OVERFLOW_COLOR = "#7a7a7a";
  const MODE_NAMES = {
    didnt_find: "didn't find", declared_success: "declared success", gave_up: "gave up",
    unfinished_write: "write not done", botched_write: "botched write", unfinished: "unfinished", other: "other",
  };
  const MODE_LABELS = {
    didnt_find: "look_alikes, widening, cross_tool_hop, hops:1", declared_success: "hard, hops:1", gave_up: "— (no label)",
    unfinished_write: "kind=write", botched_write: "kind=write", unfinished: "— (reward shaping)", other: "—",
  };
  const TREND_STEPS = 16;
  const dj = { run: null, data: null, sel: null, histMode: "planned", busy: false, open: new Set() };
  const prefsKey = "prl-dash-dj";
  let djPrefs = {};
  try { djPrefs = JSON.parse(localStorage.getItem(prefsKey) || "{}"); } catch (_) { djPrefs = {}; }
  const savePrefs = () => { try { localStorage.setItem(prefsKey, JSON.stringify(djPrefs)); } catch (_) { /* private mode */ } };

  /* ------------------------------------------------------------ helpers */
  const sum = (xs) => xs.reduce((a, b) => a + b, 0);
  const pct = (x, d = 1) => (x == null || Number.isNaN(x) ? "—" : `${(x * 100).toFixed(d)}%`);
  const p1 = (x) => { const v = Math.round(x * 1000) / 10; return Number.isInteger(v) ? v.toFixed(0) : v.toFixed(1); };
  const fmt = (format, v) => (v == null ? "—" : format === "pct" ? pct(v) : format === "num4" ? v.toFixed(4) : v.toFixed(2));
  const hhmm = (iso) => (iso ? String(iso).slice(11, 16) : "");
  const cap = (s) => (s ? s.charAt(0).toUpperCase() + s.slice(1) : s);
  const colorOf = (i) => (i < SOURCE_COLORS.length ? SOURCE_COLORS[i] : OVERFLOW_COLOR);
  const sw = (i) => `<span class="dj-sw" style="background:${colorOf(i)}"></span>`;
  const rows = (pairs) => pairs.map(([k, v]) => `<div class="tr"><span>${k}</span><span>${v}</span></div>`).join("");
  const enc = encodeURIComponent;
  const srcIndex = (name) => dj.data.sources.findIndex((s) => s.name === name);
  const step = (n) => dj.data.steps.find((s) => s.step === n);
  const primaryEval = () => dj.data.eval.sources.find((s) => s.name === dj.data.eval.primary) || { name: null, rows: [] };
  const evalSteps = () => new Set(dj.data.eval.sources.flatMap((s) => s.rows.map((r) => r.step)));

  let tipEl = null;
  function showTip(html, ev) {
    if (!tipEl) { tipEl = document.createElement("div"); tipEl.id = "dj-tip"; tipEl.hidden = true; document.body.append(tipEl); }
    tipEl.innerHTML = html; tipEl.hidden = false; moveTip(ev);
  }
  function moveTip(ev) {
    if (!tipEl) return;
    const pad = 14, r = tipEl.getBoundingClientRect();
    let x = ev.clientX + pad, y = ev.clientY + pad;
    if (x + r.width > innerWidth - 8) x = ev.clientX - r.width - pad;
    if (y + r.height > innerHeight - 8) y = ev.clientY - r.height - pad;
    tipEl.style.left = `${Math.max(4, x)}px`; tipEl.style.top = `${Math.max(4, y)}px`;
  }
  const hideTip = () => { if (tipEl) tipEl.hidden = true; };
  function bindTip(el, html) {
    el.addEventListener("mouseenter", (e) => showTip(html(), e));
    el.addEventListener("mousemove", moveTip);
    el.addEventListener("mouseleave", hideTip);
  }

  function spark(series, o) {
    const W = o.w, H = o.h, pad = 3;
    if (!series.length) return `<svg width="${W}" height="${H}"></svg>`;
    const xs = series.map((d) => d[0]), ys = series.map((d) => d[1]);
    const x0 = o.x0 ?? Math.min(...xs), x1 = o.x1 ?? Math.max(...xs);
    const lines = [o.alarm, o.watch].filter((v) => v != null);
    const lo = o.lo ?? Math.min(...ys, ...lines);
    let hi = o.hi ?? Math.max(...ys, ...lines);
    if (o.hi == null) hi += (hi - lo) * 0.08 || 0.01;
    const X = (v) => pad + ((v - x0) / (x1 - x0 || 1)) * (W - pad * 2 - 3);
    const Y = (v) => pad + (1 - (v - lo) / (hi - lo || 1)) * (H - pad * 2);
    const col = o.color || "var(--grey-5)";
    let g = "";
    if (o.band) g += `<rect x="${X(o.band[0])}" y="0" width="${Math.max(0, X(o.band[1]) - X(o.band[0]))}" height="${H}" fill="rgba(255,255,255,0.06)"/>`;
    if (o.alarm != null) g += `<line x1="0" x2="${W}" y1="${Y(o.alarm)}" y2="${Y(o.alarm)}" stroke="var(--negative)" stroke-width="1" opacity=".75"/>`;
    if (o.watch != null) g += `<line x1="0" x2="${W}" y1="${Y(o.watch)}" y2="${Y(o.watch)}" stroke="var(--timeout)" stroke-width="1" opacity=".6"/>`;
    g += `<path d="${series.map((p, k) => `${k ? "L" : "M"}${X(p[0]).toFixed(1)},${Y(p[1]).toFixed(1)}`).join("")}" fill="none" stroke="${col}" stroke-width="1.5" stroke-linejoin="round" stroke-linecap="round"/>`;
    if (o.dots) series.forEach((p) => (g += `<circle cx="${X(p[0])}" cy="${Y(p[1])}" r="2.2" fill="${col}"/>`));
    const last = series[series.length - 1];
    g += `<circle cx="${X(last[0])}" cy="${Y(last[1])}" r="3" fill="${col}" stroke="var(--grey-2)" stroke-width="1.5"/>`;
    return `<svg width="${W}" height="${H}" viewBox="0 0 ${W} ${H}" aria-hidden="true" style="display:block">${g}</svg>`;
  }

  /* ------------------------------------------------------------ wording */
  function statusPhrase(m) {
    const r = m.rule, Name = cap(m.name), lvl = m.status === "alarm" ? "alarm" : "watch";
    const at = (v) => fmt(m.format, v);
    if (!r) return `${Name} is ${at(m.now)}.`;
    if (r.kind === "ratio") return `${Name} is ${at(m.now)}, ${m.ratio.toFixed(1)}× its step-${m.start_step} value. The ${lvl} is at ${m.status === "alarm" ? r.alarm : r.watch}×.`;
    if (r.kind === "points") {
      const d = (m.now - m.start) * 100;
      return `${Name} is ${at(m.now)}, ${d >= 0 ? "+" : "−"}${Math.abs(d).toFixed(1)} points since step ${m.start_step}. The ${lvl} is at +${(m.status === "alarm" ? r.alarm : r.watch) * 100} points.`;
    }
    if (r.kind === "above") return `${Name} is ${at(m.now)}. The ${lvl} is at ${at(m.status === "alarm" ? r.alarm : r.watch)}.`;
    if (r.kind === "fraction") return `${Name} is ${at(m.now)}, under half its step-${m.start_step} value of ${at(m.start)}.`;
    if (r.kind === "drop") return `${Name} is ${at(m.now)}, ${((m.start - m.now) * 100).toFixed(1)} points below step ${m.start_step}. The alarm is at −${r.alarm * 100} points.`;
    return `${Name} is ${at(m.now)}.`;
  }
  function stuckText(s) {
    if (!s || !(s.per_call > 0)) return "off";
    return `a passing rollout loses ${p1(s.per_call)}% of its reward per stuck call after the first ${s.free}, up to ${p1(s.cap)}%`;
  }
  function unfinishedText(u) {
    return u && u.reward != null ? `cut-off and timed-out rollouts score ${u.reward}` : "cut-off rollouts keep their score and timed-out ones score 0, the default";
  }
  const weightsOf = (plan) => (plan && plan.plan && plan.plan.weights) || {};
  const labelWeights = (plan) => Object.entries(weightsOf(plan)).filter(([k]) => k.startsWith("label:")).sort().map(([k, v]) => `${k}=${v}`).join(",");
  const taskWeights = (plan) => Object.keys(weightsOf(plan)).filter((k) => k.startsWith("task:")).length;
  const sourcesOf = (plan) => JSON.stringify((plan && plan.plan && plan.plan.sources) || {});
  const tasksOf = (plan) => ((plan && plan.plan && plan.plan.tasks) || []).length;

  function changeLabel(n) {
    const st = step(n), pv = step(n - 1);
    if (!st) return "";
    const parts = [];
    if (st.deleted.length) parts.push(`plan #${st.deleted.join(", #")} deleted`);
    if (pv) {
      if (!st.plan && pv.plan && !st.deleted.length) parts.push("no plan");
      else if (st.plan && !pv.plan) parts.push(`plan #${st.plan.id}`);
      else if (st.plan && pv.plan) {
        if (sourcesOf(st.plan) !== sourcesOf(pv.plan)) parts.push("new shares");
        if (labelWeights(st.plan) !== labelWeights(pv.plan)) parts.push("new label weights");
        if (taskWeights(st.plan) !== taskWeights(pv.plan)) parts.push(taskWeights(st.plan) ? `${taskWeights(st.plan)} Tasks reweighted` : "Task weights off");
        if (tasksOf(st.plan) !== tasksOf(pv.plan)) parts.push(tasksOf(st.plan) ? `${tasksOf(st.plan)} Tasks cued` : "cue cleared");
      }
    }
    const shaping = dj.data.shaping.changes.filter((c) => c.step === n);
    for (const c of shaping) {
      if (c.change.stuck) parts.push(c.change.stuck.per_call > 0 ? `stuck penalty ${p1(c.change.stuck.per_call)}%` : "stuck penalty off");
      if (c.change.unfinished) parts.push(c.change.unfinished.reward != null ? `unfinished score ${c.change.unfinished.reward}` : "unfinished score default");
    }
    const out = parts.join(", ");
    return out && st.state === "queued" && st.plan ? `${st.plan.author}: ${out}` : out;
  }

  /* ------------------------------------------------------------ the page */
  function skeleton() {
    return `
      <div id="dj-desk">
        <section class="dj-pane dj-deck">
          <div class="dj-ph"><span class="dj-pt" id="dj-mx-t"></span><span class="dj-pm" id="dj-mx-meta"></span><span class="dj-key" id="dj-mx-key"></span></div>
          <div class="dj-strips-wrap"><div class="dj-strips" id="dj-strips"></div></div>
          <div class="dj-bus" id="dj-bus"></div>
        </section>
        <aside class="dj-side">
          <section class="dj-pane">
            <div class="dj-ph"><span class="dj-pt">Health</span></div>
            <div class="dj-line" id="dj-health-line"></div>
            <div class="dj-sparks" id="dj-sparks"></div>
            <div class="dj-shape" id="dj-shape-line"></div>
          </section>
          <section class="dj-pane">
            <div class="dj-ph"><span class="dj-pt">Pass@1</span></div>
            <div id="dj-pass-chart"></div>
            <div class="dj-tiles" id="dj-tiles"></div>
          </section>
        </aside>
        <section class="dj-pane dj-changes">
          <div class="dj-ph"><span class="dj-pt" id="dj-ch-t">Mix changes</span><span class="dj-pm">Click a step to open it.</span></div>
          <div id="dj-timeline"></div>
        </section>
        <section class="dj-pane dj-next">
          <div class="dj-ph"><span class="dj-pt" id="dj-nx-t"></span></div>
          <div id="dj-queued"></div>
        </section>
      </div>
      <div class="dj-more">Details</div>
      ${section("health", "Health metrics")}
      ${section("step", "Step")}
      ${section("sources", "Sources", "Every per-source number for the last finished steps")}
      ${section("history", "Mix history", "Shares, plans, shaping and evals, step by step")}
      ${section("log", "Plan log")}
      ${section("labels", "Pass@1 by label", "By interface, domain and provider, over the last policy versions")}
      ${section("modes", "Failure modes", "How training rollouts fail, by policy version")}
      ${section("eval", "Held-out eval")}
      ${section("builder", "Request builder", "Builds the curl command for a plan or a shaping change. Sends nothing.")}`;
  }
  function section(id, title, sub = "") {
    return `<details class="section dj-section" id="dj-d-${id}"${dj.open.has(id) ? " open" : ""}><summary><span id="dj-t-${id}">${esc(title)}</span><span class="dj-s2" id="dj-s2-${id}">${esc(sub)}</span><span class="sec-chev">›</span></summary><div class="dj-body-in" id="dj-${id}"></div></details>`;
  }
  const DETAILS = {
    health: () => renderHealthTable(), step: () => renderStep(), sources: () => renderSources(), history: () => renderHistory(),
    log: () => renderLog(), labels: () => renderLabels(), modes: () => renderModes(), eval: () => renderEval(), builder: () => renderBuilder(),
  };

  function render() {
    const body = $("#dj-body"), d = dj.data;
    if (!d) return;
    if (!d.available) {
      body.innerHTML = emptyState("no DJ view for this run", d.reason || "");
      return;
    }
    if (!$("#dj-desk")) {
      body.innerHTML = skeleton();
      for (const id of Object.keys(DETAILS)) {
        $(`#dj-d-${id}`).addEventListener("toggle", (e) => {
          if (e.target.open) { dj.open.add(id); DETAILS[id](); } else dj.open.delete(id);
        });
      }
    }
    if (dj.sel == null || dj.sel > d.steps.length) dj.sel = d.dispatching;
    renderStrips(); renderBus(); renderHealth(); renderPass(); renderTimeline(); renderQueued(); renderSummaries();
    for (const id of dj.open) if (id !== "builder" || !$("#dj-builder").innerHTML) DETAILS[id]();
    renderStatus();
  }

  function renderStatus() {
    const d = dj.data, parts = [];
    if (d.progress && d.progress.total && d.progress.read < d.progress.total)
      parts.push(`reading the trace stream: ${d.progress.read.toLocaleString("en-US")} of ${d.progress.total.toLocaleString("en-US")} rollouts`);
    if (d.live) {
      if (d.live.error) parts.push(`control API: ${esc(d.live.error)}`);
      else if (d.live.dispatching) parts.push(`live: step ${d.live.dispatching.step} dispatching, ${d.live.dispatching.groups} groups out`);
    }
    $("#dj-status").innerHTML = parts.join(" · ");
  }

  function renderSummaries() {
    const d = dj.data;
    const counts = { alarm: 0, watch: 0, ok: 0 };
    d.health.forEach((m) => { if (counts[m.status] != null) counts[m.status]++; });
    $("#dj-s2-health").textContent = `${counts.alarm} alarm, ${counts.watch} on watch, ${counts.ok} fine`;
    const sets = d.plan_events.filter((e) => e.action === "set").length, dels = d.plan_events.filter((e) => e.action === "delete").length;
    $("#dj-s2-log").textContent = `${sets} plans, ${dels} deleted, ${d.shaping.changes.length} shaping changes`;
    const ev = primaryEval(), scored = ev.rows.filter((r) => r.pass1 != null);
    const last = scored[scored.length - 1], first = scored[0];
    $("#dj-s2-eval").textContent = last
      ? `${ev.name}: ${pct(last.pass1)} at step ${last.step}${first !== last ? `, ${last.pass1 >= first.pass1 ? "up" : "down"} ${Math.abs((last.pass1 - first.pass1) * 100).toFixed(1)} points from step ${first.step}` : ""}.`
      : "No eval yet.";
    if (d.window) $("#dj-s2-sources").textContent = `Every per-source number for steps ${d.window[0]}–${d.window[1]}`;
  }

  /* ------------------------------------------------------------ the strips */
  function renderStrips() {
    const d = dj.data, now = step(d.dispatching);
    $("#dj-mx-t").textContent = `Step ${d.dispatching} mix`;
    $("#dj-mx-meta").innerHTML = (now && now.plan ? `Plan <b>#${now.plan.id}</b> by ${esc(now.plan.author || "?")}.` : "No plan: the config's mix.") +
      (d.window ? ` Pass@1 and signal cover steps ${d.window[0]}–${d.window[1]}.` : " No finished step yet.");
    $("#dj-mx-key").innerHTML = `<span><i class="dj-sw" style="background:var(--grey-5);width:12px;height:6px"></i>now</span><span><i class="dj-sw" style="border:1px dashed var(--accent);width:12px;height:7px;background:none"></i>next, if different</span><span><i class="dj-tri"></i>config</span>`;
    const fmax = Math.max(0.3, Math.ceil(Math.max(...d.sources.flatMap((s) => [s.share, s.share_now, s.share_next])) * 10 + 0.5) / 10);
    const Y = (v) => (1 - Math.min(v, fmax) / fmax) * 100;
    const ticks = []; for (let t = 0; t <= fmax + 1e-9; t += 0.1) ticks.push(t);
    $("#dj-strips").innerHTML = d.sources.map((s, i) => {
      const k = s.name.lastIndexOf("-"), n1 = k > 0 ? s.name.slice(0, k) : s.name, n2 = k > 0 ? s.name.slice(k + 1) : "";
      const lowN = s.groups < 20;
      return `<div class="dj-strip" data-i="${i}" tabindex="0">
        <div class="nm"><i style="background:${colorOf(i)}"></i><span title="${esc(s.name)}">${esc(n1)}<small>${esc(n2)}</small></span></div>
        <div class="mt"><span>pass@1</span><b>${pct(s.pass1, 0)}</b></div>
        <div class="sp" data-k="pass1"></div>
        <div class="mt"><span title="share of groups with a non-zero advantage">signal</span><b class="${lowN ? "lowv" : ""}" title="${s.groups} groups${lowN ? ": too few to decide on" : ""}">${pct(s.signal, 0)}</b></div>
        <div class="meter"><i style="width:${(s.signal || 0) * 100}%"></i></div>
        <div class="fader" aria-hidden="true">
          <div class="scale">${ticks.map((t) => `<span style="top:${Y(t)}%">${Math.round(t * 100)}</span>`).join("")}</div>
          <div class="trk">${ticks.map((t) => `<b style="top:${Y(t)}%"></b>`).join("")}
            <span class="lvl" style="height:${100 - Y(s.share_now)}%;background:${colorOf(i)}"></span>
            <span class="cfg" style="top:${Y(s.share)}%"></span>
            ${Math.abs(s.share_next - s.share_now) > 0.0005 ? `<span class="cap ghost" style="top:${Y(s.share_next)}%"></span>` : ""}
            <span class="cap" style="top:${Y(s.share_now)}%;background:${colorOf(i)}"></span></div>
        </div>
        <div class="val">${pct(s.share_now)}</div>
        <div class="sub">${s.named_now ? `<span class="chg">${s.share_now > s.share ? "up" : "down"} from ${pct(s.share)}</span>` : `config ${pct(s.share)}`}</div>
        <div class="foot"><span class="cue ${s.cued ? "on" : ""}" title="${s.cued ? `${s.cued} Task${s.cued > 1 ? "s" : ""} from this source run first this step` : "no Tasks cued from this source"}">CUE${s.cued ? ` ${s.cued}` : ""}</span></div>
      </div>`;
    }).join("");
    document.querySelectorAll("#dj-strips .dj-strip").forEach((el) => {
      const s = d.sources[+el.dataset.i], host = el.querySelector(".sp");
      host.innerHTML = spark(s.trend_pass1, { w: Math.max(80, host.clientWidth || 96), h: 22, lo: 0, hi: 1, color: colorOf(+el.dataset.i) });
      bindTip(el, () => `<div class="th">${esc(s.name)}</div>${rows([
        [`share, step ${d.dispatching}`, pct(s.share_now)], [`share, step ${d.dispatching + 1}`, pct(s.share_next)], ["config", pct(s.share)],
        [d.window ? `groups, steps ${d.window[0]}–${d.window[1]}` : "groups", s.groups], ["groups with signal", pct(s.signal)], ["train pass@1", pct(s.pass1)],
        ["turns per rollout", s.turns == null ? "—" : s.turns.toFixed(1)], ["sampled tokens per rollout", s.tokens == null ? "—" : Math.round(s.tokens).toLocaleString("en-US")],
        ["unfinished rate", pct(s.unfinished)], ["signal groups per million tokens", s.signal_per_mtok == null ? "—" : s.signal_per_mtok.toFixed(1)], ["Tasks cued this step", s.cued],
      ])}`);
    });
  }

  /* ------------------------------------------------------------ the label-weight bus */
  function knob(m) {
    const a = (v) => -135 + (270 * Math.min(v, 3)) / 3, P = (deg, r) => [17 + r * Math.sin((deg * Math.PI) / 180), 17 - r * Math.cos((deg * Math.PI) / 180)];
    const arc = (d0, d1, r) => { const [x0, y0] = P(d0, r), [x1, y1] = P(d1, r); return `M${x0.toFixed(2)},${y0.toFixed(2)} A${r},${r} 0 ${Math.abs(d1 - d0) > 180 ? 1 : 0} ${d1 > d0 ? 1 : 0} ${x1.toFixed(2)},${y1.toFixed(2)}`; };
    const [px, py] = P(a(m), 9), [tx0, ty0] = P(a(1), 15), [tx1, ty1] = P(a(1), 11);
    return `<span class="knob"><svg width="34" height="34" viewBox="0 0 34 34" aria-hidden="true"><path d="${arc(-135, 135, 13)}" fill="none" stroke="var(--grey-3)" stroke-width="3" stroke-linecap="round"/>${m !== 1 ? `<path d="${arc(a(1), a(m), 13)}" fill="none" stroke="var(--accent)" stroke-width="3" stroke-linecap="round"/>` : ""}<line x1="${tx0}" y1="${ty0}" x2="${tx1}" y2="${ty1}" stroke="var(--grey-5)" stroke-width="1"/><circle cx="17" cy="17" r="8" fill="#262626"/><line x1="17" y1="17" x2="${px}" y2="${py}" stroke="var(--grey-6)" stroke-width="1.6" stroke-linecap="round"/></svg><b>×${m}</b></span>`;
  }
  function renderBus() {
    const d = dj.data, X = (v) => (Math.min(v, 1) / 1) * 100;
    const mods = d.weights.map((w) => {
      const name = w.selector.replace(/^label:/, "");
      const bars = d.sources.map((s, i) => {
        const p = w.by_source[s.name];
        return { i, p };
      });
      const pmax = Math.max(...bars.map((b) => b.p || 0)) || 1;
      const known = w.share_before != null && w.share_now != null;
      const lo = known ? Math.min(w.share_before, w.share_now) : 0, hi = known ? Math.max(w.share_before, w.share_now) : 0;
      return `<div class="fx">
        <div class="fx-h"><span class="fx-n" title="${esc(w.selector)}">${esc(name).replace(":", ":<wbr>")}</span>${knob(w.multiplier)}</div>
        ${w.label == null ? `<div class="fx-l muted">The trace stream doesn't carry this field, so its share isn't counted.</div>` : `
        <div class="fx-l"><b>${pct(w.share_before)} → ${pct(w.share_now)}</b> <span class="k">of rollouts</span></div>
        ${known ? `<div class="db"><span class="ln" style="left:${X(lo)}%;width:${X(hi) - X(lo)}%"></span><span class="c" style="left:${X(w.share_before)}%"></span><span class="p" style="left:${X(w.share_now)}%"></span></div>` : ""}`}
        <div class="fx-l muted">On since step ${w.since}${w.author ? ` (${esc(w.author)})` : ""}</div>
        ${w.label == null ? "" : `<div class="fx-l fx-t" title="share of each source's rollouts whose Task carries the label, scaled to the largest">Where it applies</div>
        <div class="lands">${bars.map((b) => `<i data-i="${b.i}" data-p="${b.p ?? ""}" style="height:${Math.max(6, ((b.p || 0) / pmax) * 100)}%;background:${colorOf(b.i)}"></i>`).join("")}</div>`}
      </div>`;
    });
    const tw = d.task_weights;
    if (tw.count) {
      mods.push(`<div class="fx">
        <div class="fx-h"><span class="fx-n">Task weights</span>${knob(tw.multiplier)}</div>
        <div class="fx-l"><b>${tw.count} Task${tw.count > 1 ? "s" : ""}</b> <span class="k">at ×${tw.multiplier}</span></div>
        <div class="fx-l muted">On since step ${tw.since}</div>
        <div class="fx-l fx-t">Where</div>
        ${Object.entries(tw.by_source).map(([env, n]) => `<div class="fx-l">${srcIndex(env) >= 0 ? sw(srcIndex(env)) : ""} ${esc(env)} <b>${n}</b></div>`).join("")}
      </div>`);
    }
    const pools = d.pools ? Object.entries(d.pools).map(([k, v]) => `${k} ${v}`).join(", ") : null;
    const poolWeights = Object.entries(d.pool_weights || {}).map(([k, v]) => `${k} ×${v}`).join(", ");
    $("#dj-bus").innerHTML = `<div class="dj-ph"><span class="dj-pt">Label weights</span><span class="dj-pm">Boost or cut Tasks with a label, in every source. They don't change source shares. Before → now: rollouts under the config's mix, then under this weight.</span></div>
      <div class="fx-grid">${mods.join("") || `<div class="fx-l muted">No label or Task weights in step ${d.dispatching}'s plan.</div>`}</div>
      <div class="dj-bus-foot">Pool weights: <b>${poolWeights || "none"}</b>.${pools ? ` Difficulty pools weigh Tasks ${esc(pools)}.` : ""}</div>`;
    document.querySelectorAll("#dj-bus .lands i").forEach((el) => {
      const s = d.sources[+el.dataset.i];
      bindTip(el, () => `<div class="th">${esc(s.name)}</div>${rows([["rollouts whose Task carries the label", el.dataset.p === "" ? "—" : pct(+el.dataset.p)]])}`);
    });
  }

  /* ------------------------------------------------------------ health */
  function renderHealth() {
    const d = dj.data, judged = d.health.filter((m) => m.status !== "none");
    const alarms = judged.filter((m) => m.status === "alarm"), watches = judged.filter((m) => m.status === "watch");
    let line;
    if (alarms.length) line = `<span class="badge dj-alarm">alarm</span><span>${esc(statusPhrase(alarms[0]))}</span>`;
    else if (watches.length) line = `<span class="badge dj-watch">watch</span><span>${esc(statusPhrase(watches[0]))}</span>`;
    else line = `<span class="badge dj-ok">ok</span><span>${judged.length ? `All ${judged.length} metrics are inside their limits.` : "No metrics yet."}</span>`;
    const rest = [...alarms.slice(1), ...watches.slice(alarms.length ? 0 : 1)];
    if (rest.length) line += `<span class="sub">Also on watch: ${rest.map((m) => m.short).join(" and ")}.</span>`;
    $("#dj-health-line").innerHTML = line;
    $("#dj-sparks").innerHTML = d.health.map((m, n) => `<button class="dj-sp ${m.status}" data-n="${n}" aria-label="${esc(m.name)}"><span class="l">${esc(m.short)}<b>${m.status === "none" ? "—" : fmt(m.format, m.now)}</b></span><span class="s"></span></button>`).join("");
    document.querySelectorAll("#dj-sparks .dj-sp").forEach((el) => {
      const m = d.health[+el.dataset.n], host = el.querySelector(".s");
      host.innerHTML = spark(m.series, sparkOpts(m, Math.max(90, host.clientWidth || 140), 26));
      bindTip(el, () => `<div class="th">${esc(m.name)}</div>${m.status === "none" ? "no data yet" : rows([["now", `${fmt(m.format, m.now)} (step ${m.now_step})`], [`step ${m.start_step}`, fmt(m.format, m.start)], ["status", m.status]])}`);
      el.addEventListener("click", () => { const det = $("#dj-d-health"); det.open = true; det.scrollIntoView({ behavior: "smooth", block: "start" }); });
    });
    const a = d.shaping.active, since = d.shaping.since, pend = d.shaping.pending;
    const stuckOn = a.stuck && a.stuck.per_call > 0;
    $("#dj-shape-line").innerHTML = `<div class="k">Reward shaping</div>
      <div>Stuck-call penalty: <span class="on">${stuckOn ? cap(stuckText(a.stuck)) : "off"}</span>${stuckOn && since.stuck ? `. On since step ${since.stuck.step}.` : "."}</div>
      ${pend.filter((c) => c.change.stuck).map((c) => `<div class="next">${esc(cap(c.author || "someone"))} sets it to ${c.change.stuck.per_call != null ? `${p1(c.change.stuck.per_call)}% per stuck call` : "new limits"} from step ${c.step}.</div>`).join("")}
      <div>Unfinished rollouts: <span class="on">${unfinishedText(a.unfinished)}</span>.</div>
      ${pend.filter((c) => c.change.unfinished).map((c) => `<div class="next">${esc(cap(c.author || "someone"))} changes it from step ${c.step}: ${unfinishedText(c.change.unfinished)}.</div>`).join("")}`;
  }
  function sparkOpts(m, w, h) {
    const col = m.status === "alarm" ? "var(--negative)" : m.status === "watch" ? "var(--timeout)" : "var(--grey-5)";
    const isEval = m.key === "heldout_pass1";
    return { w, h, alarm: m.alarm_line, watch: m.watch_line, dots: isEval, color: col, lo: m.rule && m.rule.kind !== "fraction" && m.rule.kind !== "drop" ? 0 : undefined,
      band: m.key === "stuck_calls" && dj.data.shaping.since.stuck ? [dj.data.shaping.since.stuck.step, dj.data.dispatching] : null };
  }
  function renderHealthTable() {
    const d = dj.data;
    let h = `<table class="dj-table" style="min-width:900px"><thead><tr><th>metric</th><th class="r">now</th><th class="r">start</th><th>limit</th><th>status</th><th>series</th></tr></thead><tbody>`;
    d.health.forEach((m, n) => {
      const r = m.rule;
      const limit = !r ? "—" : r.kind === "above" ? `${r.watch != null ? `watch ${fmt(m.format, r.watch)}, ` : ""}alarm above ${fmt(m.format, r.alarm)}`
        : r.kind === "fraction" ? `alarm at half of step 1` : r.kind === "ratio" ? `watch ${r.watch}×, alarm ${r.alarm}× step 1`
        : r.kind === "points" ? `${r.watch != null ? `watch +${r.watch * 100}, ` : ""}alarm +${r.alarm * 100} points` : `alarm −${r.alarm * 100} points`;
      h += `<tr class="${m.status}"><td class="hi">${esc(m.name)}</td><td class="r hi">${m.status === "none" ? "—" : fmt(m.format, m.now)}</td><td class="r">${m.status === "none" ? "—" : `${fmt(m.format, m.start)} <span class="muted">s${m.start_step}</span>`}</td><td>${limit}</td><td class="st-${m.status}">${m.status === "none" ? "no data" : m.status}</td><td><span data-n="${n}" class="hs"></span></td></tr>`;
    });
    $("#dj-health").innerHTML = `<div class="dj-scroll">${h}</tbody></table></div><div class="dj-note">Mismatch KL, entropy and held-out pass@1 use the rl-dj skill's limits. Repetition, near-duplicate and silent calls use this page's lines, since the skill says to watch them "climbing".</div>`;
    document.querySelectorAll("#dj-health .hs").forEach((el) => { const m = d.health[+el.dataset.n]; el.innerHTML = spark(m.series, sparkOpts(m, 220, 28)); });
  }

  /* ------------------------------------------------------------ pass@1 and tiles */
  function renderPass() {
    const d = dj.data, host = $("#dj-pass-chart"), W = Math.max(240, host.clientWidth), H = 140, L = 32, Rr = 8, T = 8, B = 18;
    const evalLines = d.eval.sources.map((s) => ({ name: s.name, primary: s.name === d.eval.primary, rows: s.rows.filter((r) => r.pass1 != null) }));
    const xmax = Math.max(d.dispatching + 1, ...evalLines.flatMap((s) => s.rows.map((r) => r.step)), 10);
    const X = (s) => L + (s / xmax) * (W - L - Rr), Y = (v) => T + (1 - v) * (H - T - B);
    let g = "";
    for (const v of [0, 0.25, 0.5, 0.75, 1]) g += `<line class="${v ? "gl" : "bl"}" x1="${L}" x2="${W - Rr}" y1="${Y(v)}" y2="${Y(v)}"/><text class="ax" x="${L - 5}" y="${Y(v) + 3}" text-anchor="end">${v * 100}%</text>`;
    const tickStep = xmax > 60 ? 20 : xmax > 20 ? 10 : 5;
    for (let s = 0; s <= xmax; s += tickStep) g += `<text class="ax" x="${X(s)}" y="${H - 4}" text-anchor="middle">${s}</text>`;
    if (d.train_pass1.length) g += `<path d="${d.train_pass1.map((p, k) => `${k ? "L" : "M"}${X(p[0]).toFixed(1)},${Y(p[1]).toFixed(1)}`).join("")}" fill="none" stroke="var(--grey-5)" stroke-width="1.5" stroke-linejoin="round"/>`;
    for (const line of evalLines) {
      if (!line.rows.length) continue;
      const col = line.primary ? "var(--accent)" : "var(--grey-6)";
      g += `<path d="${line.rows.map((e, k) => `${k ? "L" : "M"}${X(e.step)},${Y(e.pass1)}`).join("")}" fill="none" stroke="${col}" stroke-width="${line.primary ? 2 : 1.5}" stroke-linejoin="round" ${line.primary ? "" : 'stroke-dasharray="4 3"'}/>`;
      line.rows.forEach((e) => (g += `<circle cx="${X(e.step)}" cy="${Y(e.pass1)}" r="${line.primary ? 3.5 : 3}" fill="${col}" stroke="var(--grey-2)" stroke-width="1.5"/>`));
    }
    const lt = d.train_pass1[d.train_pass1.length - 1];
    const evalLegend = evalLines.map((line) => {
      const le = line.rows[line.rows.length - 1];
      return `<span><span class="dj-sw" style="background:${line.primary ? "var(--accent)" : "var(--grey-6)"};height:2px"></span>${esc(line.name)}: ${le ? `${pct(le.pass1)} at step ${le.step}` : "no eval yet"}</span>`;
    });
    host.innerHTML = `<svg class="dj-svg" width="${W}" height="${H}" viewBox="0 0 ${W} ${H}" role="img" aria-label="Train pass@1 by step and each held-out source's pass@1 at evals">${g}</svg>
      <div class="dj-legend"><span><span class="dj-sw" style="background:var(--grey-5);height:2px"></span>Train: ${lt ? `${pct(lt[1], 0)} at steps ${Math.max(1, lt[0] - 2)}–${lt[0]}` : "no finished step"}</span>${evalLegend.join("") || "<span>No eval yet.</span>"}</div>`;
    const now = step(d.dispatching);
    $("#dj-tiles").innerHTML = `<div><span class="k">Groups with signal</span><span class="v">${pct(d.all.signal, 0)}</span><span class="s"> ${d.window ? `Steps ${d.window[0]}–${d.window[1]}.` : ""}${d.all.first_signal != null && d.first_window ? ` Was ${pct(d.all.first_signal, 0)} at steps ${d.first_window[0]}–${d.first_window[1]}.` : ""}</span></div>
      <div><span class="k">Step ${d.dispatching} so far</span><span class="v">${now ? now.finalized : 0}${d.batch_groups ? ` of ${d.batch_groups}` : ""}</span><span class="s"> groups finalized, ${now ? now.mixed : 0} with mixed rewards.</span></div>`;
  }

  /* ------------------------------------------------------------ timeline */
  function renderTimeline() {
    const d = dj.data, host = $("#dj-timeline"), W = Math.max(320, host.clientWidth);
    const Z = Math.min(d.steps.length, d.dispatching + 1), A = Math.max(1, Z - TREND_STEPS);
    $("#dj-ch-t").textContent = `Mix changes, steps ${A}–${Z}`;
    const L = 12, Rr = 12, lineY = 72, H = 106, X = (s) => L + ((s - A) / (Z - A || 1)) * (W - L - Rr);
    let g = `<line x1="${L}" x2="${W - Rr}" y1="${lineY}" y2="${lineY}" stroke="var(--grey-3)" stroke-width="1"/>`;
    const placed = [[], [], [], []];
    for (let s = A; s <= Z; s++) {
      const label = changeLabel(s);
      if (!label) continue;
      const x = X(s), wd = label.length * 6.4 + 6, end = x + wd > W - Rr;
      const span = end ? [x + 3 - wd, x + 3] : [x - 3, x - 3 + wd];
      let r = placed.findIndex((row) => row.every(([p, q]) => span[1] + 8 < p || span[0] > q + 8));
      if (r < 0) r = placed.length - 1;
      placed[r].push(span);
      const y = 11 + r * 13;
      g += `<line x1="${x}" x2="${x}" y1="${y + 3}" y2="${lineY - 7}" stroke="var(--hairline)"/>`;
      g += `<text class="tl-t ${s >= d.dispatching ? "hl" : ""}" x="${end ? x + 3 : x - 3}" y="${y}" text-anchor="${end ? "end" : "start"}">${esc(label)}</text>`;
    }
    const evaluated = evalSteps();
    for (let s = A; s <= Z; s++) {
      const st = step(s), x = X(s);
      if (!st) continue;
      if (s === d.dispatching) g += `<circle cx="${x}" cy="${lineY}" r="9" fill="none" stroke="var(--grey-6)" stroke-width="1"/>`;
      if (st.deleted.length) g += `<circle cx="${x}" cy="${lineY}" r="4.5" fill="var(--grey-2)" stroke="var(--negative)" stroke-width="1.5"/><path d="M${x - 2.5},${lineY - 2.5} l5,5 M${x + 2.5},${lineY - 2.5} l-5,5" stroke="var(--negative)" stroke-width="1.2"/>`;
      else if (!st.plan) g += `<circle cx="${x}" cy="${lineY}" r="4" fill="var(--grey-2)" stroke="var(--grey-4)" stroke-width="1.2"/>`;
      else {
        const col = st.plan.author === "claude" ? "var(--accent)" : "var(--grey-6)";
        g += st.state === "queued" ? `<circle cx="${x}" cy="${lineY}" r="5" fill="none" stroke="${col}" stroke-width="1.5" stroke-dasharray="2 2"/>` : `<circle cx="${x}" cy="${lineY}" r="5" fill="${col}" stroke="var(--grey-2)" stroke-width="1.5"/>`;
      }
      if (d.shaping.changes.some((c) => c.step === s)) g += `<rect x="${x - 3}" y="${lineY + 9}" width="6" height="6" fill="var(--grey-6)"/>`;
      const every = Z - A > 12 ? 2 : 1;
      const lab = s === d.dispatching ? "now" : evaluated.has(s) ? `${s} eval` : (s - A) % every === 0 ? String(s) : "";
      if (lab) g += `<text class="ax${s === d.dispatching || evaluated.has(s) ? "2" : ""}" x="${x}" y="${H - 4}" text-anchor="middle">${lab}</text>`;
      g += `<circle class="tl-hit" data-s="${s}" cx="${x}" cy="${lineY}" r="12" fill="transparent" style="cursor:pointer"/>`;
    }
    host.innerHTML = `<svg class="dj-svg" width="${W}" height="${H}" viewBox="0 0 ${W} ${H}" role="img" aria-label="Plan and shaping changes by step">${g}</svg>
      <div class="dj-legend"><span><span class="dj-sw" style="background:var(--accent);border-radius:50%"></span>plan by claude</span><span><span class="dj-sw" style="background:var(--grey-6);border-radius:50%"></span>plan by a person</span><span><span class="dj-sw" style="border:1px solid var(--grey-4);border-radius:50%;background:none"></span>no plan</span><span><span class="dj-sw" style="border:1px solid var(--negative);border-radius:50%;background:none"></span>plan deleted</span><span><span class="dj-sw" style="background:var(--grey-6);width:6px;height:6px"></span>shaping change</span></div>`;
    host.querySelectorAll(".tl-hit").forEach((el) => {
      const s = +el.dataset.s;
      bindTip(el, () => stepTip(s));
      el.addEventListener("click", () => selectStep(s, true));
    });
  }
  function stepTip(n) {
    const st = step(n), d = dj.data;
    if (!st) return `<div class="th">step ${n}</div>`;
    const pairs = d.sources.map((s, i) => [`${sw(i)}${esc(s.name)}`, pct(st.shares[s.name])]);
    pairs.push(["stuck-call penalty", st.shaping && st.shaping.stuck && st.shaping.stuck.per_call > 0 ? `${p1(st.shaping.stuck.per_call)}% per call` : "off"]);
    if (st.finalized) pairs.push(["groups finalized", `${st.finalized}, ${st.mixed} with mixed rewards`]);
    return `<div class="th">step ${n} · ${st.plan ? `plan #${st.plan.id} (${esc(st.plan.author || "?")})` : "no plan"}${st.state === "queued" ? " · queued" : ""}</div>${rows(pairs)}`;
  }

  /* ------------------------------------------------------------ next */
  function renderQueued() {
    const d = dj.data, n = d.next.step, e = d.next.plan, now = step(d.dispatching);
    $("#dj-nx-t").textContent = `Next: step ${n}`;
    let h;
    if (e) {
      const diff = [];
      const nowPlan = now && now.plan;
      diff.push(sourcesOf(nowPlan) === JSON.stringify(e.plan.sources || {}) && labelWeights(nowPlan) === labelWeights({ plan: e.plan }) ? "Same shares and label weights." : "Different shares or label weights.");
      if (tasksOf(nowPlan) && !(e.plan.tasks || []).length) diff.push(`No cued Tasks (step ${d.dispatching} cued ${tasksOf(nowPlan)}).`);
      if (taskWeights(nowPlan) > taskWeights({ plan: e.plan })) diff.push(`No Task weights (step ${d.dispatching} reweighted ${taskWeights(nowPlan)}).`);
      h = `<div class="dj-kv"><span>Plan <b>#${e.id}</b> by <b>${esc(e.author || "?")}</b>${e.time ? `, posted at ${hhmm(e.time)} UTC` : ""}.</span></div>
        ${e.note ? `<div class="dj-note-full dj-clamp">${esc(e.note)}</div>` : ""}
        <div class="dj-qrow"><span class="k">vs now</span><span>${diff.join(" ")}</span></div>`;
    } else h = `<div class="dj-kv"><span>Nothing queued: step ${n} runs on the config's mix.</span></div>`;
    for (const c of d.shaping.pending) {
      const what = [c.change.stuck ? `stuck-call penalty: ${stuckText(d.shaping.active.stuck ? { ...d.shaping.active.stuck, ...c.change.stuck } : c.change.stuck)}` : "", c.change.unfinished ? `unfinished rollouts: ${unfinishedText(c.change.unfinished)}` : ""].filter(Boolean).join("; ");
      h += `<div class="dj-qrow"><span class="k">shaping</span><span>From step ${c.step}${c.author ? ` (${esc(c.author)})` : ""}: ${esc(what)}.</span></div>`;
    }
    h += `<div class="dj-note">Takes effect when step ${n} starts dispatching.</div>`;
    $("#dj-queued").innerHTML = h;
  }

  /* ------------------------------------------------------------ step detail */
  function selectStep(n, open) {
    dj.sel = n;
    renderStep();
    if (open) { const det = $("#dj-d-step"); det.open = true; det.scrollIntoView({ behavior: "smooth", block: "start" }); }
    if (dj.open.has("history")) renderHistory();
  }
  function renderStep() {
    const d = dj.data, st = step(dj.sel);
    if (!st) return;
    $("#dj-t-step").textContent = `Step ${st.step}${st.state === "dispatching" ? " (now)" : st.state === "queued" ? " (queued)" : ""}`;
    $("#dj-s2-step").textContent = st.plan ? `Plan #${st.plan.id} by ${st.plan.author || "?"}` : st.deleted.length ? `No plan: #${st.deleted.join(", #")} was deleted` : "No plan: ran on the config";
    let h = "";
    if (st.step !== d.dispatching) h += `<div><a href="#" class="dj-back muted">← step ${d.dispatching}</a></div>`;
    if (st.plan) h += `<div class="dj-kv"><span>Plan <b>#${st.plan.id}</b> by <b>${esc(st.plan.author || "?")}</b>${st.plan.time ? `, posted at ${hhmm(st.plan.time)} UTC` : ""}</span></div>${st.plan.note ? `<div class="dj-note-full">${esc(st.plan.note)}</div>` : ""}`;
    if (st.state !== "queued") {
      const via = st.via || {};
      const parts = [via.tasks ? `${via.tasks} from the Task list` : "", via.weights ? `${via.weights} by plan weights` : "", via.sampler ? `${via.sampler} by the sampler` : ""].filter(Boolean).join(", ");
      h += `<div class="dj-kv"><span>Dispatched <b>${sum(Object.values(st.dispatched))}</b>${parts ? ` (${parts})` : ""}</span><span>finalized <b>${st.finalized}</b>, <b>${st.mixed}</b> with mixed rewards</span>${st.stuck_penalized != null ? `<span>stuck penalty cut <b>${st.stuck_penalized}</b> rollouts, unfinished score set <b>${st.unfinished_scored}</b></span>` : ""}</div>`;
    }
    h += `<div class="dj-kv"><span>Stuck-call penalty: <b>${stuckText(st.shaping && st.shaping.stuck)}</b></span><span>Unfinished rollouts: <b>${unfinishedText(st.shaping && st.shaping.unfinished)}</b></span></div>`;
    const total = sum(Object.values(st.dispatched));
    const named = (st.plan && st.plan.plan && st.plan.plan.sources) || {};
    const fmax = Math.max(0.3, ...d.sources.map((s) => Math.max(s.share, st.shares[s.name] || 0, total ? (st.dispatched[s.name] || 0) / total : 0))) * 1.04;
    const sc = (v) => (v / fmax) * 100;
    let t = `<div class="dj-scroll"><table class="dj-table" style="min-width:720px"><thead><tr><th>source</th><th class="r">config</th><th class="r">plan</th><th class="r">× config</th><th class="r">dispatched</th><th>▮ plan  │ config  ○ dispatched</th></tr></thead><tbody>`;
    d.sources.forEach((s, i) => {
      const p = st.shares[s.name] ?? s.share, disp = total ? (st.dispatched[s.name] || 0) / total : null, isNamed = s.name in named;
      t += `<tr><td>${sw(i)} ${esc(s.name)}</td><td class="r">${pct(s.share)}</td><td class="r ${isNamed ? "hi" : ""}">${st.plan ? pct(p) : "—"}</td><td class="r ${isNamed ? "hi" : "muted"}">${st.plan ? `×${(p / s.share).toFixed(2)}` : ""}</td>
        <td class="r">${total ? `${st.dispatched[s.name] || 0} · ${pct(disp, 0)}` : "—"}</td>
        <td><div class="mbar"><div class="tr"></div><div class="pl" style="width:${sc(p)}%;background:${colorOf(i)}"></div><div class="cf" style="left:${sc(s.share)}%"></div>${disp != null ? `<div class="dp" style="left:${sc(disp)}%"></div>` : ""}</div></td></tr>`;
    });
    h += t + "</tbody></table></div>";
    const plan = st.plan && st.plan.plan;
    if (plan && (plan.weights || plan.tasks)) {
      const w = Object.entries(plan.weights || {}).map(([k, v]) => `${esc(k)} ×${v}`).join(", ");
      h += `<div class="dj-pairs">${w ? `<span class="k">weights</span><span class="v">${w}</span>` : ""}${plan.tasks ? `<span class="k">Task list</span><span class="v">${plan.tasks.map((x) => `${esc(x.env)} · ${esc(x.key)}`).join("<br>")}</span>` : ""}</div>`;
    }
    $("#dj-step").innerHTML = h;
    const back = $("#dj-step .dj-back");
    if (back) back.addEventListener("click", (e) => { e.preventDefault(); selectStep(dj.data.dispatching); });
  }

  /* ------------------------------------------------------------ sources */
  function renderSources() {
    const d = dj.data;
    const row = (name, s, share) => `<tr><td>${name}</td><td class="r">${share}</td><td class="r">${s.groups}</td><td class="r hi">${pct(s.signal, 0)}</td><td class="r">${pct(s.pass1, 0)}</td><td class="r">${s.turns == null ? "—" : s.turns.toFixed(1)}</td><td class="r">${s.tokens == null ? "—" : Math.round(s.tokens).toLocaleString("en-US")}</td><td class="r">${pct(s.unfinished, 0)}</td><td class="r hi">${s.signal_per_mtok == null ? "—" : s.signal_per_mtok.toFixed(1)}</td></tr>`;
    let h = `<div class="dj-scroll"><table class="dj-table" style="min-width:960px"><thead><tr><th>source</th><th class="r">share now</th><th class="r">groups</th><th class="r">with signal</th><th class="r">train pass@1</th><th class="r">turns per rollout</th><th class="r">sampled tokens per rollout</th><th class="r">unfinished rate</th><th class="r">signal groups per million tokens</th></tr></thead><tbody>`;
    d.sources.forEach((s, i) => (h += row(`${sw(i)} ${esc(s.name)}`, s, pct(s.share_now))));
    h += row("all sources", { ...d.all, tokens: null }, "100%");
    $("#dj-sources").innerHTML = `${h}</tbody></table></div><div class="dj-note">${d.window ? `Steps ${d.window[0]}–${d.window[1]}. ` : ""}A group is one Task's rollouts. Signal: groups whose rollouts' rewards differ. Sampled tokens are the tokens the policy generated.</div>`;
  }

  /* ------------------------------------------------------------ history */
  function renderHistory() {
    const d = dj.data, host = $("#dj-history");
    if (!host.querySelector("#dj-hist-chart")) {
      host.innerHTML = `<div><div class="t-label" style="margin-bottom:6px">Run · ${d.max_steps || "?"} steps</div><div class="dj-strip-run" id="dj-run-strip"></div>
        <div class="dj-legend" style="margin-top:10px"><span><span class="dj-sw" style="background:var(--accent)"></span>plan</span><span><span class="dj-sw" style="background:var(--grey-3)"></span>no plan</span><span><span class="dj-sw" style="border:1px dashed var(--accent);background:none"></span>queued</span><span><span class="dj-sw" style="width:5px;height:5px;border-radius:50%;background:var(--grey-5)"></span>eval</span></div></div>
        <div style="display:flex;gap:10px;align-items:center;flex-wrap:wrap"><span class="t-label">Shares by step</span><div class="seg" id="dj-seg-hist"><button data-h="planned">planned</button><button data-h="dispatched">dispatched</button></div></div>
        <div id="dj-hist-chart"></div><div class="dj-legend" id="dj-hist-legend"></div>`;
      host.querySelectorAll("#dj-seg-hist button").forEach((b) => b.addEventListener("click", () => { dj.histMode = b.dataset.h; renderHistory(); }));
    }
    host.querySelectorAll("#dj-seg-hist button").forEach((b) => b.classList.toggle("active", b.dataset.h === dj.histMode));
    const evaluated = evalSteps(), interval = d.eval.interval;
    let cells = "";
    const total = d.max_steps || d.steps.length;
    for (let s = 1; s <= total; s++) {
      const st = step(s);
      const isEval = evaluated.has(s) || (interval && s % interval === 0);
      cells += `<i class="${st ? (st.state === "queued" ? "pend" : st.plan ? "plan" : "none") : ""} ${s === d.dispatching ? "now" : ""} ${isEval ? `ev ${s > d.dispatching ? "fut" : ""}` : ""}" data-s="${s}"></i>`;
    }
    $("#dj-run-strip").innerHTML = cells;
    host.querySelectorAll("#dj-run-strip i").forEach((el) => { const s = +el.dataset.s; bindTip(el, () => stepTip(s)); if (step(s)) el.addEventListener("click", () => selectStep(s, true)); });

    const chart = $("#dj-hist-chart"), W = Math.max(300, chart.clientWidth);
    const N = d.steps.length, L = 84, Rm = 6, T = 14, PH = 190, laneH = 12, laneGap = 5, lane0 = T + PH + 12;
    const LANES = ["plan", "penalty", "unfin. score", "eval"];
    const laneY = (k) => lane0 + k * (laneH + laneGap), AXY = laneY(LANES.length) + 6, H = AXY + 6;
    const bw = (W - L - Rm) / Math.max(N, 1), X = (s) => L + (s - 1) * bw, Y = (v) => T + PH * (1 - v);
    let g = "";
    for (const v of [0, 0.25, 0.5, 0.75, 1]) g += `<line class="${v ? "gl" : "bl"}" x1="${L}" x2="${W - Rm}" y1="${Y(v)}" y2="${Y(v)}"/><text class="ax" x="${L - 6}" y="${Y(v) + 3}" text-anchor="end">${v * 100}%</text>`;
    for (const st of d.steps) {
      const tot = sum(Object.values(st.dispatched));
      const shares = d.sources.map((s) => (dj.histMode === "dispatched" && tot ? (st.dispatched[s.name] || 0) / tot : st.shares[s.name] ?? s.share));
      let acc = 0;
      const x0 = X(st.step) + 1, w = Math.max(1, bw - 2), op = st.state === "queued" ? 0.45 : 1;
      shares.forEach((v, i) => { if (v <= 0) return; g += `<rect x="${x0.toFixed(1)}" y="${Y(acc + v).toFixed(1)}" width="${w.toFixed(1)}" height="${Math.max(0.6, v * PH - 1).toFixed(1)}" fill="${colorOf(i)}" opacity="${op}"/>`; acc += v; });
      if (st.state === "queued") g += `<rect x="${x0}" y="${Y(1)}" width="${w}" height="${PH}" fill="none" stroke="var(--accent)" stroke-dasharray="3 2"/>`;
      if (st.plan) g += `<rect x="${x0}" y="${laneY(0)}" width="${w}" height="${laneH}" fill="${st.state === "queued" ? "none" : "var(--accent)"}" ${st.state === "queued" ? 'stroke="var(--accent)" stroke-dasharray="2 2"' : ""}/>`;
      else g += `<rect x="${x0}" y="${laneY(0)}" width="${w}" height="${laneH}" fill="var(--grey-3)"/>`;
      if (st.deleted.length) g += `<rect x="${x0}" y="${laneY(0) - 3}" width="${w}" height="2" fill="var(--negative)"/>`;
      if (bw >= 13 && st.plan && st.plan.author) g += `<text class="lt" x="${x0 + w / 2}" y="${laneY(0) + laneH - 3}" text-anchor="middle" ${st.state === "queued" ? 'style="fill:var(--accent)"' : ""}>${esc(st.plan.author[0])}</text>`;
      const stuckOn = st.shaping && st.shaping.stuck && st.shaping.stuck.per_call > 0, unfOn = st.shaping && st.shaping.unfinished && st.shaping.unfinished.reward != null;
      [stuckOn, unfOn].forEach((on, k) => {
        const y1 = laneY(1 + k);
        g += `<line x1="${X(st.step)}" x2="${X(st.step) + bw}" y1="${y1 + laneH / 2}" y2="${y1 + laneH / 2}" stroke="var(--hairline)"/>`;
        if (on) g += `<rect x="${X(st.step)}" y="${y1 + 3}" width="${bw}" height="${laneH - 6}" fill="var(--grey-5)" opacity="${op}"/>`;
      });
      if (evaluated.has(st.step)) g += `<circle cx="${X(st.step) + bw / 2}" cy="${laneY(3) + laneH / 2}" r="4" fill="var(--grey-5)" stroke="var(--grey-2)" stroke-width="2"/>`;
    }
    LANES.forEach((name, k) => (g += `<text class="ax" x="${L - 6}" y="${laneY(k) + laneH - 2}" text-anchor="end">${name}</text>`));
    if (d.dispatching) g += `<path d="M${X(d.dispatching) + bw / 2 - 4},${T - 9} l4,5 l4,-5 z" fill="var(--accent)"/>`;
    if (dj.sel) g += `<rect x="${X(dj.sel) + 0.5}" y="${Y(1) - 2}" width="${Math.max(1, bw - 1)}" height="${PH + 4}" fill="none" stroke="var(--grey-6)"/>`;
    const tick = N > 60 ? 20 : N > 20 ? 5 : 1;
    for (let s = 1; s <= N; s++) if (s === 1 || s % tick === 0) g += `<text class="ax" x="${X(s) + bw / 2}" y="${AXY}" text-anchor="middle">${s}</text>`;
    g += `<rect id="dj-hist-hover" x="0" y="${T}" width="${bw}" height="${AXY - T - 8}" fill="rgba(255,255,255,0.06)" visibility="hidden" pointer-events="none"/>`;
    g += `<rect id="dj-hist-hit" x="${L}" y="0" width="${W - L - Rm}" height="${H}" fill="transparent" style="cursor:pointer"/>`;
    chart.innerHTML = `<svg class="dj-svg" width="${W}" height="${H}" viewBox="0 0 ${W} ${H}" role="img" aria-label="Share of groups by source per step">${g}</svg>`;
    const hit = $("#dj-hist-hit"), hov = $("#dj-hist-hover");
    const stepAt = (e) => Math.min(N, Math.max(1, Math.floor((e.clientX - chart.querySelector("svg").getBoundingClientRect().left - L) / bw) + 1));
    hit.addEventListener("mousemove", (e) => { const s = stepAt(e); hov.setAttribute("x", X(s)); hov.setAttribute("visibility", "visible"); showTip(stepTip(s), e); });
    hit.addEventListener("mouseleave", () => { hov.setAttribute("visibility", "hidden"); hideTip(); });
    hit.addEventListener("click", (e) => selectStep(stepAt(e), true));
    $("#dj-hist-legend").innerHTML = d.sources.map((s, i) => `<span>${sw(i)}${esc(s.name)}</span>`).join("");
  }

  /* ------------------------------------------------------------ log */
  function renderLog() {
    const d = dj.data, items = [];
    d.plan_events.forEach((e) => items.push({ t: e.time || "", kind: e.action === "delete" ? "del" : "plan", e }));
    d.shaping.changes.forEach((e) => items.push({ t: e.time || "", kind: "shape", e }));
    let start = null;
    for (const st of [...d.steps, { step: Infinity, plan: {} }]) {
      const none = st.step <= d.dispatching && !st.plan;
      if (none && start == null) start = st.step;
      if (!none && start != null) { items.push({ t: "", kind: "none", a: start, b: st.step - 1 }); start = null; }
    }
    items.sort((x, y) => (y.kind === "none" ? y.b : y.e.step) - (x.kind === "none" ? x.b : x.e.step) || (y.t > x.t ? 1 : -1));
    let h = `<table class="dj-table" style="min-width:960px"><thead><tr><th class="r">#</th><th>posted (UTC)</th><th class="r">step</th><th>event</th><th>by</th><th>note</th><th>plan</th></tr></thead><tbody>`;
    for (const it of items) {
      if (it.kind === "none") { h += `<tr class="dim clk" data-s="${it.a}"><td class="r">—</td><td></td><td class="r">${it.a === it.b ? it.a : `${it.a}–${it.b}`}</td><td>no plan</td><td>—</td><td>Ran on the config.</td><td>—</td></tr>`; continue; }
      const e = it.e;
      if (it.kind === "shape") {
        const what = [e.change.stuck ? `stuck-call penalty: ${stuckText({ ...d.shaping.default.stuck, ...e.change.stuck })}` : "", e.change.unfinished ? `unfinished: ${unfinishedText(e.change.unfinished)}` : ""].filter(Boolean).join("; ");
        h += `<tr class="clk" data-s="${e.step}"><td class="r">s${e.id}</td><td>${hhmm(e.time)}</td><td class="r">${e.step}+</td><td class="hi">shaping${e.step > d.dispatching ? " · queued" : ""}</td><td>${esc(e.author || "—")}</td><td class="dj-note-full">${esc(e.note || "")}</td><td>${esc(what)}</td></tr>`;
        continue;
      }
      if (it.kind === "del") { h += `<tr class="dim clk" data-s="${e.step}"><td class="r">${e.id}</td><td>${hhmm(e.time)}</td><td class="r">${e.step}</td><td class="st-alarm">delete #${e.deletes}</td><td>${esc(e.author || "—")}</td><td class="dj-note-full">${esc(e.note || "")}</td><td>—</td></tr>`; continue; }
      const deleted = d.deleted.includes(e.id), state = e.step === "default" ? "default" : e.step > d.dispatching ? "queued" : deleted ? "deleted" : e.step === d.dispatching ? "now" : "used";
      const plan = e.plan || {}, parts = [];
      for (const [k, v] of Object.entries(plan.sources || {})) parts.push(`${esc(k)} ${pct(v)}`);
      for (const [k, v] of Object.entries(plan.weights || {})) parts.push(`${esc(k)} ×${v}`);
      if (plan.tasks) parts.push(`${plan.tasks.length} Tasks cued`);
      h += `<tr class="clk ${deleted ? "dim" : ""}" data-s="${e.step}"><td class="r">${e.id}</td><td>${hhmm(e.time)}</td><td class="r">${e.step}</td><td style="color:var(--accent)">plan · ${state}</td><td>${esc(e.author || "—")}</td><td class="dj-note-full">${esc(e.note || "")}</td><td style="min-width:30ch">${parts.join("<br>")}</td></tr>`;
    }
    $("#dj-log").innerHTML = `<div class="dj-scroll" style="max-height:460px;overflow:auto">${h}</tbody></table></div>`;
    document.querySelectorAll("#dj-log tr.clk").forEach((tr) => tr.addEventListener("click", () => { const s = +tr.dataset.s; if (step(s)) selectStep(s, true); }));
  }

  /* ------------------------------------------------------------ labels, modes, eval */
  function renderLabels() {
    const d = dj.data, lw = d.labels, weighted = new Set(dj.data.weights.map((w) => w.label));
    if (!lw.rows.length) { $("#dj-labels").innerHTML = `<div class="dj-note">No labelled training rollouts read yet.</div>`; return; }
    const fam = (prefix, limit) => {
      const rs = lw.rows.filter((r) => r.label.startsWith(prefix)).sort((a, b) => b.rollouts - a.rollouts).slice(0, limit);
      if (!rs.length) return "";
      let h = `<div class="dj-lbl"><span class="h">${prefix.slice(0, -1)}</span><span class="h">pass@1 · │ all ${pct(lw.pass_now, 0)}</span><span class="h r">${lw.versions_before.length ? `v${lw.versions_before[0]}–${lw.versions_before.slice(-1)[0]} → v${lw.versions_now[0]}–${lw.versions_now.slice(-1)[0]}` : "now"}</span><span class="h r">rollouts</span>`;
      for (const r of rs) {
        const name = r.label.slice(prefix.length), w = weighted.has(r.label), thin = r.rollouts < 24;
        h += `<span class="n">${esc(name)}${w ? `<span class="wt">weighted</span>` : ""}${r.lagging ? ` <span class="flag">⚑</span>` : ""}</span>
          <span class="bar"><i class="${w ? "w" : ""}" style="width:${(r.pass_now || 0) * 100}%;${thin ? "opacity:.4" : ""}"></i>${lw.pass_now != null ? `<span class="ref" style="left:${lw.pass_now * 100}%"></span>` : ""}</span>
          <span class="v ${r.lagging ? "flag" : ""}">${r.pass_before != null ? `<span class="muted">${pct(r.pass_before, 0)} →</span> ` : ""}${pct(r.pass_now, 0)}</span>
          <span class="g">${r.rollouts.toLocaleString("en-US")}</span>`;
      }
      return `${h}</div>`;
    };
    $("#dj-labels").innerHTML = `<div class="dj-lbl-grid"><div>${fam("interface:", 8)}</div><div>${fam("domain:", 20)}</div><div>${fam("provider:", 8)}</div></div>
      <div class="dj-note">⚑ Lagging: at least 5 points below all rollouts and no better than the versions before (24 rollouts in each at least). Lime bars are weighted in step ${d.dispatching}'s plan. Faded bars have fewer than 24 rollouts.</div>`;
  }
  function heat(v, max) { const f = Math.min(1, Math.max(0, v / max)), a = [0x26, 0x26, 0x26], b = [0xfc, 0xda, 0xa4]; return `rgb(${a.map((x, k) => Math.round(x + (b[k] - x) * f)).join(",")})`; }
  function renderModes() {
    const m = dj.data.modes;
    if (!m.versions.length) { $("#dj-modes").innerHTML = `<div class="dj-note">No failed training rollouts read yet.</div>`; return; }
    const max = Math.max(0.05, ...m.rows.flatMap((r) => r.values.filter((v) => v != null)));
    const half = Math.floor(m.versions.length / 2);
    let h = `<table class="dj-heat"><thead><tr><th class="l">failure mode</th>${m.versions.map((v) => `<th>v${v}</th>`).join("")}<th class="l">trend</th><th class="l">labels that target it</th></tr></thead><tbody>`;
    for (const r of m.rows) {
      const vals = r.values.map((v) => v ?? 0), before = sum(vals.slice(0, half)) / Math.max(1, half), now = sum(vals.slice(half)) / Math.max(1, vals.length - half);
      h += `<tr><td class="l">${MODE_NAMES[r.mode] || r.mode}</td>${r.values.map((v) => `<td style="background:${heat(v || 0, max)};color:${(v || 0) / max > 0.55 ? "var(--grey-1)" : "var(--grey-5)"}">${v == null ? "—" : (v * 100).toFixed(1)}</td>`).join("")}<td class="d">${half ? `${(before * 100).toFixed(1)} → ${(now * 100).toFixed(1)} ${now > before ? "▲" : "▼"}` : ""}</td><td class="f">${esc(MODE_LABELS[r.mode] || "")}</td></tr>`;
    }
    h += `<tr><td class="l muted">passed</td>${m.passed.map((v) => `<td class="muted">${v == null ? "—" : (v * 100).toFixed(1)}</td>`).join("")}<td></td><td></td></tr>`;
    $("#dj-modes").innerHTML = `<div class="dj-scroll">${h}</tbody></table></div><div class="dj-note">Percent of training rollouts each policy version sampled. First-pass modes from Ocean's verdict, the final answer and the stop condition, as gather.py reads them.</div>`;
  }
  function renderEval() {
    const d = dj.data;
    if (!d.eval.sources.length) { $("#dj-eval").innerHTML = `<div class="dj-note">No eval yet.</div>`; return; }
    const table = (source) => `<div class="dj-eval-src"><div class="t-label">${esc(source.name)}${source.name === d.eval.primary ? " · the watchlist's" : ""}</div>
      <div class="dj-scroll"><table class="dj-table"><thead><tr><th class="r">step</th><th class="r">pass@1</th><th class="r">unfinished rate</th><th class="r">repetition rate</th><th class="r">rollouts</th></tr></thead><tbody>
      ${source.rows.map((e) => `<tr><td class="r">${e.step}</td><td class="r hi">${pct(e.pass1)}</td><td class="r">${pct(e.unfinished)}</td><td class="r">${pct(e.repetition)}</td><td class="r">${e.rollouts ?? "—"}</td></tr>`).join("")}</tbody></table></div></div>`;
    $("#dj-eval").innerHTML = `<div class="dj-eval-grid">${d.eval.sources.map(table).join("")}</div>
      <div class="dj-note">${d.eval.interval ? `Every ${d.eval.interval} steps. ` : ""}The watchlist judges ${esc(d.eval.primary || "the first source")} only. Confirmation only: plans steer from training rollouts, never from eval Tasks.</div>`;
  }

  /* ------------------------------------------------------------ request builder */
  const B = { step: "next", author: "", note: "", shares: "", weights: "", tasks: "", stuck: "", unfinished: "" };
  function renderBuilder() {
    const base = (djPrefs.control && djPrefs.control[state.run]) || "http://localhost:8041";
    $("#dj-builder").innerHTML = `<div class="dj-cb">
      <div class="dj-fields">
        <label>step <input type="text" id="dj-b-step" value="${esc(B.step)}"></label>
        <label>author <input type="text" id="dj-b-author" value="${esc(B.author)}" placeholder="your name"></label>
        <label class="wide">note <textarea id="dj-b-note" rows="2" placeholder="why, with the numbers">${esc(B.note)}</textarea></label>
        <label class="wide">shares · one "source share" per line (0–1) <textarea id="dj-b-shares" rows="3" placeholder="${esc(dj.data.sources[0].name)} 0.2">${esc(B.shares)}</textarea></label>
        <label class="wide">weights · one "selector multiplier" per line <textarea id="dj-b-weights" rows="3" placeholder="label:interface:rpc 1.5">${esc(B.weights)}</textarea></label>
        <label class="wide">Task list · one "source key" per line <textarea id="dj-b-tasks" rows="2">${esc(B.tasks)}</textarea></label>
        <label>stuck penalty per call (0–1, blank: leave) <input type="text" id="dj-b-stuck" value="${esc(B.stuck)}"></label>
        <label>unfinished score (blank: leave, "default": the env's) <input type="text" id="dj-b-unf" value="${esc(B.unfinished)}"></label>
      </div>
      <div id="dj-b-out"></div></div>`;
    const bind = (id, key) => $(id).addEventListener("input", (e) => { B[key] = e.target.value; renderBuilderOut(base); });
    [["#dj-b-step", "step"], ["#dj-b-author", "author"], ["#dj-b-note", "note"], ["#dj-b-shares", "shares"], ["#dj-b-weights", "weights"], ["#dj-b-tasks", "tasks"], ["#dj-b-stuck", "stuck"], ["#dj-b-unf", "unfinished"]].forEach(([id, key]) => bind(id, key));
    renderBuilderOut(base);
  }
  function renderBuilderOut(base) {
    const lines = (s) => s.split("\n").map((l) => l.trim()).filter(Boolean).map((l) => l.split(/\s+/));
    const plan = {};
    const shares = lines(B.shares), weights = lines(B.weights), tasks = lines(B.tasks);
    if (shares.length) plan.sources = Object.fromEntries(shares.map(([k, v]) => [k, +v]));
    if (weights.length) plan.weights = Object.fromEntries(weights.map(([k, v]) => [k, +v]));
    if (tasks.length) plan.tasks = tasks.map(([env, ...key]) => ({ env, key: key.join(" ") }));
    const stepVal = B.step === "next" || B.step === "default" ? B.step : +B.step;
    const q = (o) => JSON.stringify(o).replace(/'/g, "'\\''");
    const out = [];
    if (Object.keys(plan).length) out.push(`curl -s -X POST ${base}/v1/plans -H 'content-type: application/json' -d '${q({ step: stepVal, author: B.author || undefined, note: B.note || undefined, plan })}'`);
    const shape = {};
    if (B.stuck.trim()) shape.stuck = { per_call: +B.stuck };
    if (B.unfinished.trim()) shape.unfinished = { reward: B.unfinished.trim() === "default" ? null : +B.unfinished };
    if (Object.keys(shape).length) out.push(`curl -s -X POST ${base}/v1/shaping -H 'content-type: application/json' -d '${q({ step: stepVal, author: B.author || undefined, note: B.note || undefined, ...shape })}'`);
    out.push(`curl -s -X DELETE '${base}/v1/plans/${dj.data.dispatching + 1}${B.author ? `?author=${enc(B.author)}` : ""}'`);
    $("#dj-b-out").innerHTML = `<div class="t-label">requests</div>${out.map((c) => `<pre class="dj-code">${esc(c)}</pre>`).join("")}<div class="dj-note">The control API refuses a step whose dispatch has started (409) and a plan it can't apply (422). The last line deletes the plan queued for step ${dj.data.dispatching + 1}.</div>`;
  }

  /* ------------------------------------------------------------ data */
  async function load() {
    if (!state.run || dj.busy) return;
    dj.busy = true;
    try {
      const control = (djPrefs.control && djPrefs.control[state.run]) || "";
      const data = await api(`/api/runs/${enc(state.run)}/dj${control ? `?control=${enc(control)}` : ""}`);
      if (dj.run !== state.run) { dj.sel = null; $("#dj-body").innerHTML = ""; }
      dj.run = state.run;
      dj.data = data;
      render();
    } catch (err) {
      $("#dj-status").textContent = `could not read the DJ view: ${err.message}`;
    } finally {
      dj.busy = false;
    }
  }
  window.resetDj = () => { dj.run = null; dj.data = null; dj.sel = null; const body = $("#dj-body"); if (body) body.innerHTML = ""; };
  window.refreshDj = load;
  window.initDj = async () => {
    const input = $("#dj-control");
    input.value = (djPrefs.control && djPrefs.control[state.run]) || "";
    if (!input.dataset.bound) {
      input.dataset.bound = "1";
      input.addEventListener("change", () => {
        djPrefs.control = { ...(djPrefs.control || {}), [state.run]: input.value.trim() };
        savePrefs();
        load();
      });
    }
    await load();
  };
  let resizeTimer;
  window.addEventListener("resize", () => {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(() => { if (state.tab === "dj" && dj.data && dj.data.available) render(); }, 150);
  });
})();
