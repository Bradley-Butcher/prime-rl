"""The DJ tab's view of a training run: the Task mix playing now, what is queued next, how the mix changed, and
whether it is working.

Everything is read from the run dir, as the rest of the dashboard: the orchestrator's curriculum records
(``curriculum/plans.jsonl``, ``dispatch.jsonl``, ``groups.jsonl`` and ``shaping.jsonl``; see
``prime_rl.orchestrator.curriculum``), the trainer's and eval's rows in ``metrics.jsonl``, the resolved
orchestrator config (sources, ratios, the default shaping), and the trace stream, folded by ``rollout_stats`` into
the rl-dj skill's numbers. Given the control API's URL, the view also carries its live ``/v1/status``.
"""

from __future__ import annotations

import re
import threading
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any

import orjson

from prime_rl.dashboard.rollout_stats import MODES, merge, report, stream_fold
from prime_rl.monitors.file.traces import get_index_path, get_trace_stream
from prime_rl.utils.pathing import get_curriculum_dir, get_file_monitor_dir

WINDOW = 5
"""Finished steps the per-source numbers pool, and policy versions per label window."""
TREND = 15
"""Steps in each trend series."""
TREND_POOL = 3
"""Steps each trend point pools, so a small source's points are not single groups."""
LOW_GROUPS = 20
"""Fewer groups than this in the window is too few to decide a share on (the rl-dj skill's limit)."""
LABEL_MIN_ROLLOUTS, LABEL_LAG = 24, 0.05
"""propose.py's lagging rule: at least this many rollouts in each window, and this far below all rollouts."""
CATCH_UP_S = 1.0
"""How long a request may fold new trace-stream records before answering; a background thread reads the rest."""
TRAINER_KEYS = {"entropy/all/mean": "entropy", "mismatch_kl/all/mean": "mismatch_kl", "optim/grad_norm": "grad_norm"}
EVAL_KEY = re.compile(r"^eval/(.+)/all/([^/]+)/avg@(\d+)$")
WATCHLIST = [
    # key, name, short name, format, rule. The rules are the rl-dj skill's where it gives numbers; for repetition,
    # near-duplicate and silent calls, which it says to watch "climbing", these are the page's lines.
    ("mismatch_kl", "inference/trainer mismatch KL", "mismatch KL", "num4", {"kind": "above", "alarm": 0.01}),
    ("entropy", "policy entropy", "entropy", "num2", {"kind": "fraction", "alarm": 0.5}),
    ("grad_norm", "grad norm", "grad norm", "num2", None),
    ("repetition", "repetition rate", "repetition", "pct", {"kind": "ratio", "watch": 2.0, "alarm": 3.0}),
    ("near_dup", "near-duplicate call rate", "near-dup calls", "pct", {"kind": "ratio", "watch": 1.5, "alarm": 2.0}),
    ("silent", "silent-call rate", "silent calls", "pct", {"kind": "points", "watch": 0.10, "alarm": 0.15}),
    ("unfinished", "unfinished rate (cut-off + timeout)", "unfinished", "pct", {"kind": "points", "alarm": 0.15}),
    ("stuck_calls", "stuck calls per rollout", "stuck calls", "num2", None),
    ("heldout_pass1", "held-out pass@1", "held-out pass@1", "pct", {"kind": "drop", "alarm": 0.12}),
]
"""The held-out pass@1 is the primary eval source's: ``held-out`` when the run has it."""

_cache_lock = threading.Lock()
_records: dict[Path, tuple[tuple[int, float], list[dict]]] = {}
_metrics: dict[Path, dict] = {}


def records(path: Path) -> list[dict]:
    """A JSONL record file's lines, cached by size and mtime; a line a crash cut short is dropped. A resume
    rewrites these files, so the cache is keyed on both."""
    try:
        stat = path.stat()
    except OSError:
        return []
    key = (stat.st_size, stat.st_mtime)
    with _cache_lock:
        cached = _records.get(path)
        if cached and cached[0] == key:
            return cached[1]
    rows = []
    for line in path.read_bytes().splitlines():
        try:
            rows.append(orjson.loads(line))
        except orjson.JSONDecodeError:
            continue
    with _cache_lock:
        _records[path] = (key, rows)
    return rows


def since_reset(events: list[dict]) -> list[dict]:
    """The events after the last reset, which a fresh start logs (plans.jsonl, shaping.jsonl)."""
    resets = [index for index, event in enumerate(events) if event.get("action") == "reset"]
    return events[resets[-1] + 1 :] if resets else events


# ── config ─────────────────────────────────────────────────────────────


def source_name(source: dict) -> str:
    env = source.get("env") or {}
    return source.get("name") or (env.get("taskset") or {}).get("id") or env.get("id") or "?"


def run_sources(config: dict) -> list[dict]:
    """Each training source's name, config share, group size, default shaping and difficulty pools."""
    train = config.get("train") or {}
    sources = train.get("source") or []
    total = sum(source.get("ratio") or 1.0 for source in sources) or 1.0
    out = []
    for source in sources:
        sampler = (source.get("curriculum") or {}).get("sampler") or {}
        out.append(
            {
                "name": source_name(source),
                "share": (source.get("ratio") or 1.0) / total,
                "group_size": source.get("group_size") or train.get("group_size") or 1,
                "shaping": ((source.get("algo") or {}).get("shaping")),
                "pools": {name: pool.get("weight") for name, pool in (sampler.get("pools") or {}).items()}
                if sampler.get("type") == "difficulty_pool"
                else None,
            }
        )
    return out


# ── metrics.jsonl ──────────────────────────────────────────────────────


def metric_series(run_dir: Path) -> dict:
    """The trainer's entropy, mismatch KL and grad norm, and each eval env's avg@k, by step; read on from where the
    last call stopped."""
    path = get_file_monitor_dir(run_dir) / "metrics.jsonl"
    size = path.stat().st_size if path.is_file() else 0
    with _cache_lock:
        state = _metrics.get(path)
        if state is None or size < state["offset"]:
            state = _metrics[path] = {"offset": 0, "trainer": {}, "eval": {}}
        offset = state["offset"]
    if size > offset:
        with path.open("rb") as f:
            f.seek(offset)
            for line in f:
                if not line.endswith(b"\n"):
                    break
                offset += len(line)
                if b"entropy/all/mean" not in line and b"optim/grad_norm" not in line and b"/avg@" not in line:
                    continue
                try:
                    row = orjson.loads(line)
                except orjson.JSONDecodeError:
                    continue
                step = row.get("step")
                if step is None:
                    continue
                for key, name in TRAINER_KEYS.items():
                    if isinstance(row.get(key), (int, float)):
                        state["trainer"].setdefault(name, {})[step] = row[key]
                for key, value in row.items():
                    match = EVAL_KEY.match(key)
                    if match and isinstance(value, (int, float)):
                        state["eval"].setdefault(match.group(1), {})[step] = value
        with _cache_lock:
            state["offset"] = offset
    return state


# ── the curriculum records ─────────────────────────────────────────────


def plan_history(events: list[dict]) -> tuple[dict[int, dict], list[dict], set[int]]:
    """The plan in force for each step, every set and delete event since the last reset, and the deleted plans."""
    events = since_reset(events)
    by_step: dict[Any, dict] = {}
    deleted: set[int] = set()
    for event in events:
        if event.get("action") == "set":
            by_step[event["step"]] = event
        elif event.get("action") == "delete":
            by_step.pop(event["step"], None)
            if event.get("deletes") is not None:
                deleted.add(event["deletes"])
    return by_step, [event for event in events if event.get("action") in ("set", "delete")], deleted


def dispatch_steps(rows: list[dict]) -> tuple[dict[int, dict], int]:
    """Each step's dispatch: groups by source, what chose them, and its plan; and the latest step dispatching."""
    steps: dict[int, dict] = {}
    for row in rows:
        step = steps.setdefault(row["step"], {"counts": Counter(), "via": Counter(), "plan": None, "origin": None})
        if row.get("type") == "task":
            step["counts"][row["env"]] += 1
            step["via"][row.get("via") or "sampler"] += 1
            step["plan"], step["origin"] = row.get("plan"), row.get("origin")
        elif row.get("type") == "step":
            step["plan"], step["origin"], step["done"] = row.get("plan"), row.get("origin"), True
    dispatching = max((step for step, entry in steps.items() if entry["counts"]), default=0)
    return steps, dispatching


GROUP_FIELDS = ("groups", "mixed", "scored", "reward", "rollouts", "turns", "truncated", "timed_out", "errored")


def group_tallies(rows: list[dict]) -> tuple[dict[int, dict[str, dict]], dict[int, dict]]:
    """Finalized groups by step and source, and each step's shaping and how many rollouts each lever changed."""
    tallies: dict[int, dict[str, dict]] = {}
    shaping: dict[int, dict] = {}
    for row in rows:
        tally = tallies.setdefault(row["step"], {}).setdefault(row["env"], dict.fromkeys(GROUP_FIELDS, 0))
        tally["groups"] += 1
        tally["mixed"] += bool(row.get("mixed"))
        if row.get("reward") is not None:
            tally["scored"] += 1
            tally["reward"] += row["reward"]
        for name in ("rollouts", "turns", "truncated", "timed_out", "errored"):
            tally[name] += row.get(name) or 0
        if "shaping" in row:
            step = shaping.setdefault(
                row["step"], {"shaping": row["shaping"], "stuck_penalized": 0, "unfinished_scored": 0, "rollouts": 0}
            )
            step["stuck_penalized"] += row.get("stuck_penalized") or 0
            step["unfinished_scored"] += row.get("unfinished_scored") or 0
            step["rollouts"] += row.get("rollouts") or 0
    return tallies, shaping


def pooled(tallies: dict[int, dict[str, dict]], steps, envs) -> dict:
    total = dict.fromkeys(GROUP_FIELDS, 0)
    for step in steps:
        for env in envs:
            for name, value in (tallies.get(step) or {}).get(env, {}).items():
                total[name] += value
    return total


def rates(tally: dict) -> dict:
    """A pooled tally's rates: groups with signal (mixed rewards), pass@1 (mean group reward), turns and the
    unfinished share of rollouts."""
    return {
        "groups": tally["groups"],
        "signal": tally["mixed"] / tally["groups"] if tally["groups"] else None,
        "pass1": tally["reward"] / tally["scored"] if tally["scored"] else None,
        "turns": tally["turns"] / tally["rollouts"] if tally["rollouts"] else None,
        "unfinished": (tally["truncated"] + tally["timed_out"]) / tally["rollouts"] if tally["rollouts"] else None,
    }


def resolve_shaping(default: dict, changes: list[dict]) -> dict:
    """``default`` with each change's fields set in turn, as ``curriculum.shaping._resolve`` does."""
    fields = {lever: dict(values) for lever, values in default.items()}
    for change in changes:
        for lever, values in (change.get("change") or {}).items():
            fields[lever] = fields.get(lever, {}) | values
    return fields


def shaping_view(events: list[dict], default: dict | None, dispatching: int) -> dict:
    """The shaping in force at the step dispatching, the changes since the last reset, and when each lever was last
    set."""
    changes = sorted((event for event in since_reset(events) if event.get("action") == "set"), key=lambda e: e["step"])
    default = default or {
        "stuck": {"metric": "stuck_calls", "per_call": 0.0, "free": 2, "cap": 0.3},
        "unfinished": {"reward": None},
    }
    applied = [change for change in changes if change["step"] <= dispatching]
    since = {}
    for change in applied:
        for lever in change.get("change") or {}:
            since[lever] = {"step": change["step"], "author": change.get("author")}
    return {
        "default": default,
        "active": resolve_shaping(default, applied),
        "since": since,
        "changes": changes,
        "pending": [change for change in changes if change["step"] > dispatching],
    }


# ── the view ───────────────────────────────────────────────────────────


def judge(rule: dict | None, series: list[list[float]]) -> dict:
    """A watchlist metric's status against its rule: ``alarm``, ``watch`` or ``ok``, with its lines."""
    if not series:
        return {"status": "none"}
    start, now = series[0][1], series[-1][1]
    out: dict[str, Any] = {
        "status": "ok",
        "start": start,
        "start_step": series[0][0],
        "now": now,
        "now_step": series[-1][0],
    }
    if rule is None:
        return out
    kind, alarm, watch = rule["kind"], rule.get("alarm"), rule.get("watch")
    if kind == "above":
        out["alarm_line"], out["watch_line"] = alarm, watch
        out["status"] = "alarm" if now > alarm else "watch" if watch is not None and now >= watch else "ok"
    elif kind == "fraction":
        out["alarm_line"] = start * alarm
        out["status"] = "alarm" if now <= start * alarm else "ok"
    elif kind == "ratio" and start > 0:
        out["ratio"] = now / start
        out["alarm_line"], out["watch_line"] = start * alarm, start * watch
        out["status"] = "alarm" if now / start >= alarm else "watch" if now / start >= watch else "ok"
    elif kind == "points":
        out["alarm_line"] = start + alarm
        out["watch_line"] = start + watch if watch is not None else None
        out["status"] = (
            "alarm" if now - start >= alarm else "watch" if watch is not None and now - start >= watch else "ok"
        )
    elif kind == "drop":
        out["alarm_line"] = start - alarm
        out["status"] = "alarm" if start - now >= alarm else "ok"
    return out


def weight_since(plans: dict[int, dict], selector: str, step: int) -> int:
    """The first step of the unbroken run of plans up to ``step`` that carry ``selector``."""
    while step > 1 and selector in ((plans.get(step - 1) or {}).get("plan") or {}).get("weights", {}):
        step -= 1
    return step


def label_of(selector: str) -> str | None:
    """What a ``label:`` weight matches among the labels the fold counts: a label, or ``kind=<v>``, the one Task
    field the fold reads."""
    value = selector.removeprefix("label:")
    if "=" in value and not value.startswith("kind="):
        return None
    return value


def control_status(url: str) -> dict:
    """The control API's ``/v1/status``, or what stopped the read."""
    if not re.match(r"^https?://", url):
        return {"error": "the control URL must start with http:// or https://"}
    try:
        with urllib.request.urlopen(url.rstrip("/") + "/v1/status", timeout=2) as response:
            return orjson.loads(response.read())
    except (urllib.error.URLError, TimeoutError, OSError, orjson.JSONDecodeError) as error:
        return {"error": str(error)}


def view(run_dir: Path, config: dict, control_url: str | None = None) -> dict:
    curriculum = get_curriculum_dir(run_dir)
    sources = run_sources(config)
    if not (curriculum / "dispatch.jsonl").is_file() or not sources:
        return {
            "available": False,
            "reason": "This run has no curriculum records. They are written by an RL run with [orchestrator.curriculum_plans].",
        }
    names = [source["name"] for source in sources]
    plans, plan_events, deleted = plan_history(records(curriculum / "plans.jsonl"))
    steps, dispatching = dispatch_steps(records(curriculum / "dispatch.jsonl"))
    tallies, shaped = group_tallies(records(curriculum / "groups.jsonl"))
    shaping = shaping_view(records(curriculum / "shaping.jsonl"), sources[0]["shaping"], dispatching)
    metrics = metric_series(run_dir)
    stream = get_trace_stream(run_dir)
    fold = stream_fold(stream) if get_index_path(stream).is_file() else None
    if fold is not None:
        fold.catch_up(CATCH_UP_S)
    batch_groups = (config.get("batch_size") or 0) // (sources[0]["group_size"] or 1) or None
    config_shares = {source["name"]: source["share"] for source in sources}

    by_id = {event["id"]: event for event in plan_events if event.get("action") == "set"}

    def planned(step: int) -> tuple[dict | None, dict[str, float]]:
        """A step's plan (the one its dispatch records, or the one queued for it) and its shares."""
        if step <= dispatching:
            event = by_id.get((steps.get(step) or {}).get("plan"))
        else:
            event = plans.get(step) or plans.get("default")
        return event, (event or {}).get("shares") or config_shares

    last = max([dispatching, *(step for step in plans if isinstance(step, int))], default=0)
    step_rows = []
    for step in range(1, last + 1):
        event, shares = planned(step)
        used = steps.get(step) or {}
        groups = tallies.get(step) or {}
        step_rows.append(
            {
                "step": step,
                "state": "dispatching" if step == dispatching else "queued" if step > dispatching else "done",
                "origin": used.get("origin") if step <= dispatching else ("step" if event else None),
                "plan": event
                and {key: event.get(key) for key in ("id", "author", "note", "time", "requested", "plan")},
                "shares": shares,
                "dispatched": dict(used.get("counts") or {}),
                "via": dict(used.get("via") or {}),
                "finalized": sum(tally["groups"] for tally in groups.values()),
                "mixed": sum(tally["mixed"] for tally in groups.values()),
                "shaping": (shaped.get(step) or {}).get("shaping")
                or resolve_shaping(shaping["default"], [c for c in shaping["changes"] if c["step"] <= step]),
                "stuck_penalized": (shaped.get(step) or {}).get("stuck_penalized"),
                "unfinished_scored": (shaped.get(step) or {}).get("unfinished_scored"),
                "deleted": [
                    event["deletes"]
                    for event in plan_events
                    if event.get("action") == "delete" and event["step"] == step
                ],
            }
        )

    finished = [step for step in range(1, dispatching) if step in tallies]
    window = finished[-WINDOW:]
    first = finished[:WINDOW]

    def trend(envs, key):
        points = []
        for step in finished[-TREND:]:
            pool = rates(pooled(tallies, range(step - TREND_POOL + 1, step + 1), envs))
            if pool[key] is not None:
                points.append([step, pool[key]])
        return points

    token_totals: dict[str, list[int]] = {}
    carried: dict[tuple[int, str], Counter] = {}
    if fold is not None:
        with fold.lock:
            for (step, source), (tokens, count) in fold.tokens.items():
                if step in window:
                    entry = token_totals.setdefault(source, [0, 0])
                    entry[0] += tokens
                    entry[1] += count
            carried = {key: Counter(counter) for key, counter in fold.carried.items()}

    def per_mtok(rate: dict, envs) -> float | None:
        tokens = [token_totals.get(env) for env in envs]
        sampled, count = sum(t[0] for t in tokens if t), sum(t[1] for t in tokens if t)
        if not count or rate["signal"] is None:
            return None
        group_tokens = sampled / count * sources[0]["group_size"]
        return rate["signal"] / (group_tokens / 1e6) if group_tokens else None

    now_event, now_shares = planned(dispatching)
    next_event, next_shares = planned(dispatching + 1)
    now_plan = (now_event or {}).get("plan") or {}
    cued = Counter(task["env"] for task in now_plan.get("tasks") or [])
    source_rows = []
    for source in sources:
        rate = rates(pooled(tallies, window, [source["name"]]))
        source_rows.append(
            source
            | rate
            | {
                "share_now": now_shares.get(source["name"], source["share"]),
                "share_next": next_shares.get(source["name"], source["share"]),
                "named_now": source["name"] in (now_plan.get("sources") or {}),
                "cued": cued.get(source["name"], 0),
                "tokens": token_totals[source["name"]][0] / token_totals[source["name"]][1]
                if token_totals.get(source["name"], [0, 0])[1]
                else None,
                "signal_per_mtok": per_mtok(rate, [source["name"]]),
                "trend_pass1": trend([source["name"]], "pass1"),
                "trend_signal": trend([source["name"]], "signal"),
            }
        )
    all_rate = rates(pooled(tallies, window, names))

    # Label weights: what share of training rollouts carry the label, per step and per source.
    def label_share(label: str, step_range) -> float | None:
        hits = rollouts = 0
        for (step, _source), counter in carried.items():
            if step in step_range:
                hits += counter.get(label, 0)
                rollouts += counter.get("", 0)
        return hits / rollouts if rollouts else None

    weights = []
    for selector, multiplier in (now_plan.get("weights") or {}).items():
        if not selector.startswith("label:"):
            continue
        label, since = label_of(selector), weight_since(plans, selector, dispatching)
        weighted = set(range(since, dispatching + 1))
        unweighted = {
            row["step"]
            for row in step_rows
            if row["step"] < dispatching
            and selector not in (((row["plan"] or {}).get("plan") or {}).get("weights") or {})
        }
        by_source = {}
        for name in names:
            hits = sum(c.get(label, 0) for (step, src), c in carried.items() if src == name) if label else 0
            total = sum(c.get("", 0) for (step, src), c in carried.items() if src == name)
            by_source[name] = hits / total if label and total else None
        weights.append(
            {
                "selector": selector,
                "label": label,
                "multiplier": multiplier,
                "since": since,
                "author": (plans.get(since) or {}).get("author"),
                "share_before": label_share(label, unweighted) if label else None,
                "share_now": label_share(label, weighted) if label else None,
                "by_source": by_source,
            }
        )
    task_weights = {
        selector: value for selector, value in (now_plan.get("weights") or {}).items() if selector.startswith("task:")
    }
    task_env = {}
    for row in records(curriculum / "dispatch.jsonl"):
        if row.get("type") == "task" and f"task:{row['key']}" in task_weights:
            task_env[row["key"]] = row["env"]
    first_task = next(iter(task_weights), None)

    # The watchlist, by step.
    train_cells, eval_cells, policy_cells, label_cells = {}, {}, {}, {}
    stuck_metric = (shaping["active"].get("stuck") or {}).get("metric") or "stuck_calls"
    stuck_series = []
    if fold is not None:
        with fold.lock:
            for (kind, step, policy, source), cell in fold.cells.items():
                if kind == "eval":
                    eval_cells.setdefault(source, {}).setdefault(step, []).append(cell)
                    continue
                train_cells.setdefault(step, []).append(cell)
                if policy is not None:
                    policy_cells.setdefault(policy, []).append(cell)
            for (step, policy, label), cell in fold.labels.items():
                if policy is not None:
                    label_cells.setdefault(label, {}).setdefault(policy, []).append(cell)
            stuck_series = sorted(
                [step, total / count]
                for (step, name), (total, count) in fold.metrics.items()
                if name == stuck_metric and count
            )
    by_step = {step: report(merge(cells)) for step, cells in train_cells.items()}

    def stream_series(key):
        return [
            [step, by_step[step][key]]
            for step in sorted(by_step)
            if step < dispatching and by_step[step][key] is not None
        ]

    # Each eval source by step: pass@1 from the eval's own metric, the rest from its rollouts. The watchlist judges
    # the primary one, `held-out` when the run has it.
    eval_sources = []
    for name in sorted(set(metrics["eval"]) | set(eval_cells)):
        reported, cells = metrics["eval"].get(name) or {}, eval_cells.get(name) or {}
        eval_rows = []
        for step in sorted(set(reported) | set(cells)):
            cell = report(merge(cells[step])) if step in cells else None
            eval_rows.append(
                {
                    "step": step,
                    "pass1": reported.get(step, cell and cell["pass_rate"]),
                    "unfinished": cell and cell["unfinished"],
                    "repetition": cell and cell["repetitive"],
                    "rollouts": cell and cell["rollouts"],
                }
            )
        eval_sources.append({"name": name, "rows": eval_rows})
    names_eval = [source["name"] for source in eval_sources]
    primary = "held-out" if "held-out" in names_eval else (names_eval[0] if names_eval else None)
    heldout = [
        [row["step"], row["pass1"]]
        for source in eval_sources
        if source["name"] == primary
        for row in source["rows"]
        if row["pass1"] is not None
    ]
    unfinished = [[step, rates(pooled(tallies, [step], names))["unfinished"]] for step in finished]
    series = {
        "mismatch_kl": sorted([int(s), v] for s, v in metrics["trainer"].get("mismatch_kl", {}).items()),
        "entropy": sorted([int(s), v] for s, v in metrics["trainer"].get("entropy", {}).items()),
        "grad_norm": sorted([int(s), v] for s, v in metrics["trainer"].get("grad_norm", {}).items()),
        "repetition": stream_series("repetitive"),
        "near_dup": stream_series("near_dup_calls"),
        "silent": stream_series("silent_calls"),
        "unfinished": [point for point in unfinished if point[1] is not None],
        "stuck_calls": [point for point in stuck_series if point[0] < dispatching],
        "heldout_pass1": heldout,
    }
    health = [
        {"key": key, "name": name, "short": short, "format": fmt, "rule": rule, "series": series[key]}
        | judge(rule, series[key])
        for key, name, short, fmt, rule in WATCHLIST
    ]

    # Labels and failure modes by policy version: the last window of versions against the one before.
    versions = sorted(policy_cells)
    now_versions, before_versions = versions[-WINDOW:], versions[-2 * WINDOW : -WINDOW]
    pass_now = report(merge(c for v in now_versions for c in policy_cells[v]))["pass_rate"] if now_versions else None
    pass_before = (
        report(merge(c for v in before_versions for c in policy_cells[v]))["pass_rate"] if before_versions else None
    )
    label_rows = []
    for label, by_version in label_cells.items():
        now = report(merge(c for v in now_versions for c in by_version.get(v, [])))
        before = report(merge(c for v in before_versions for c in by_version.get(v, [])))
        if not now["rollouts"]:
            continue
        lagging = (
            pass_now is not None
            and now["pass_rate"] is not None
            and before["pass_rate"] is not None
            and min(now["rollouts"], before["rollouts"]) >= LABEL_MIN_ROLLOUTS
            and now["pass_rate"] <= min(pass_now - LABEL_LAG, before["pass_rate"])
        )
        label_rows.append(
            {
                "label": label,
                "rollouts": now["rollouts"],
                "pass_before": before["pass_rate"],
                "pass_now": now["pass_rate"],
                "unfinished": now["unfinished"],
                "lagging": lagging,
            }
        )
    mode_versions = versions[-2 * WINDOW :]
    mode_rows = []
    for mode in MODES:
        values = []
        for version in mode_versions:
            cell = merge(policy_cells[version])
            values.append(cell["modes"].get(mode, 0) / cell["rollouts"] if cell["rollouts"] else None)
        mode_rows.append({"mode": mode, "values": values})
    passed = [report(merge(policy_cells[v]))["pass_rate"] for v in mode_versions]

    eval_config = config.get("eval") or {}
    return {
        "available": True,
        "dispatching": dispatching,
        "max_steps": config.get("max_steps"),
        "batch_groups": batch_groups,
        "eval": {"primary": primary, "interval": eval_config.get("interval"), "sources": eval_sources},
        "window": [window[0], window[-1]] if window else None,
        "first_window": [first[0], first[-1]] if first else None,
        "sources": source_rows,
        "all": all_rate
        | {
            "first_signal": rates(pooled(tallies, first, names))["signal"],
            "signal_per_mtok": per_mtok(all_rate, names),
        },
        "train_pass1": [
            [step, rates(pooled(tallies, range(step - TREND_POOL + 1, step + 1), names))["pass1"]]
            for step in finished
            if step >= TREND_POOL
        ],
        "steps": step_rows,
        "plan_events": plan_events,
        "deleted": sorted(deleted),
        "now": {
            "step": dispatching,
            "plan": now_event and now_event.get("id"),
            "origin": (steps.get(dispatching) or {}).get("origin"),
        },
        "next": {
            "step": dispatching + 1,
            "plan": next_event
            and {key: next_event.get(key) for key in ("id", "author", "note", "time", "plan", "shares")},
        },
        "shaping": shaping,
        "weights": weights,
        "task_weights": {
            "count": len(task_weights),
            "multiplier": task_weights[first_task] if first_task else None,
            "since": weight_since(plans, first_task, dispatching) if first_task else None,
            "by_source": dict(Counter(task_env.get(selector.removeprefix("task:"), "?") for selector in task_weights)),
        },
        "pool_weights": {
            selector.removeprefix("pool:"): value
            for selector, value in (now_plan.get("weights") or {}).items()
            if selector.startswith("pool:")
        },
        "pools": next((source["pools"] for source in sources if source["pools"]), None),
        "health": health,
        "labels": {
            "versions_now": now_versions,
            "versions_before": before_versions,
            "pass_now": pass_now,
            "pass_before": pass_before,
            "rows": label_rows,
        },
        "modes": {"versions": mode_versions, "rows": mode_rows, "passed": passed},
        "progress": {"read": fold.read, "total": fold.total} if fold is not None else None,
        "live": control_status(control_url) if control_url else None,
    }
