"""Per-rollout numbers for the DJ tab, folded from a training run's trace stream.

The numbers are the rl-dj skill's ``gather.py`` (coral-training ``.claude/skills/rl-dj/scripts/gather.py``), ported
here so the page and the skill agree: training rollouts by the policy that sampled them and by dispatch step, the
held-out eval by step, and first-pass failure modes. Per rollout:

- repetitive: the last 10,000 characters of the student's own output (its messages and tool calls, tool results
  left out) compress more than 5x with zlib;
- a call is a near-duplicate when it repeats the tool of one of the rollout's previous 8 calls with arguments at
  least 0.9 similar (difflib, ``intent`` and ``task_id`` left out);
- a tool-call turn is silent when it has no text before its calls;
- unfinished: cut off by the turn cap or timed out.

An eval rollout is one whose work is eval or whose env's name starts with ``held-out``, as gather.py has it; evals are
kept by source.

The fold reads the stream through its index, one record at a time, in a background thread per stream, and keeps
only counts, so a stream of gigabytes costs memory by step, policy, source and label, not by rollout.
"""

from __future__ import annotations

import difflib
import json
import re
import threading
import time
import zlib
from collections import Counter
from dataclasses import dataclass, field
from itertools import groupby
from pathlib import Path

import orjson

from prime_rl.monitors.file.traces import get_index_path
from prime_rl.monitors.file.traces.chunks import open_chunk

TAIL, REPETITIVE = 10_000, 5.0
NEAR, NEAR_WINDOW = 0.9, 8
UNFINISHED = {"max_turns", "agent_timeout"}
STOP_OR_ASK = re.compile(r"blocked|unanswerable|vague|abstain")
COUNTS = (
    "rollouts",
    "scored",
    "passed",
    "unfinished",
    "turns",
    "repetitive",
    "calls",
    "near_dup",
    "tool_turns",
    "silent",
)
LABEL_KINDS = ("interface:", "domain:", "provider:")
MODES = ("didnt_find", "declared_success", "gave_up", "unfinished_write", "botched_write", "unfinished", "other")

# Ocean's failure texts, matched anywhere ("Turn 2: The Run never read ..." too).
NEVER_READ = re.compile(
    r"\b(never|didn't|did not) (read|found|saw|see|looked|queried|opened|searched|checked)\b|was never shown"
)
JUDGED_FALSE = re.compile(r"judge found false")
DOESNT_SAY = re.compile(r"\b(reply|answer) (doesn't|does not|didn't) (say|name|give)")
WROTE_NOTHING = re.compile(r"wrote nothing|opened no\b")
BOTCHED = re.compile(
    r"The Run (added|changed|moved|cancelled|deleted|removed|opened \d|took|asked|logged)\b|it was to|^Besides"
)
NO_ANSWER = re.compile(r"gave no answer|answered none|stopped before turn")
DECLINES = re.compile(
    r"\b(could(n't| not)|was(n't| not) able|were(n't| not) able|unable to|not able to|can(not|'t)"
    r"|did(n't| not) (find|manage))\b"
)


def text_of(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(part.get("text") or "" for part in content if isinstance(part, dict))
    return ""


def call_of(call: dict) -> tuple[str, str]:
    function = call.get("function") or call
    arguments = function.get("arguments")
    if not isinstance(arguments, str):
        arguments = json.dumps(arguments, sort_keys=True)
    return function.get("name") or "", arguments


def core(arguments: str) -> str:
    """A call's arguments without the bookkeeping (``intent``, ``task_id``) that differs between otherwise equal
    calls."""
    try:
        parsed = json.loads(arguments)
    except ValueError:
        return arguments
    if isinstance(parsed, dict):
        parsed = {key: value for key, value in parsed.items() if key not in ("intent", "task_id")}
    return json.dumps(parsed, sort_keys=True)


def near_duplicates(calls: list[tuple[str, str]]) -> int:
    count = 0
    for index, (name, arguments) in enumerate(calls):
        for before, earlier in calls[max(0, index - NEAR_WINDOW) : index]:
            if before != name:
                continue
            matcher = difflib.SequenceMatcher(None, earlier, arguments, autojunk=False)
            if earlier == arguments or (
                matcher.real_quick_ratio() >= NEAR and matcher.quick_ratio() >= NEAR and matcher.ratio() >= NEAR
            ):
                count += 1
                break
    return count


def first_pass(failure: str, answer: str, kind: str, stop: str, solvable: bool) -> str:
    """The failure mode from Ocean's text, the final answer and the stop condition. A run cut off or timed out is
    unfinished first, as Ocean's verdict then says only where it stopped."""
    declines = bool(DECLINES.search(answer[:600]))
    if stop in UNFINISHED:
        return "unfinished"
    if NEVER_READ.search(failure):
        return "didnt_find"
    if (
        answer.strip()
        and not declines
        and (JUDGED_FALSE.search(failure) or (kind != "write" and DOESNT_SAY.search(failure)))
    ):
        return "declared_success"
    if solvable and declines:
        return "gave_up"
    if WROTE_NOTHING.search(failure):
        return "unfinished_write"
    if BOTCHED.search(failure) or (kind == "write" and failure and not NO_ANSWER.search(failure)):
        return "botched_write"
    if NO_ANSWER.search(failure) or not answer.strip():
        return "unfinished"
    return "other"


@dataclass
class Rollout:
    """One episode's first trace, as the fold counts it."""

    kind: str
    step: int
    policy: int | None
    source: str
    counts: dict[str, float]
    labels: list[str]
    mode: str | None
    output_tokens: int
    metrics: dict[str, float]


def rollout(episode: dict) -> Rollout | None:
    """An episode's counts, or None for one without a trace or a step (a lost or unstamped episode)."""
    work = (episode.get("run") or {}).get("work") or {}
    if not episode.get("traces") or work.get("step") is None:
        return None
    trace = episode["traces"][0]
    data = (trace.get("task") or {}).get("data") or {}
    info = trace.get("info") or {}
    ocean = info.get("ocean") or {}
    stop = str(trace.get("stop_condition"))
    pieces, calls, tool_turns, silent, turns = [], [], 0, 0, 0
    for node in trace.get("nodes") or []:
        if not node.get("sampled"):
            continue
        turns += 1
        message = node.get("message") or {}
        content = message.get("content")
        pieces.append(content if isinstance(content, str) else json.dumps(content) if content else "")
        tool_calls = message.get("tool_calls") or []
        for call in tool_calls:
            pieces.append(json.dumps(call.get("function") or call, sort_keys=True))
            name, arguments = call_of(call)
            calls.append((name, core(arguments)))
        if tool_calls:
            tool_turns += 1
            silent += not text_of(content).strip()
    tail = "".join(pieces)[-TAIL:].encode()
    score = ((trace.get("rewards") or {}).get("passed") or {}).get("score")
    if score is None and ocean.get("passed") is not None:
        score = float(ocean["passed"])
    counts = {
        "rollouts": 1,
        "scored": score is not None,
        "passed": score or 0.0,
        "unfinished": stop in UNFINISHED,
        "turns": turns,
        "repetitive": len(tail) >= TAIL and len(tail) / len(zlib.compress(tail, 6)) > REPETITIVE,
        "calls": len(calls),
        "near_dup": near_duplicates(calls),
        "tool_turns": tool_turns,
        "silent": silent,
    }
    source = (episode.get("env") or {}).get("name") or (episode.get("env") or {}).get("id") or "?"
    kind = "eval" if work.get("type") == "eval" or source.startswith("held-out") else "train"
    labels = [str(label) for label in data.get("labels") or []]
    mode = None
    if kind == "train" and score is not None and score < 1 and stop != "task_error":
        key = (trace.get("task") or {}).get("key") or data.get("id") or ""
        solvable = not any(STOP_OR_ASK.search(label) for label in labels) and not STOP_OR_ASK.search(str(key))
        answers = info.get("turns") or []
        answer = (answers[-1].get("answer") if answers else None) or ""
        mode = first_pass(ocean.get("failure") or "", answer, data.get("kind") or "", stop, solvable)
    if data.get("kind"):
        labels.append(f"kind={data['kind']}")
    metrics = {name: value for name, value in (trace.get("metrics") or {}).items() if isinstance(value, (int, float))}
    return Rollout(
        kind=kind,
        step=int(work["step"]),
        policy=(work.get("policy") or {}).get("start"),
        source=source,
        counts=counts,
        labels=labels,
        mode=mode,
        output_tokens=int(episode.get("num_output_tokens") or 0),
        metrics=metrics,
    )


def new_cell() -> dict:
    return {name: 0 for name in COUNTS} | {"modes": Counter()}


def add(cell: dict, item: Rollout) -> None:
    for name in COUNTS:
        cell[name] += item.counts[name]
    if item.mode:
        cell["modes"][item.mode] += 1


def merge(cells) -> dict:
    total = new_cell()
    for cell in cells:
        for name in COUNTS:
            total[name] += cell[name]
        total["modes"].update(cell["modes"])
    return total


def share(part: float, whole: float) -> float | None:
    return round(part / whole, 4) if whole else None


def report(cell: dict) -> dict:
    """gather.py's report of a cell: rates over its rollouts, calls and tool turns."""
    return {
        "rollouts": cell["rollouts"],
        "pass_rate": share(cell["passed"], cell["scored"]),
        "unfinished": share(cell["unfinished"], cell["rollouts"]),
        "mean_turns": share(cell["turns"], cell["rollouts"]),
        "repetitive": share(cell["repetitive"], cell["rollouts"]),
        "near_dup_calls": share(cell["near_dup"], cell["calls"]),
        "silent_calls": share(cell["silent"], cell["tool_turns"]),
        "modes": dict(sorted(cell["modes"].items(), key=lambda item: -item[1])),
    }


@dataclass
class Fold:
    """Running counts of one trace stream, as far as its index has been read."""

    index: Path
    offset: int = 0
    """Bytes of the index already folded."""
    read: int = 0
    total: int = 0
    cells: dict[tuple[str, int, int | None, str], dict] = field(default_factory=dict)
    """(kind, step, policy, source) -> counts."""
    labels: dict[tuple[int, int | None, str], dict] = field(default_factory=dict)
    """Training rollouts by (step, policy, label) for the interface, domain and provider labels."""
    carried: dict[tuple[int, str], Counter] = field(default_factory=dict)
    """Training rollouts by (step, source): every label their Task carries, and ``""`` for all of them."""
    tokens: dict[tuple[int, str], list[int]] = field(default_factory=dict)
    """Training rollouts by (step, source): [sampled tokens, rollouts]."""
    metrics: dict[tuple[int, str], list[float]] = field(default_factory=dict)
    """Training rollouts by (step, env metric): [sum, rollouts with it]."""
    lock: threading.Lock = field(default_factory=threading.Lock)
    """Held while the counts change, so a reader sees whole records."""
    advancing: threading.Lock = field(default_factory=threading.Lock)
    """Held while records are read, so two readers never fold the same ones."""

    def fold(self, item: Rollout) -> None:
        add(self.cells.setdefault((item.kind, item.step, item.policy, item.source), new_cell()), item)
        if item.kind != "train":
            return
        for label in item.labels:
            if label.startswith(LABEL_KINDS):
                add(self.labels.setdefault((item.step, item.policy, label), new_cell()), item)
        carried = self.carried.setdefault((item.step, item.source), Counter())
        carried[""] += 1
        carried.update(set(item.labels))
        tokens = self.tokens.setdefault((item.step, item.source), [0, 0])
        tokens[0] += item.output_tokens
        tokens[1] += 1
        for name, value in item.metrics.items():
            entry = self.metrics.setdefault((item.step, name), [0.0, 0])
            entry[0] += value
            entry[1] += 1

    def advance(self, max_rows: int = 200) -> bool:
        """Fold up to ``max_rows`` more records; False once the index has nothing new."""
        with self.advancing:
            return self._advance(max_rows)

    def _advance(self, max_rows: int) -> bool:
        size = self.index.stat().st_size if self.index.is_file() else 0
        if size < self.offset:  # a new stream under the same path: start over
            self.reset()
        if size == self.offset:
            return False
        rows = []
        with self.index.open("rb") as f:
            f.seek(self.offset)
            for line in f:
                if not line.endswith(b"\n") or len(rows) >= max_rows:
                    break
                rows.append((len(line), orjson.loads(line)))
        stream = self.index.with_name(self.index.name.removesuffix(".index.jsonl"))
        folded = []
        for chunk, chunk_rows in groupby(rows, key=lambda pair: pair[1].get("chunk", 0)):
            with open_chunk(stream, chunk) as f:
                for length, row in chunk_rows:
                    f.seek(row.get("offset", 0))
                    try:
                        item = rollout(orjson.loads(f.readline()))
                    except orjson.JSONDecodeError:
                        item = None
                    folded.append((length, item))
        with self.lock:
            for length, item in folded:
                if item is not None:
                    self.fold(item)
                self.offset += length
                self.read += 1
            self.total = max(self.total, self.read)
        return bool(rows)

    def catch_up(self, budget_s: float) -> None:
        """Fold what is new for up to ``budget_s`` seconds, so a request sees a short stream whole; the background
        thread reads the rest."""
        deadline = time.monotonic() + budget_s
        while time.monotonic() < deadline and self.advance():
            pass

    def reset(self) -> None:
        with self.lock:
            self.offset = self.read = self.total = 0
            self.cells, self.labels, self.carried, self.tokens, self.metrics = {}, {}, {}, {}, {}

    def count_rows(self) -> None:
        """How many records the index lists, for the page's progress."""
        if self.index.is_file():
            with self.index.open("rb") as f:
                total = sum(1 for line in f if line.endswith(b"\n"))
            with self.lock:
                self.total = max(total, self.read)


_folds: dict[Path, Fold] = {}
_folds_lock = threading.Lock()
IDLE_POLL_S = 5.0


def stream_fold(stream: Path) -> Fold:
    """The fold of ``stream``, kept current by a background thread started on first use."""
    index = get_index_path(stream)
    key = index.resolve()
    with _folds_lock:
        fold = _folds.get(key)
        if fold is not None:
            return fold
        fold = _folds[key] = Fold(index)

    def follow() -> None:
        fold.count_rows()
        while True:
            if not fold.advance():
                time.sleep(IDLE_POLL_S)
                fold.count_rows()

    threading.Thread(target=follow, name=f"dj-fold-{stream.parent.parent.parent.name}", daemon=True).start()
    return fold
