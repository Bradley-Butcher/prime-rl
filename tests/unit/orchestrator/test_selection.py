import pytest

from prime_rl.orchestrator.selection import Candidate, CandidatePool, reward_variance


def candidate(name: str, score: float, *, version: int | None = 0, rollouts: int = 4) -> Candidate:
    return Candidate(
        group_id=name,
        env_name="env",
        task_key=name,
        step=1,
        version=version,
        score=score,
        trace_ids=[f"{name}{i}" for i in range(rollouts)],
    )


def pooled(cap: int, *candidates: Candidate) -> CandidatePool:
    pool = CandidatePool(cap)
    for c in candidates:
        assert pool.add(c) == []
    return pool


def test_variance_scores_pass_fail_groups_by_p_times_one_minus_p():
    scores = {passed: reward_variance([1.0] * passed + [0.0] * (8 - passed)) for passed in range(9)}

    assert scores[4] == pytest.approx(0.25) and scores[1] == scores[7] == pytest.approx(7 / 64)
    assert scores[0] == scores[8] == 0.0 and reward_variance([]) == 0.0
    assert max(scores, key=scores.get) == 4
    assert scores[3] == scores[5] > scores[2] == scores[6] > scores[1]


def test_ties_go_to_the_oldest_group_frozen_sources_last():
    pool = pooled(
        100,
        candidate("v3", 0.25, version=3),
        candidate("v1-first", 0.25, version=1),
        candidate("frozen", 0.25, version=None),
        candidate("v1-second", 0.25, version=1),
        candidate("best", 0.3, version=5),
    )

    assert [c.group_id for c in pool.ranked()] == ["best", "v1-first", "v1-second", "v3", "frozen"]


def test_a_batch_takes_the_top_groups_and_carries_the_rest():
    pool = pooled(100, candidate("low", 0.1), candidate("top", 0.25), candidate("mid", 0.2), candidate("min", 0.05))

    first = pool.select(6)

    assert first.trace_ids == ["top0", "top1", "top2", "top3", "mid0", "mid1"]
    assert [(c.group_id, n) for c, n in first.selected] == [("top", 4), ("mid", 2)]
    assert [c.group_id for c in first.passed] == ["low", "min"]
    assert first.metrics() == {
        "selection/pool": 4.0,
        "selection/candidates": 4.0,
        "selection/selected": 2.0,
        "selection/carried": 3.0,
        "selection/stale": 0.0,
        "selection/displaced": 0.0,
        "selection/score/selected": pytest.approx(0.225),
        "selection/score/passed": pytest.approx(0.075),
    }
    assert pool.size == 10

    pool.add(candidate("new", 0.15))
    second = pool.select(6)

    # The split group's rest comes first, then the new group outranks the ones passed over before.
    assert second.trace_ids == ["mid2", "mid3", "new0", "new1", "new2", "new3"]
    assert second.admitted == 1 and [c.group_id for c in second.passed] == ["low", "min"]
    assert [record["passed"] for record in pool.select(8).records(3)] == [2, 2]
    assert pool.size == 0 and not pool.candidates


def test_a_group_past_the_cap_displaces_the_lowest_ranked():
    pool = pooled(8, candidate("a", 0.25), candidate("b", 0.1))

    displaced = pool.add(candidate("c", 0.2))
    assert [(c.group_id, ids) for c, ids in displaced] == [("b", ["b0", "b1", "b2", "b3"])]

    # A group that ranks last displaces itself; an equal score loses to the older group.
    assert [c.group_id for c, _ in pool.add(candidate("d", 0.05))] == ["d"]
    assert [c.group_id for c, _ in pool.add(candidate("e", 0.2))] == ["e"]
    assert sorted(pool.candidates) == ["a", "c"] and pool.size == 8

    # A group that fills the pool short of its cap without the lowest-ranked displaces nothing.
    assert pooled(8, candidate("a", 0.25, rollouts=6)).add(candidate("b", 0.1)) == []

    selection = pool.select(8)
    assert selection.dropped == {"displaced": 3} and selection.metrics()["selection/displaced"] == 3.0


def test_a_dropped_group_leaves_the_pool_once_none_of_its_traces_are_queued():
    pool = pooled(100, candidate("old", 0.25), candidate("new", 0.1))

    assert pool.drop(["old0", "old1"], "stale") == []
    assert [(c.group_id, ids) for c, ids in pool.drop(["old2", "old3", "unknown"], "stale")] == [
        ("old", ["old2", "old3"])
    ]
    assert [c.group_id for c in pool.ranked()] == ["new"] and pool.size == 4
    assert pool.select(4).metrics()["selection/stale"] == 1.0
