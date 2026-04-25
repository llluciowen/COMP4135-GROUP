"""Analyse A/B test events from ab_events.jsonl.

Usage:
    python analyze_ab_events.py path/to/ab_events.jsonl

Outputs to stdout:
    1. Cohort sizes per (ui_variant, algo_variant)
    2. Per-cohort metrics: CTR, Like Rate, avg ratings, avg session time
    3. Mann-Whitney U tests for the two required A/B comparisons:
       - Experiment 1 (algorithm): UI=A, Algo-A vs Algo-B
       - Experiment 2 (UI):       Algo=B, UI-A vs UI-B
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

try:
    from scipy.stats import mannwhitneyu
except ImportError:
    mannwhitneyu = None


def load_events(path: Path) -> list[dict]:
    events = []
    with path.open(encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return events


def parse_ts(ts: str) -> datetime:
    return datetime.fromisoformat(ts)


def group_by_user(events: list[dict]) -> dict[tuple[str, str, str], list[dict]]:
    """Group events by (ab_seed, ui_variant, algo_variant) — one cell per user-cohort.

    A single ab_seed (browser) can switch variants via URL params, so we key on the
    full triple. Events without ab_seed (eg page_view) are attached to the closest
    surrounding cohort if a seed is in scope, otherwise dropped.
    """
    by_user: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    last_seed = None
    for e in events:
        seed = e.get('ab_seed') or last_seed
        if e.get('ab_seed'):
            last_seed = e['ab_seed']
        ui = e.get('ui_variant')
        algo = e.get('algo_variant')
        if not (seed and ui and algo):
            continue
        by_user[(seed, ui, algo)].append(e)
    return by_user


def compute_user_metrics(events: list[dict]) -> dict[str, float]:
    """Per-user metrics within one cohort cell."""
    likes = sum(1 for e in events if e['event'] == 'like_toggled' and e.get('liked'))
    ratings = sum(1 for e in events if e['event'] == 'rating_updated')
    refreshes = sum(1 for e in events if e['event'] == 'async_refresh')
    filters = sum(1 for e in events if e['event'] == 'filter_changed')
    explanations = sum(1 for e in events if e['event'] == 'attention_explanation_opened')

    # Rough recommendation impressions: each async_refresh re-renders the rec list.
    impressions = max(refreshes, 1)
    ctr = filters / impressions  # interaction with the result list as a CTR proxy
    like_rate = likes / impressions

    timestamps = sorted(parse_ts(e['timestamp']) for e in events)
    duration_s = (timestamps[-1] - timestamps[0]).total_seconds() if len(timestamps) > 1 else 0.0

    return {
        'events': len(events),
        'ratings': ratings,
        'likes': likes,
        'refreshes': refreshes,
        'filters': filters,
        'explanations': explanations,
        'ctr_proxy': ctr,
        'like_rate': like_rate,
        'duration_s': duration_s,
    }


def aggregate_cohort(per_user: list[dict[str, float]]) -> dict[str, float]:
    if not per_user:
        return {}
    keys = per_user[0].keys()
    out = {'n_users': len(per_user)}
    for k in keys:
        out[f'avg_{k}'] = sum(u[k] for u in per_user) / len(per_user)
    return out


def print_table(rows: list[tuple]):
    if not rows:
        print('  (no data)')
        return
    widths = [max(len(str(r[i])) for r in rows) for i in range(len(rows[0]))]
    for r in rows:
        print('  ' + '  '.join(str(c).ljust(widths[i]) for i, c in enumerate(r)))


def run_test(name: str, group_a: list[float], group_b: list[float], label_a: str, label_b: str):
    print(f"\n=== {name} ===")
    print(f"  {label_a}: n={len(group_a)} mean={sum(group_a)/max(len(group_a),1):.3f}")
    print(f"  {label_b}: n={len(group_b)} mean={sum(group_b)/max(len(group_b),1):.3f}")
    if mannwhitneyu is None:
        print('  (install scipy to get p-values: pip install scipy)')
        return
    if len(group_a) < 2 or len(group_b) < 2:
        print('  (need >=2 users per group for stats test)')
        return
    try:
        stat, p = mannwhitneyu(group_a, group_b, alternative='two-sided')
        print(f"  Mann-Whitney U={stat:.1f}  p={p:.4f}  "
              f"{'**SIGNIFICANT**' if p < 0.05 else '(not significant)'}")
    except ValueError as exc:
        print(f"  test failed: {exc}")


def main():
    if len(sys.argv) < 2:
        print('Usage: python analyze_ab_events.py path/to/ab_events.jsonl')
        sys.exit(1)
    path = Path(sys.argv[1])
    events = load_events(path)
    print(f"Loaded {len(events)} events from {path}")

    by_user = group_by_user(events)
    print(f"Distinct (user, ui, algo) cells: {len(by_user)}")

    # Aggregate per cohort
    cohorts: dict[tuple[str, str], list[dict[str, float]]] = defaultdict(list)
    for (seed, ui, algo), evs in by_user.items():
        cohorts[(ui, algo)].append(compute_user_metrics(evs))

    print('\n========== Cohort Summary ==========')
    headers = ('UI/Algo', 'n_users', 'avg_ratings', 'avg_likes', 'avg_filters',
               'avg_like_rate', 'avg_duration_s')
    rows = [headers]
    for key in sorted(cohorts.keys()):
        agg = aggregate_cohort(cohorts[key])
        rows.append((
            f"{key[0]}/{key[1]}",
            agg.get('n_users', 0),
            f"{agg.get('avg_ratings', 0):.2f}",
            f"{agg.get('avg_likes', 0):.2f}",
            f"{agg.get('avg_filters', 0):.2f}",
            f"{agg.get('avg_like_rate', 0):.3f}",
            f"{agg.get('avg_duration_s', 0):.1f}",
        ))
    print_table(rows)

    # Required A/B tests
    def metric(cohort_key, m):
        return [u[m] for u in cohorts.get(cohort_key, [])]

    print('\n========== Required A/B Tests ==========')

    # Experiment 1: UI fixed = A, compare algorithms
    run_test('Experiment 1: Algorithm (UI=A) — like_rate',
             metric(('A', 'A'), 'like_rate'), metric(('A', 'B'), 'like_rate'),
             'UI=A Algo=A (KNN)', 'UI=A Algo=B (SASRec)')
    run_test('Experiment 1: Algorithm (UI=A) — likes count',
             metric(('A', 'A'), 'likes'), metric(('A', 'B'), 'likes'),
             'UI=A Algo=A (KNN)', 'UI=A Algo=B (SASRec)')

    # Experiment 2: Algo fixed = B, compare UIs
    run_test('Experiment 2: UI (Algo=B) — likes count',
             metric(('A', 'B'), 'likes'), metric(('B', 'B'), 'likes'),
             'UI=A Algo=B', 'UI=B Algo=B')
    run_test('Experiment 2: UI (Algo=B) — duration_s',
             metric(('A', 'B'), 'duration_s'), metric(('B', 'B'), 'duration_s'),
             'UI=A Algo=B', 'UI=B Algo=B')
    run_test('Experiment 2: UI (Algo=B) — filters used',
             metric(('A', 'B'), 'filters'), metric(('B', 'B'), 'filters'),
             'UI=A Algo=B', 'UI=B Algo=B')


if __name__ == '__main__':
    main()
