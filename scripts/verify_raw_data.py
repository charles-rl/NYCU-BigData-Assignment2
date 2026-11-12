"""Audit the raw GH Archive downloads hour by hour.

Why this exists: the raw files are not uniformly complete. Two failure modes
both pass a naive "file exists and is > 1 KB" check:

  * volume - an hour decompresses to a few thousand events instead of ~150k.
  * mix    - an hour is full-sized but a whole event type is absent from it,
             and the others are inflated to fill the space. This is the one that
             matters: in 2026-08 the corpus still carries 167k events/hour but
             only ~110 `PullRequestEvent`s (vs ~18,000 in 2025-12), so five of
             the six modelled activities have effectively no rows.

The mix test is therefore per activity, not a PushEvent-share proxy. A one-sided
`push_share >= 0.30` check passes a 97%-push hour whose PR-ish types are at
0.0%, which is exactly the hour you must not train on.

For each of the six modelled activities this script measures the density
(events of that type / total events) against a reference period, and marks the
hour unusable for that activity when the density collapses. Because the loss is
activity-specific, usability is reported per (day, activity), so a stretch that
lost only `PullRequestEvent`s still contributes `push` rows.

Output: coverage.csv (one row per hour, one usable flag per activity) and
coverage_activity.csv (one row per day, one usable flag per activity).

Run from the repository root:
    python -m scripts.verify_raw_data --workers 32
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

RAW_DIR = Path("raw_data")

# The six modelled activities mapped to the raw event type whose density stands
# in for them. `open_pr`/`open_issue` are approximated by their event types
# (PullRequestEvent also carries closes and merges, IssuesEvent all actions);
# that is fine here because the test asks "is this type being dropped?", not
# "what is the opened-only rate?".
ACTIVITY_EVENTS = {
    "open_pr": "PullRequestEvent",
    "open_issue": "IssuesEvent",
    "issue_comment": "IssueCommentEvent",
    "pr_review": "PullRequestReviewEvent",
    "pr_review_comment": "PullRequestReviewCommentEvent",
    "push": "PushEvent",
}
EVENT_TYPES = (
    "CommitCommentEvent",
    "CreateEvent",
    "DeleteEvent",
    "DiscussionEvent",
    "ForkEvent",
    "GollumEvent",
    "IssueCommentEvent",
    "IssuesEvent",
    "MemberEvent",
    "PublicEvent",
    "PullRequestEvent",
    "PullRequestReviewEvent",
    "PullRequestReviewThreadEvent",
    "PullRequestReviewCommentEvent",
    "PushEvent",
    "ReleaseEvent",
    "SponsorshipEvent",
    "WatchEvent",
)


def count_file(path: str) -> dict[str, object]:
    """Total events, per-type counts, and distinct actors for the six types.

    The actor count exists because density alone does not catch every kind of
    damage: 2026-08 carries a healthy `PushEvent` density yet reports 10.3 pushes
    per actor against 2.8 in 2025-12, i.e. the same few accounts repeated. A
    density test passes that hour; a distinct-actor test does not.
    """
    counts: Counter[str] = Counter()
    actors: dict[str, set[object]] = {t: set() for t in ACTIVITY_EVENTS.values()}
    total = 0
    bad = 0
    try:
        with gzip.open(path, "rt", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                total += 1
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    bad += 1
                    continue
                event_type = str(payload.get("type"))
                counts[event_type] += 1
                if event_type in actors:
                    actor = payload.get("actor")
                    if isinstance(actor, dict):
                        actors[event_type].add(actor.get("id"))
    except (OSError, EOFError) as exc:  # truncated or unreadable gzip member
        return {"file": Path(path).name, "error": f"{type(exc).__name__}: {exc}"}

    name = Path(path).name
    match = re.match(r"(\d{4}-\d{2}-\d{2})-(\d+)\.json\.gz$", name)
    row: dict[str, object] = {
        "file": name,
        "day": match.group(1) if match else name,
        "hour": int(match.group(2)) if match else -1,
        "bytes": Path(path).stat().st_size,
        "events": total,
        "unparsed": bad,
    }
    for event_type in EVENT_TYPES:
        row[event_type] = counts.get(event_type, 0)
    for event_type in ACTIVITY_EVENTS.values():
        row[f"{event_type}_actors"] = len(actors[event_type])
    row["other"] = sum(v for k, v in counts.items() if k not in EVENT_TYPES)
    return row


def reference_densities(
    frame: pd.DataFrame,
    month: str | None,
    quantile: float,
    min_volume_share: float,
) -> dict[str, float]:
    """Per-activity events/total-events density used as the "healthy" yardstick.

    Two guards matter here, both learned the hard way:

    1. Gate on volume *before* measuring density. A 3k-event hour made of 29%
       PullRequestEvents looks denser than any real hour, and an ungated 90th
       percentile lands on exactly those crumbs (39% for `open_pr`, against 11%
       in the healthiest month) which then rejects the good hours.
    2. Take the reference from *monthly medians*, not from raw hours. Density
       drifts monotonically through the corpus, so the median of all hours
       describes the damage rather than the healthy state. With `month` unset
       the reference is the `quantile` of the per-month medians, which resolves
       to the healthiest months; with `month` set it is that month's median.
    """
    usable = cast(pd.DataFrame, frame.loc[frame["events"].notna()])
    events = cast(pd.Series, usable["events"]).astype(float)
    median_events = float(events.median())
    usable = cast(pd.DataFrame, usable.loc[events >= min_volume_share * median_events])
    if usable.empty:
        raise SystemExit("no hour clears the volume gate; cannot build a reference")

    if month:
        usable = cast(pd.DataFrame, usable.loc[usable["day"].str[:7] == month])
        if usable.empty:
            raise SystemExit(f"reference month {month} has no usable hours")

    usable = cast(
        pd.DataFrame,
        usable.assign(month=usable["day"].astype(str).str.slice(0, 7)),
    )
    out: dict[str, float] = {}
    for activity, event_type in ACTIVITY_EVENTS.items():
        density = (
            cast(pd.Series, usable[event_type]).astype(float).to_numpy()
            / cast(pd.Series, usable["events"]).astype(float).clip(lower=1.0).to_numpy()
        )
        per_month = (
            pd.DataFrame({"month": usable["month"].to_numpy(), "density": density})
            .groupby("month")["density"]
            .median()
        )
        out[activity] = float(np.quantile(per_month.to_numpy(), quantile))
    return out


def reference_actors(
    frame: pd.DataFrame, reference: dict[str, float], min_volume_share: float
) -> dict[str, float]:
    """Median pushes-per-actor (etc.) among hours whose density looks healthy.

    Only hours that already pass the density test feed this, so a degraded hour
    cannot set the bar for what "normal repetition" means.
    """
    usable = cast(pd.DataFrame, frame.loc[frame["events"].notna()])
    events = cast(pd.Series, usable["events"]).astype(float)
    usable = cast(
        pd.DataFrame,
        usable.loc[events >= min_volume_share * float(events.median())],
    )
    out: dict[str, float] = {}
    for activity, event_type in ACTIVITY_EVENTS.items():
        observed = cast(pd.Series, usable[event_type]).astype(float)
        actors = cast(pd.Series, usable[f"{event_type}_actors"]).astype(float)
        healthy = observed >= 0.30 * reference[activity] * cast(
            pd.Series, usable["events"]
        ).astype(float)
        ratio = (observed / actors.clip(lower=1.0)).loc[healthy]
        out[activity] = float(ratio.median()) if len(ratio) else 1.0
    return out


def local_densities(
    frame: pd.DataFrame,
    reference: dict[str, float],
    window: int,
    min_volume_share: float,
) -> pd.DataFrame:
    """Per-(day, activity) yardstick taken from the surrounding days.

    The corpus-wide reference above answers "how dense is this activity at its
    healthiest", which is the wrong question for detecting *loss*: densities
    fall monotonically by more than an order of magnitude through the year, so a
    uniform week in a sparse regime compares badly against the best month and is
    rejected wholesale. Running the decay analysis against that reference left
    5 of 6 activities with **zero** usable anchors.

    Loss is a *local* discontinuity, so the yardstick is the median density of
    the same activity over the surrounding `window` days, excluding the day under
    test. A regime shift moves the neighbours with it and cancels out; a zero-push
    day or a dump does not. Days at the corpus edge have too few neighbours and
    fall back to the global reference.
    """
    usable = cast(pd.DataFrame, frame.loc[frame["events"].notna()]).copy()
    events = cast(pd.Series, usable["events"]).astype(float)
    usable = cast(
        pd.DataFrame, usable.loc[events >= min_volume_share * events.median()]
    )
    if usable.empty:
        return pd.DataFrame(columns=pd.Index(list(ACTIVITY_EVENTS)))
    day_key = pd.to_datetime(usable["day"]).dt.normalize()
    grouped = usable.groupby(day_key, sort=True)
    totals = cast(pd.Series, grouped["events"].sum()).astype(float)
    densities = {
        activity: cast(pd.Series, grouped[event_type].sum()).astype(float)
        / totals.clip(lower=1.0)
        for activity, event_type in ACTIVITY_EVENTS.items()
    }
    table = pd.DataFrame(densities)
    index = pd.DatetimeIndex(table.index)
    values = table.to_numpy(dtype=float)
    rows: list[np.ndarray] = []
    for i, day in enumerate(index):
        lo = index.searchsorted(day - pd.Timedelta(days=window), side="left")
        hi = index.searchsorted(day + pd.Timedelta(days=window), side="right")
        neighbours = np.concatenate([values[lo:i], values[i + 1 : hi]])
        if neighbours.size == 0:
            rows.append(np.array([reference[a] for a in ACTIVITY_EVENTS], dtype=float))
        else:
            rows.append(np.nanmedian(neighbours, axis=0))
    local = pd.DataFrame(rows, index=index, columns=pd.Index(list(ACTIVITY_EVENTS)))
    # Activities that never appear near this day fall back to the global value.
    fallback = pd.Series(reference)
    return cast(pd.DataFrame, local.fillna(fallback))


def add_activity_flags(
    frame: pd.DataFrame,
    reference: dict[str, float],
    actor_reference: dict[str, float],
    min_volume_share: float,
    min_activity_share: float,
    max_activity_ratio: float,
    max_actor_repetition: float,
    local: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Boolean `ok_<activity>` per hour from a density band and a repetition test.

    The density test is two-sided. Too little of an activity is loss; too much is
    a redistribution dump -- 2026-09-04..09-06 holds 1.4-1.7k PullRequestEvents
    per day against a ~110/day neighbourhood while pushes vanish, which a
    lower-bound-only test happily accepts and then feeds into the curve.
    """
    usable = cast(pd.DataFrame, frame.loc[frame["events"].notna()]).copy()
    events = cast(pd.Series, usable["events"]).astype(float)
    median_events = float(events.median())
    volume_ok = events >= min_volume_share * median_events
    if local is not None and not local.empty:
        day_key = pd.DatetimeIndex(pd.to_datetime(usable["day"]).dt.normalize())
        expected_density = cast(pd.DataFrame, local.reindex(day_key)).to_numpy(
            dtype=float
        )
        expected_density = np.where(
            np.isfinite(expected_density),
            expected_density,
            np.array([reference[a] for a in ACTIVITY_EVENTS], dtype=float),
        )
    else:
        expected_density = np.tile(
            np.array([reference[a] for a in ACTIVITY_EVENTS], dtype=float),
            (len(usable), 1),
        )
    for j, (activity, event_type) in enumerate(ACTIVITY_EVENTS.items()):
        observed = cast(pd.Series, usable[event_type]).astype(float).to_numpy()
        actors = (
            cast(pd.Series, usable[f"{event_type}_actors"]).astype(float).to_numpy()
        )
        expected = expected_density[:, j] * events.clip(lower=1.0).to_numpy()
        enough = observed >= min_activity_share * expected
        not_dumped = observed <= max_activity_ratio * expected
        # Few actors emitting many events means the samples were concentrated on
        # some subset rather than drawn faithfully; such an hour over-represents
        # those users and silently turns everyone else into a negative label.
        repetition = observed / np.clip(actors, 1.0, None)
        normal = repetition <= max_actor_repetition * actor_reference[activity]
        usable[f"ok_{activity}"] = volume_ok.to_numpy() & enough & not_dumped & normal
    return usable


def summarise(
    usable: pd.DataFrame,
    reference: dict[str, float],
    min_volume_share: float,
    local: pd.DataFrame | None = None,
    min_activity_share: float = 0.30,
    max_activity_ratio: float = 4.0,
    min_day_hours_share: float = 0.75,
) -> pd.DataFrame:
    """One row per day x activity, so a partially-good day is not discarded."""
    rows: list[dict[str, object]] = []
    local_index = None if local is None or local.empty else local.index
    for day, block in usable.groupby("day", sort=True):
        day_ts = pd.Timestamp(str(day))
        for activity, event_type in ACTIVITY_EVENTS.items():
            observed = float(cast(pd.Series, block[event_type]).sum())
            events = float(cast(pd.Series, block["events"]).sum())
            hours = len(block)
            expected_density = reference[activity]
            if local is not None and local_index is not None and day_ts in local_index:
                value = float(local.loc[day_ts, activity])
                if np.isfinite(value):
                    expected_density = value
            hours_ok = int(cast(pd.Series, block[f"ok_{activity}"]).sum())
            ratio = observed / max(expected_density * events, 1.0)
            rows.append(
                {
                    "day": day,
                    "activity": activity,
                    "hours": hours,
                    "hours_ok": hours_ok,
                    "hours_ok_share": hours_ok / max(hours, 1),
                    "events": int(events),
                    "observed": int(observed),
                    "density": observed / max(events, 1.0),
                    "reference_density": expected_density,
                    "density_ratio": ratio,
                    # `clean` demands a flawless day. That is the right bar for a
                    # validation anchor but unreachable in a sparse regime: with
                    # 19/24 hours typical, P(24/24) is under 1%, so a whole month
                    # reads as unusable. `usable` is the day-level band a curve
                    # actually needs -- the day's total is within
                    # `min_activity_share .. max_activity_ratio` of its local
                    # baseline and most of its hours were intact.
                    "clean": hours_ok == hours,
                    "usable": (
                        ratio >= min_activity_share
                        and ratio <= max_activity_ratio
                        and hours_ok >= min_day_hours_share * hours
                    ),
                }
            )
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--out", type=Path, default=RAW_DIR / "coverage.csv")
    parser.add_argument(
        "--activity-out", type=Path, default=RAW_DIR / "coverage_activity.csv"
    )
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument(
        "--min-volume-share",
        type=float,
        default=0.25,
        help="an hour needs this share of the median event count",
    )
    parser.add_argument(
        "--min-activity-share",
        type=float,
        default=0.30,
        help="an hour needs this share of the reference density for an activity",
    )
    parser.add_argument(
        "--max-activity-ratio",
        type=float,
        default=4.0,
        help="reject an hour carrying more than this multiple of the local "
        "density (catches redistribution dumps)",
    )
    parser.add_argument(
        "--min-day-hours-share",
        type=float,
        default=0.75,
        help="share of a day's hours that must be intact for the day to be "
        "usable for that activity (`clean` still requires all of them)",
    )
    parser.add_argument(
        "--local-window",
        type=int,
        default=7,
        help="days of neighbourhood used for the per-day density baseline "
        "(0 disables, falling back to the corpus reference)",
    )
    parser.add_argument(
        "--reference-month",
        default=None,
        help="YYYY-MM to take reference densities from (default: corpus quantile)",
    )
    parser.add_argument(
        "--reference-quantile",
        type=float,
        default=0.90,
        help="quantile of the monthly medians used when --reference-month is unset",
    )
    parser.add_argument(
        "--max-actor-repetition",
        type=float,
        default=2.0,
        help="reject an hour whose events-per-actor exceeds this multiple of "
        "the healthy reference (catches hours built from a few repeated accounts)",
    )
    args = parser.parse_args()

    files = sorted(args.raw_dir.glob("*.json.gz"))
    if not files:
        print(f"no *.json.gz under {args.raw_dir}")
        return 1
    print(f"scanning {len(files)} hour files with {args.workers} workers ...")

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(count_file, [str(f) for f in files], chunksize=4))

    frame = cast(pd.DataFrame, pd.DataFrame(rows)).sort_values(["day", "hour"])
    frame = cast(pd.DataFrame, frame.reset_index(drop=True))

    if "error" in frame.columns:
        failed = cast(pd.DataFrame, frame.loc[frame["error"].notna()])
        if len(failed):
            print(f"\n{len(failed)} file(s) failed to decompress:")
            for _, row in failed.head(20).iterrows():
                print(f"  {row['file']}: {row['error']}")

    readable = int(cast(pd.Series, frame["events"].notna()).sum())
    if readable == 0:
        print("no readable files")
        return 1

    reference = reference_densities(
        frame,
        args.reference_month,
        args.reference_quantile,
        args.min_volume_share,
    )
    actor_reference = reference_actors(frame, reference, args.min_volume_share)
    local = (
        local_densities(frame, reference, args.local_window, args.min_volume_share)
        if args.local_window > 0
        else None
    )
    usable = add_activity_flags(
        frame,
        reference,
        actor_reference,
        args.min_volume_share,
        args.min_activity_share,
        args.max_activity_ratio,
        args.max_actor_repetition,
        local,
    )
    usable.to_csv(args.out, index=False)

    per_day = summarise(
        usable,
        reference,
        args.min_volume_share,
        local,
        args.min_activity_share,
        args.max_activity_ratio,
        args.min_day_hours_share,
    )
    per_day.to_csv(args.activity_out, index=False)

    events = cast(pd.Series, usable["events"]).astype(float)
    print(f"\nwrote {args.out} and {args.activity_out}")
    print(f"hours scanned        : {len(usable):,}")
    print(f"median events/hour   : {float(events.median()):,.0f}")
    print("\nreference density (events of type / total events):")
    for activity, density in reference.items():
        print(
            f"  {activity:>20}  {density * 100:6.2f}%   "
            f"healthy events/actor {actor_reference[activity]:.2f}"
        )

    print("\nusable hours per activity:")
    for activity in ACTIVITY_EVENTS:
        col = f"ok_{activity}"
        n_ok = int(cast(pd.Series, usable[col]).sum())
        print(
            f"  {activity:>20}  {n_ok:5,d}/{len(usable):,} "
            f"({n_ok / len(usable) * 100:5.1f}%)"
        )

    print("\ndays per activity (usable = day-level band, clean = every hour):")
    for activity in ACTIVITY_EVENTS:
        block = cast(pd.DataFrame, per_day.loc[per_day["activity"] == activity])
        usable_block = cast(pd.DataFrame, block.loc[block["usable"].astype(bool)])
        clean = cast(pd.DataFrame, block.loc[block["clean"].astype(bool)])
        span = (
            f"  span {usable_block['day'].min()} .. {usable_block['day'].max()}"
            if len(usable_block)
            else ""
        )
        print(
            f"  {activity:>20}  usable {len(usable_block):4d}"
            f"  clean {len(clean):4d}/{len(block):4d}{span}"
        )

    print("\nby month (clean days per activity):")
    per_day = cast(
        pd.DataFrame, per_day.assign(month=per_day["day"].astype(str).str[:7])
    )
    table = per_day.pivot_table(
        index="month", columns="activity", values="clean", aggfunc="sum"
    )
    days_pm = cast(pd.Series, per_day.groupby("month")["day"].nunique())
    counts = cast(pd.DataFrame, table.reindex(columns=list(ACTIVITY_EVENTS)))
    counts = cast(
        pd.DataFrame,
        counts.assign(days=[int(days_pm[m]) for m in counts.index]),
    )
    table = counts[["days", *ACTIVITY_EVENTS]]
    print(table.to_string())

    all_clean = per_day.groupby("day")["clean"].all()
    print(
        f"\ndays clean for all six activities: {int(all_clean.sum())}/{len(all_clean)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
