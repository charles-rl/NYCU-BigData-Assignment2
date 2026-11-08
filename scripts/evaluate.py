"""Check the format of your output file and, if you have the answers, compute the score.

    python scripts/evaluate.py --pred output_{studentID}.csv                                    # format check only
    python scripts/evaluate.py --pred output_{studentID}.csv --input submission/test_input.csv  # + the UserIDs match the test list
    python scripts/evaluate.py --pred YOUR_PREDICTION_FILE --truth YOUR_ANSWER_FILE             # + score against the answers

    output_{studentID}.csv  your output file, e.g. output_314551051.csv
    YOUR_PREDICTION_FILE    your predictions for a validation week
    YOUR_ANSWER_FILE        the answers for that week

The answer file (--truth) has the same format as the output file. You can make one yourself
from the tables, for a week you hold out for validation.

Score = (macro-F1 + micro-F1) / 2, over the 6 activities. Activities listed with --skip are
not scored (used when GH Archive is missing data for that activity in the test week).
"""
import argparse
import sys

import pandas as pd

USER_COL = "UserID"
EVENT_COLS = ["open_pr", "open_issue", "issue_comment", "pr_review", "pr_review_comment", "push"]
OUTPUT_COLUMNS = [USER_COL] + EVENT_COLS


class FormatError(Exception):
    pass


def read_predictions(path, expected_users=None):
    """Read an output file and check its format. Raises FormatError on any problem."""
    try:
        df = pd.read_csv(path, dtype=str)
    except Exception as e:
        raise FormatError(f"cannot read {path} as CSV: {e}")
    if list(df.columns) != OUTPUT_COLUMNS:
        raise FormatError(f"header must be exactly {','.join(OUTPUT_COLUMNS)}; got {','.join(df.columns)}")
    if not len(df):
        raise FormatError("no prediction rows")
    if df.isna().any().any():
        raise FormatError("empty cells are not allowed")
    for col in df.columns:
        if not df[col].str.fullmatch(r"\d+").all():
            raise FormatError(f"column {col} must contain only non-negative integers")
    df = df.astype("int64")
    bad = ~df[EVENT_COLS].isin([0, 1])
    if bad.any().any():
        raise FormatError(f"prediction columns must be 0 or 1 (found other values in "
                          f"{', '.join(c for c in EVENT_COLS if bad[c].any())})")
    if df[USER_COL].duplicated().any():
        raise FormatError(f"duplicate UserIDs: {df[USER_COL][df[USER_COL].duplicated()].head().tolist()}")
    if expected_users is not None:
        got, want = set(df[USER_COL]), set(expected_users)
        if got != want:
            raise FormatError(f"UserIDs differ from the list: {len(want - got)} missing, "
                              f"{len(got - want)} unexpected")
    return df


def f1(tp, fp, fn):
    return 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 1.0


def scores(pred, truth, events=EVENT_COLS):
    """macro-F1 (mean of the F1 of each activity), micro-F1 (F1 of all activities pooled)
    and Score = their mean, over the given activities."""
    p = pred.set_index(USER_COL).loc[truth[USER_COL], events].to_numpy()
    t = truth[events].to_numpy()
    tp = ((p == 1) & (t == 1)).sum(0)
    fp = ((p == 1) & (t == 0)).sum(0)
    fn = ((p == 0) & (t == 1)).sum(0)
    per_event = {e: float(f1(*c)) for e, c in zip(events, zip(tp, fp, fn))}
    macro = sum(per_event.values()) / len(events)
    micro = float(f1(tp.sum(), fp.sum(), fn.sum()))
    return {"per_event": per_event, "macro_f1": macro, "micro_f1": micro, "score": (macro + micro) / 2}


def main():
    ap = argparse.ArgumentParser(description="Check an output file and compute its score.")
    ap.add_argument("--pred", required=True, help="your output CSV")
    ap.add_argument("--input", help="test input CSV (one UserID column): check the UserIDs match")
    ap.add_argument("--truth", help="answer CSV, same format as the output: check the UserIDs and score")
    ap.add_argument("--skip", nargs="*", default=[], choices=EVENT_COLS, metavar="ACTIVITY",
                    help="activities that are not scored")
    args = ap.parse_args()

    expected = None
    if args.truth:
        truth = pd.read_csv(args.truth)
        expected = truth[USER_COL].tolist()
    elif args.input:
        expected = pd.read_csv(args.input)[USER_COL].tolist()

    try:
        pred = read_predictions(args.pred, expected)
    except FormatError as e:
        print(f"Format check: ERROR - {e}", file=sys.stderr)
        sys.exit(1)
    print(f"Format check: no errors ({len(pred)} users)")

    if args.truth:
        events = [e for e in EVENT_COLS if e not in args.skip]
        if not events:
            sys.exit("Every activity is skipped; nothing to score.")
        s = scores(pred, truth, events)
        for e in EVENT_COLS:
            print(f"  {e:<18} " + (f"F1 {100 * s['per_event'][e]:6.2f}%" if e in events else "not scored"))
        print(f"macro-F1: {100 * s['macro_f1']:.2f}%")
        print(f"micro-F1: {100 * s['micro_f1']:.2f}%")
        print(f"Score:    {100 * s['score']:.2f}%")


if __name__ == "__main__":
    main()
